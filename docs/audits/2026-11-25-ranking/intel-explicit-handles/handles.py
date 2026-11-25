"""Offline identity and per-process lifecycle qualification for explicit oneMKL SGEMM handles."""
import csv
import hashlib
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

HANDLES = ('mkl-handle-oneshot', 'mkl-handle-prepared')
STATES = {0:'MKL_JIT_SUCCESS', 1:'MKL_NO_JIT', 2:'MKL_JIT_ERROR'}
KEY = ('method','family','layout','mode','seed','round')
SDK = {
    'libmkl_intel_lp64.so.3':'eb5a6748ec4e85dc07bd4dbf860af21e86f26647dbba837a94338a8289bf1faa',
    'libmkl_sequential.so.3':'812f56b8de337f2bc591426462232a4b114a443b284dabfb985e9af0c39f32be',
    'libmkl_core.so.3':'2b7ed4833447281eec1edee574115f663169dec03d38c7cd2ed8fe2db8c0861e',
}
SOURCE_PINS = {
    'vendor_backend.c':'dea6e72836e4cceb42148af9a379e61b509475569fa94742a33452371ed72536',
    'vendor_adapter.c':'56109f058436a5ede403cd70a7ea294c8cd0cbb866e9759404eb028c57593596',
}
HANDLES_CORE_PINS = {
    'source/handle.c':'3cd99ee6ca7da943094ef0d1a4c9875c21b2dd5e77eb831b51ed261b127e767a',
    'source/observe.c':'32713054440abb9e6bdff51a5cccd432c488ab0d0e165054d007155cd0434ae7',
    'source/fixture.c':'50f2f7c018dfb7a8583b59c750d0fde6997bbfbab957777689405c738ad314fa',
    'source/handle.h':'601cf7ddd3a4ab93d1e0ecba46e00e9894f5fd50910ffe6ff7aafd004153dadc',
    'source/plan-cost.c':'b3af7a306181348fd93fffc93da76d18dda41aca24380799a8197be2540e1896',
}
HEADERS = {
    'mkl_cblas.h':'1aa51dc501e06db3f758b499ec550e43d25a982faa4253c52882ab3d0e8cec08',
    'mkl_service.h':'02ff230034b626727576886decbdce2a01b76728e59b9d708e884d5bd153e2c7',
    'mkl_types.h':'0039030497854ba329fe20f6a89bef50c9ab3901f44a1aaef39fbc842a1a0cc4',
}
DIGEST=re.compile(r'[a-f0-9]{64}')


def need(test, why):
    if not test:
        raise ValueError(why)


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def unique(rows, fields):
    out={}
    for row in rows:
        key=tuple(str(row[f]) for f in fields)
        need(key not in out, 'Duplicate native observation '+str(key))
        out[key]=row
    return out


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    need(DIGEST.fullmatch(value), 'Invalid SHA-256 identifier')
    return value


def checksum_list(path):
    out={}
    for line in path.read_text().splitlines():
        value,name=line.split(maxsplit=1)
        need(name not in out, 'Duplicate identity in '+path.name)
        out[name]=digest(value)
    return out


def observations(root,rawworkers,cpu,status_by_cell):
    raw=unique((r for r in rawworkers if r['method'] in HANDLES), KEY)
    handles=unique(read(root/'handle-observations.csv'),KEY)
    need(set(raw)==set(handles),'Timed handles lack corresponding observed execution status, or vice versa')
    result=defaultdict(list)
    for key, r in raw.items():
        method,family,layout,mode,seed,trial=key
        p=handles[key]
        prepared=method==HANDLES[1]
        initial,final,setup=map(int,(p['first_status'],p['last_status'],p['prepared_status']))
        expected=status_by_cell[(method,family,layout)]
        need(initial==final==expected and initial in (0,1), 'Measured native first/last operation and route disagree '+str(key))
        preparation=float(p['prepare_s'])
        need(setup == (initial if prepared else -1) and math.isfinite(preparation) and
             (preparation>0 if prepared else preparation==0), 'Preparation placement/states disagree '+str(key))
        first=float(r['first_s'])
        need(math.isfinite(first) and first>0 and int(r['cpu'])==int(cpu) and int(r['threads'])==1,
             'One pinned worker and finite first call required')
        result[(method,family,layout)].append(dict(mode=mode,seed=int(seed),round=int(trial),actual_status=STATES[initial],
            external_prepare_seconds=preparation,first_product_seconds=first,setup_plus_first_seconds=preparation+first))
    return result


def imports(path):
    return set(re.findall(r'^\s*U\s+(\w+)',path.read_text(),re.MULTILINE))


def substrate(root,methods):
    direct=checksum_list(root/'mkl-identity.sha256')
    need(all(direct[name]==value for name,value in {**SDK,**SOURCE_PINS}.items()) and
         direct['mkl.so']==methods['mkl']['sha256'], 'Ordinary oneMKL substrate differs from the field or pinned runtime')
    symbols=imports(root/'mkl-backend-imports.txt')
    need('cblas_sgemm' in symbols and 'dladdr' in symbols and not symbols.intersection(('mkl_jit_create_sgemm','mkl_sgemm_jit','sgemm_direct')),
         'Ordinary comparator did not use the normal CBLAS interface')
    return {'mkl_adapter_sha256':direct['mkl.so'],'mkl_backend_sha256':direct['mkl-backend.so'],
            'sdk_library_sha256':SDK,'ordinary_provider_version':'2026.2.1'}

def check_route_stages(panelrow, entries, method, cpu, page):
    prepared=method==HANDLES[1]
    stages=['first','new-pointers','changed-ldc','restore']
    if prepared:
        stages=['prepare',*stages,'release']
    need([r['stage'] for r in entries]==stages, 'Incomplete or reordered actual handle stages')
    normal_status=None
    max_error=0.
    checked=guards=0
    for r in entries:
        stage=r['stage']
        multiply=stage not in ('prepare','release')
        altered=stage=='changed-ldc'
        dims=[int(panelrow[k]) for k in ('m','n','k','lda','ldb','ldc')]
        if altered: dims[-1]+=1
        m,n,k,lda,ldb,ldc=dims
        need([int(r[f]) for f in ('m','n','k','lda','ldb','ldc')]==dims,'Handle was created or tested with wrong strides')
        need(int(r['cpu'])==int(cpu) and r['threads']=='1' and r['passed']=='1','Lifecycle worker affinity/passing proof')
        counts=(1,1,1,1)
        if prepared:
            counts={'prepare':(1,1,0,0),'first':(0,0,1,0),'new-pointers':(0,0,1,0),
                    'changed-ldc':(1,1,1,1),'restore':(1,1,1,1),'release':(0,0,0,1)}[stage]
        creates,gets,calls,destroys=counts
        need(tuple(int(r[f]) for f in ('creates','gets','calls','destroys'))==counts,'Unexpected actual handle lifecycle '+stage)
        status=int(r['ordinary_status'])
        need(status in (0,1) and int(r['trace_status'])==status and r['status_name']==STATES[status], 'Diagnostic/warmed SGEMM status')
        if not altered:
            if normal_status is None: normal_status=status
            need(normal_status==status,'Restored key did not reproduce the original dispatch')
        need((int(r['create_status']),int(r['execution_status']),int(r['destroy_status']))==
             (status if creates else -1, status if calls else -1, 0 if destroys else -1),'Wrong create/call/destroy outcome '+stage)
        if creates:
            need(tuple(int(r[k]) for k in ('order','transa','transb'))==(101,111,111) and float(r['alpha'])==1 and float(r['beta'])==0,
                 'Wrong observed explicit API operation')
        need(int(r['callable'])==int(bool(calls or gets)) and int(r['operands_checked'])==calls,'Missing actual kernel pointer or operands')
        if prepared and stage in ('first','new-pointers'):
            need(r['same_handle']==r['same_kernel']=='1','Prepared old handle/function pointer not reused')
        if prepared and stage in ('changed-ldc','restore'):
            need(r['same_handle']=='0','Geometry changed yet the old retained handle stayed')
        if gets or calls:
            need('x' in r['callable_permissions'] and r['callable_owner'] not in ('none','released',''),'Callable was not mapped executable')
        need(all(int(r[k])==int(panelrow[k]) for k in ('ar','ac','br','bc','cr','cc')), 'Wrong initialized parent origin')
        want=[4*(int(panelrow[x+'r'])*ld+int(panelrow[x+'c']))%64 for x,ld in zip('abc',(lda,ldb,ldc))]
        need([int(r[x+'_mod64']) for x in 'abc']==want,'Wrong tested input or output address residues')
        area=m*n if multiply else 0
        out_cells=(((int(panelrow['cr'])+m+2)*ldc*4+page-1)//page)*(page//4)-m*n if multiply else 0
        need(int(r['checked'])==area and int(r['guards'])==out_cells,'Incomplete route FP64/parent checks')
        errors=[float(r[k]) for k in ('ordinary_error','trace_error')]
        need(all(0<=e<=0.002 for e in errors),'Numerical route witness failed')
        max_error=max(max_error,*errors);checked+=area*2;guards+=out_cells*2
    return normal_status,dict(stages=len(stages),ordinary_plus_trace_output_checks=checked,
                              ordinary_plus_trace_guard_checks=guards,maximum_witness_error=max_error,
                              changed_ldc_status=next(r['status_name'] for r in entries if r['stage']=='changed-ldc'))

def proof_binaries(root,methods):
    source=checksum_list(root/'handle-source.sha256')
    api=checksum_list(root/'handle-provider.sha256')
    require_sources={f'source/{n}' for n in ('handle.c','handle.h','observe.c','fixture.c','plan-cost.c','build.sh','routes.py','route-worker.py')}
    need(require_sources<=set(source) and api=={**SDK,**HEADERS} and all(source.get(k)==v for k,v in HANDLES_CORE_PINS.items()),'Missing source or changed pinned MKL ABI')
    result=substrate(root,methods)
    binaries=unique(read(root/'handle-builds.csv'),('method',))
    need(set(binaries)=={(m,) for m in HANDLES},'Explicit handle ordinary and trace build inventory')
    checks={}
    for m in HANDLES:
        b=binaries[m,]
        need(b['source_sha256']==source['source/handle.c'] and b['observer_sha256']==source['source/observe.c'] and
             b['ordinary_sha256']==methods[m]['sha256'] and digest(b['trace_sha256'])!=digest(b['ordinary_sha256']),
             'Explicit worker or source mismatch '+m)
        need([b[x] for x in ('lp64_sha256','seq_sha256','core_sha256')]==list(SDK.values()),'MKL payloads differ '+m)
        targets={s.lower() for s in imports(root/(m+'.imports'))}
        need({'mkl_cblas_jit_create_sgemm','mkl_jit_get_sgemm_ptr','mkl_jit_destroy','mkl_get_max_threads'}<=targets and
             not targets.intersection(('cblas_sgemm','sgemm_','mkl_sgemm_jit','sgemm_direct')),
             'Ordinary adapter used a different compute ABI: '+m)
        ld=(root/(m+'.dependencies')).read_text()
        need(not re.search(r'not found|lib(?:iomp|gomp|mkl_\w*thread|tbb|openblas|blas\.so)',ld,re.I) and
             all(re.search(r'^\s*'+re.escape(library)+r'\s+=>\s+\S*'+re.escape(library)+r'\s+\(',ld,re.M) for library in SDK),
             'Adapter shared-library resolution '+m)
        symbols=set(re.findall(r'\bT\s+(\w+)$',(root/(m+'.symbols')).read_text(),re.MULTILINE))
        need('rank_handle_last_status' in symbols and 'rank_vendor_sgemm' in symbols and
             not any(sym.startswith('rank_handle_trace') for sym in symbols) and
             (('rank_handle_prepare' in symbols)==(m==HANDLES[1])) and (('rank_handle_release' in symbols)==(m==HANDLES[1])),
             'Ordinary method ABI, prepared token or diagnostic contamination')
        dynamic=(root/(m+'.dynamic')).read_text()
        needed=set(re.findall(r'\(NEEDED\).*?\[([^]]+)\]',dynamic))
        need({x for x in needed if x.startswith('libmkl')}==set(SDK),'ELF dependencies differ from pinned sequential MKL')
        checks[m]={'ordinary_sha256':b['ordinary_sha256'],'trace_sha256':b['trace_sha256'],'explicit_api':True}
    need(len({v[k] for v in checks.values() for k in ('ordinary_sha256','trace_sha256')})==4,'Collapsed handle modes/diagnostics')
    result.update(adapters=checks,handle_source_sha256=source['source/handle.c'],observer_sha256=source['source/observe.c'],
                  sdk_header_sha256={k:v for k,v in api.items() if k.endswith('.h')})
    return result


def recovery(root,cpu,unavailable=()):
    state=unique(read(root/'handle-faults.csv'),('method','scenario'))
    keys=unique(read(root/'handle-keys.csv'),('method','key','step'))
    need(set(state)=={(m,s) for m in HANDLES if m not in unavailable for s in ('create-error','null-callable')},'Missing simulated native API error test')
    for (m,s),r in state.items():
        null=s=='null-callable'
        need(int(r['cpu'])==cpu and int(r['status'])==(-3 if null else -2) and (r['recovered'],r['creates'],r['gets'],r['calls'],r['destroys'])
             ==('1','1',str(int(null)),'0',str(int(null))),f'Error and recovery witness {m} {s}')
    expected={(HANDLES[1],item,step) for item in ('m','n','k','lda','ldb','ldc') for step in ('change','restore')}
    if HANDLES[1] in unavailable: expected=set()
    need(set(keys)==expected,'Six-argument retained-handle key is not verified')
    for r in keys.values():
        need(int(r['cpu'])==cpu and r['status'] in ('0','1') and tuple(int(r[k]) for k in ('creates','gets','calls','destroys'))==(1,1,0,1),
             'A changed shape/stride key did not replace its ordinary prepared handle')
    return {'injected_error_recovery_cases':len(state),'changed_or_restored_geometry_keys':len(keys)}


def plan_costs(root,panel,cpu):
    owners={}
    for row in (root/'handle-api-owners.txt').read_text().splitlines():
        match=re.fullmatch(r'plan-cost (create|get|destroy) provider=(\S+)',row)
        need(match is not None and match[2] in SDK,'Unknown create/get/destroy API owner')
        need(match[1] not in owners,'Duplicate plan-only API soname')
        owners[match[1]]=match[2]
    errors=unique(read(root/'handle-plan-errors.csv'),('family','layout'))
    need(set(errors)<=set(panel),'Unknown plan-only failure shape')
    if len(errors)<len(panel): need(set(owners)=={'create','get','destroy'},'Missing actual plan-only API owner')
    samples=unique(read(root/'handle-plan-cost.csv'),('family','layout','sample'))
    expected={(*key,str(i)) for key in panel for i in range(16)}
    need(set(samples)<=expected,'Additional plan-only diagnostics')
    summary={}
    for key,p in panel.items():
        work=[]
        for i in range(16):
            if (*key,str(i)) not in samples:
                need(key in errors,'Missing plan-only observation without failure receipt')
                continue
            r=samples[(*key,str(i))]
            need(int(r['cpu'])==cpu and all(int(r[k])==int(p[k]) for k in ('m','n','k','lda','ldb','ldc')),
                 'Isolated plan-only sample tests a different shape/stride/CPU')
            times={k:int(r[k]) for k in ('create_ns','get_ns','destroy_ns','clock_pair_ns')}
            need(times['clock_pair_ns']>=0 and times['create_ns']>=0 and r['status'] in STATES.values(),'Invalid plan setup timer/status')
            callable_=r['callable']=='1'
            good=callable_ and r['status'] in (STATES[0],STATES[1]) and r['destroy_status']=='0'
            if good:
                need(min(times.values())>=0,'Invalid successful plan clocks')
            else:
                need(r['callable'] in ('0','1'),'Invalid callable flag')
                if r['status']==STATES[2]:
                    need(not callable_ and int(r['destroy_status'])==-1 and times['get_ns']==times['destroy_ns']==-1,'MKL_JIT_ERROR incorrectly callable')
            work.append(dict(**times,sample=i,status=r['status'],valid=good,callable=callable_,destroy_status=int(r['destroy_status'])))
        if key in errors:
            failure=errors[key]
            need(int(failure['samples'])==len(work) and failure['reason'] and (len(work)<16 or int(failure['returncode'])!=0), 'Unexplained plan-only measurement failure')
        else:
            need(len(work)==16 and all(x['valid'] for x in work),'Cost samples incomplete or invalid without reported failure')
        subsequent=[x for x in work if x['sample']>0 and x['valid']]
        summary[key[0]]={'samples':len(work),'valid':sum(x['valid'] for x in work),'first':next((x for x in work if x['sample']==0),None),'failure':errors.get(key,{}).get('reason',''),
                         'subsequent_valid_median_ns':{k:statistics.median(x[k] for x in subsequent) for k in ('create_ns','get_ns','destroy_ns','clock_pair_ns')} if subsequent else None,
                         'status_counts':{s:sum(x['status']==s for x in work) for s in STATES.values()}}
    return dict(actual_owners=owners,layouts=summary)

def verify_handles(root,panel,rawworkers,methods,cpu):
    """Return independently reconstructed explicit API status, call policy, setup costs and exclusions."""
    root=Path(root)
    panel=list(panel.values()) if isinstance(panel,dict) else list(panel)
    need(sha256(root/'panel.csv')=='2b4d6871262a4d038533b6cab819e819c3cff1eaf3ef606678b2ad6e30520ed5' and panel==read(root/'panel.csv'),
         'Explicit API panel is different from its frozen eleven physical views')
    cases=unique(panel,('family','layout'))
    methods={r['method']:r for r in (methods.values() if isinstance(methods,dict) else methods)}
    cpu=int(cpu)
    page=int((root/'page-size.txt').read_text())
    need(page>=64 and page&(page-1)==0,'Invalid native page size')
    for m in HANDLES:
        need(methods[m]['role']=='independent' and methods[m]['policy']==('one-shot' if m==HANDLES[0] else 'retained'),'Explicit handle policy declared incorrectly')
    binary=proof_binaries(root,methods)
    excl=unique(read(root/'handle-route-exclusions.csv'),('method','family','layout'))
    errors=unique(read(root/'handle-route-errors.csv'),('method','family','layout'))
    admit=unique((r for r in read(root/'admissions.csv') if r['method'] in HANDLES),('method','family','layout'))
    population={(m,*key) for m in HANDLES for key in cases}
    need(set(admit)==population and set(errors)<=set(excl)<=population,'Undeclared or unregistered native route failure')
    stages=defaultdict(list)
    for r in read(root/'handle-routes.csv'):
        stages[r['method'],r['family'],r['layout']].append(r)
    need(set(stages)==population-set(excl),'Missing routine witnesses without failure metadata, or excluded method reported valid')
    controls_bad=[m for m in HANDLES if all((m,*k) in excl and excl[(m,)+k]['reason'].startswith('method-control:') for k in cases)]
    controls=recovery(root,cpu,controls_bad)
    codes={}
    details={}
    for key in sorted(population):
        method,family,layout=key
        if key in excl:
            need(excl[key]['reason'] and admit[key]['qualified']=='0', 'Failed native handle was incorrectly timed or unexplained')
            raw_error=errors.get(key)
            if raw_error:
                need(int(raw_error['ordinary_status'])<0 or int(raw_error['trace_status'])<0,'Failure row lacks an actual status')
                creation=int(raw_error['create_status'])
                need(creation in (-1,0,1,2) and raw_error['reason'],'Invalid native error')
                if creation==2:
                    need(int(raw_error['creates'])==1 and int(raw_error['gets'])==int(raw_error['calls'])==0,
                         'MKL_JIT_ERROR incorrectly retrieved or called a handle')
                if int(raw_error['trace_status'])==-3:
                    need(int(raw_error['gets'])==1 and int(raw_error['calls'])==0 and int(raw_error['destroys'])==1,
                         'Null callable cleanup did not dispose of the uncallable handle')
            details[key]={'route_qualified':False,'reason':excl[key]['reason'],'reported_native_error':raw_error}
        else:
            code,data=check_route_stages(cases[family,layout],stages[key],method,cpu,page)
            codes[key]=code
            details[key]=dict(data,route_qualified=True,api_status=STATES[code],numerically_qualified=admit[key]['qualified']=='1')
    workers=observations(root,rawworkers,cpu,codes)
    count={}
    for key,entry in details.items():
        batch=workers.get(key,[])
        if admit[key]['qualified']=='1':
            need(len(batch)==10 and sorted(r['round'] for r in batch)==[-1]*4+list(range(6)), 'Missing validated worker setup durations')
        entry['workers']=batch
        method=key[0]
        if method not in count: count[method]={'generated':0,'library_fallback':0,'route_excluded':0,'numeric_excluded':0}
        cat=('route_excluded' if key in excl else 'numeric_excluded' if admit[key]['qualified']!='1' else 'generated' if codes[key]==0 else 'library_fallback')
        count[method][cat]+=1
    return {'counts':count,'cases':[dict(method=k[0],family=k[1],layout=k[2],**details[k]) for k in details],
            'provider':binary,'controls':controls,'isolated_plan_only':plan_costs(root,cases,cpu),'timed_worker_observations':sum(map(len,workers.values()))}
