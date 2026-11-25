"""Qualified one-thread FP32 public entry and native mixed-precision benchmark."""
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANEL = '5f5735f5d4b464c0f665dd0595fb695384893ba593406c9b897131a1afe069e7'
SEEDS = (131658409, 959813377)
FIXTURES = (('random', SEEDS[0]), ('random', SEEDS[1]), ('cancel', SEEDS[0]), ('scaled', SEEDS[1]))
ABI = 'm n k lda ldb ldc ar ac br bc cr cc'.split()
MIXED = ('mixed1', 'mixed3', 'mixed6', 'mixed6-bands', 'mixed9', 'fused6-amx')
BASE = 'family method role mode seed round position'.split()
METRICS = ('checked guards a_mod64 b_mod64 c_mod64 cpu threads max_abs max_rho gamma '
           'over_abs over_gamma nonfinite zero_weight first_s warm_s first_cpu_s warm_cpu_s '
           'a_sha256 b_sha256 c_sha256 implementation plans cached route_evidence').split()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def write(path, fields):
    file = path.open('w', newline='')
    w = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    def emit(row):
        w.writerow(row)
        file.flush()
    return emit

def records(output, tag):
    return [ln.split(',')[1:] for ln in output.splitlines() if ln.startswith(tag+',')]

def record(output, tag):
    values = records(output, tag)
    require(len(values)==1, 'missing or duplicate '+tag)
    return values[0]

def inventory(work, vendor, out, pre):
    x86 = platform.machine() == 'x86_64'
    host = 'mkl' if x86 else 'armpl'
    labels = {}
    ordinary = work/'ordinary.worker'
    report = write(out/'methods.csv', 'method role worker_sha256 adapter_sha256 available global_accuracy reason'.split())
    def add(name, role, plugin=None, worker=ordinary, reason=''):
        plugin = 'public' if plugin is None else plugin
        available = worker.is_file() and (plugin=='public' or plugin.is_file()) and not reason
        global_ok = True
        if role in ('mixed', 'precision-reference'):
            registration = pre.get(name,{})
            available = available and registration.get('available') is True
            global_ok = registration.get('accuracy_qualified') is True and role == 'mixed'
            if not available or not global_ok:
                reason = registration.get('reason') or 'global admission missing or precision-only control'
        if not available and not reason:
            reason = 'qualified native build unavailable'
        labels[name] = dict(role=role,worker=worker,plugin=plugin,available=available,global_ok=global_ok,reason=reason)
        report(dict(method=name,role=role,worker_sha256=digest(worker) if worker.is_file() else '',
                    adapter_sha256=digest(plugin) if plugin!='public' and plugin.is_file() else '',
                    available=int(available),global_accuracy=int(global_ok),reason=reason))
    add('cob-released','standalone')
    add('cob-released-hk15','standalone', worker=work/'cob-released-hk15.worker')
    for name in ['openblas',host]+(['mkl-jit','aocl'] if x86 else [])+['blasfeo','fastmmw-768','fastmmw-1536']:
        add(name,'independent-oneshot',vendor/(name+'.so'))
    if x86:
        status=vendor/'tblis-status.txt'
        for n in ['tblis-ab-1','tblis-ab-2']:
            add(n,'independent-oneshot',vendor/(n+'.so'),reason='' if status.is_file() and status.read_text().startswith('qualified:') else 'native TBLIS build unqualified')
    for mode in ('oneshot','cached-plan'):
        add('onednn-'+mode,'independent-oneshot' if mode=='oneshot' else 'independent-retained',
            vendor/'onednn'/('onednn-f32-'+mode+'.so'), reason='' if (vendor/'onednn/qualified').is_file() else 'strict native oneDNN build unqualified')
    manifest=work/'vendor-release.csv'
    built={r['method']:r for r in read(manifest)} if manifest.is_file() else {}
    for key in ('openblas',host):
        for level in ('two','aligned-two'):
            name=f'release-{key}-{level}'
            add(name,'release-assisted',worker=Path(built.get(name,{}).get('worker',work/(name+'.precision.worker'))),
                reason='released profile missing from qualification manifest' if manifest.is_file() and name not in built else '')
    names = [*MIXED]+([] if x86 else ['arm-'+m for m in MIXED if m!='fused6-amx'])
    for name in names:
        add(name,'precision-reference' if name.removeprefix('arm-') in ('mixed1','mixed3') else 'mixed',work/'candidates'/(name+'.so'))
    return labels

def execute(obj, p, mode, seed, count, env):
    proc = subprocess.run([obj['worker'],obj['plugin'],mode,str(seed),str(count),*(p[k] for k in ABI)],
                          env=dict(os.environ,**{k:str(v) for k,v in env.items()}),capture_output=True,text=True,timeout=210)
    require(proc.returncode==0,'worker exit '+str(proc.returncode)+' '+proc.stderr[-450:].replace('\n',' '))
    return proc.stdout

def native_record(stdout, name, k, count, routes):
    mode=name.removeprefix('arm-')
    p={'mixed1':1,'mixed3':3,'mixed6':6,'mixed6-bands':3,'mixed9':9,'fused6-amx':6}[mode]
    expected=[] if mode=='fused6-amx' else ([3*k,2*k,k] if mode=='mixed6-bands' else [k])
    declared=routes.get('required_depths')
    require(routes.get('eligible') is True and (declared==expected or (mode=='fused6-amx' and declared in ([],[k]))),
            'required shape/depth or native route disagreement')
    calls=records(stdout,'MIXED')
    plans=records(stdout,'PLAN')
    require([c[0] for c in calls]==(['FIRST','WARM'] if count else ['FIRST']),'missing native per-call receipts')
    proof=[]
    for stage,route,terms,plan_count,description in calls:
        require(int(route)==1 and int(terms)==p and int(plan_count)==len(expected),'measured range rescue, fallback, or mixed computation count changed: '+description)
        rows=[r for r in plans if r[0]==stage]
        require(len(rows)==len(expected),'native plan receipts missing')
        impl=[]
        for i,(r,depth) in enumerate(zip(rows,expected)):
            require((int(r[1]),int(r[2]),int(r[3]))==(i,depth,1) and len(r)>4,'actual native primitive/depth mismatch')
            nm=','.join(r[4:]);require(nm in routes.get('impls',[]),'native implementation differs from preflight: '+nm)
            impl.append(nm)
        proof.append(dict(stage=stage,route=int(route),executions=int(terms),depths=expected,implementations=impl,reason=description))
    return json.dumps(proof,separators=(',',':'))

def engine(work,vendor,out,cpu):
    require(os.sched_getaffinity(0)=={cpu},'controller affinity must be the declared singleton')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('panel.csv','protocol.md'):
        if (out/name).resolve()!=(HERE/name).resolve():
            shutil.copyfile(HERE/name,out/name)
    require(digest(out/'panel.csv')==PANEL,'panel checksum mismatch')
    for name in ('admission.json','native-routes.json'):
        if (work/name).resolve()!=(out/name).resolve():
            shutil.copyfile(work/name,out/name)
    pre=json.loads((out/'admission.json').read_text())
    native=json.loads((out/'native-routes.json').read_text())
    panel=read(out/'panel.csv');roster=inventory(work,vendor,out,pre)
    refs={};eligible={};anchor={}
    report=write(out/'goldens.csv','family mode seed a_sha256 b_sha256 reference_sha256 weight_sha256 bytes_each checked signed_spots signed_spot_error weight_spots weight_spot_error cpu threads'.split())
    accuracy=write(out/'accuracy.csv',BASE+METRICS+['structural','numerical','reason'])
    timings=write(out/'timings.csv',BASE+METRICS+['structural','numerical','reason'])
    q=write(out/'qualification.csv','family method role native eligible_strict timed reason'.split())
    files=work/'precision-references'; files.mkdir(exist_ok=True)
    (out/'measurement.json').write_text(json.dumps(dict(architecture=platform.machine(),cpu=cpu,page=os.sysconf('SC_PAGESIZE'),panel_sha256=PANEL,seeds=SEEDS),indent=2)+'\n')
    for p in panel:
        family=p['family'];m,n,k=map(int,(p['m'],p['n'],p['k']))
        for mode,seed in FIXTURES:
            ref=files/f'{family}-{mode}-{seed}.signed';weight=ref.with_suffix('.weight')
            require(not ref.exists() and not weight.exists(),'reference files already exist')
            stdout=execute(roster['openblas'],p,mode,seed,0,{'RAGGED_WRITE_REFERENCE':ref,'PRECISION_WRITE_WEIGHT':weight})
            a,b,checked,spots,err,observed,t=record(stdout,'GOLD')
            wspots,werr=record(stdout,'WEIGHT')
            require((int(checked),int(spots),int(wspots),int(observed),int(t))==(m*n,23,23,cpu,1) and
                    all(0<=float(x)<=1e-9 for x in (err,werr)) and ref.stat().st_size==weight.stat().st_size==m*n*8,
                    'incomplete or misplaced FP64 references')
            refs[family,mode,seed]=(ref,weight,a,b)
            report(dict(family=family,mode=mode,seed=seed,a_sha256=a,b_sha256=b,reference_sha256=digest(ref),weight_sha256=digest(weight),
                        bytes_each=m*n*8,checked=checked,signed_spots=spots,signed_spot_error=err,weight_spots=wspots,weight_spot_error=werr,cpu=cpu,threads=1))
        print('verified FP64 oracles:',family,flush=True)

    def observe(p,name,mode,seed,rnd,position,emit):
        shape=p['family'];obj=roster[name];ref,weight,aa,bb=refs[shape,mode,seed]
        m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc=(int(p[i]) for i in ABI);count=int(rnd>0)
        row=dict(family=shape,method=name,role=obj['role'],mode=mode,seed=seed,round=rnd,position=position)
        try:
            s=execute(obj,p,mode,seed,count,{'RAGGED_REFERENCE':ref,'PRECISION_WEIGHT':weight})
            v=record(s,'VIEW');h=record(s,'RAGGED');e=record(s,'ACCURACY')
            check,guards,am,bm,cm,observed,threads=map(int,v[:7]);absolute,first,warm=map(float,v[7:10])
            pc_first,pc_warm=map(float,h[3:5]);gamma,rho=map(float,e[:2]);over_abs,over_gamma,nonfinite,zeros=map(int,e[2:])
            page=os.sysconf('SC_PAGESIZE')
            expected_guards=((cr+m+2)*lc*4+page-1)//page*(page//4)-m*n
            require((check,guards,observed,threads)==(m*n,expected_guards,cpu,1) and
                    [am,bm,cm]==[4*x%64 for x in (ar*la+ac,br*lb+bc,cr*lc+cc)] and
                    list(map(int,v[10:]))==[la,lb,lc], 'guard, CPU, thread or view mismatch')
            require(h[:2]==[aa,bb] and re.fullmatch('[a-f0-9]{64}',h[2]),'input or output identity mismatch')
            require(all(math.isfinite(x) and x>0 for x in ([first,pc_first,warm,pc_warm] if count else [first,pc_first])),'invalid clock')
            require(math.isclose(gamma,k/(16777216-k),rel_tol=1e-14) and all(0<=x<=m*n for x in (over_abs,over_gamma,nonfinite,zeros)),'bad numerical accounting')
            impl='';plans='';cached='';route=''
            if name.startswith('onednn-'):
                d=record(s,'DNN');impl=','.join(d[:-2]);plans,cached=map(int,d[-2:])
                require(impl and not impl.lower().startswith('ref') and cached==int(name.endswith('cached-plan')) and plans==(1 if cached else 1+count),'strict executable or cached plan mismatch')
            row.update(dict(zip(METRICS,[check,guards,am,bm,cm,observed,threads,absolute,rho,gamma,over_abs,over_gamma,nonfinite,zeros,
                       first,warm,pc_first,pc_warm,aa,bb,h[2],impl,plans,cached,route])))
            if obj['role'] in ('mixed','precision-reference'):
                row['route_evidence']=native_record(s,name,k,count,native[shape][name])
            if count:require(anchor[shape,name,mode,seed]==h[2],'warm output differs from same-input first-call qualification')
            else:anchor[shape,name,mode,seed]=h[2]
            numeric=bool(math.isfinite(absolute) and absolute<=.002 and math.isfinite(rho) and not (over_abs or over_gamma or nonfinite))
            why='' if numeric else 'all-cell historical absolute or componentwise FP64 accuracy failed'
            emit(dict(row,structural=1,numerical=int(numeric),reason=why))
            return True,numeric,why
        except (ValueError,OSError,IndexError,KeyError,subprocess.TimeoutExpired) as exc:
            emit(dict(row,structural=0,numerical=0,reason=str(exc)));return False,False,str(exc)

    for p in panel:
        f=p['family']
        for name,obj in roster.items():
            mixed=obj['role'] in ('mixed','precision-reference');shape_route=native.get(f,{}).get(name,{})
            safe=obj['available'] and (not mixed or shape_route.get('eligible') is True)
            issues=[];numbers=True
            if not safe:
                issues.append((shape_route.get('reason') if mixed else '') or obj['reason'] or 'native route unavailable')
            else:
                for mode,seed in FIXTURES:
                    structurally_ok,numeric,why=observe(p,name,mode,seed,0,-1,accuracy)
                    safe &= structurally_ok; numbers &= numeric
                    if why: issues.append(why)
            strict=safe and numbers and obj['global_ok']
            if not obj['global_ok']:issues.append(obj['reason'] or 'precision-only or global accuracy unqualified')
            eligible[f,name]=safe
            q(dict(family=f,method=name,role=obj['role'],native=int(safe),eligible_strict=int(strict),timed=int(safe),reason='; '.join(dict.fromkeys(issues))))
        print('qualified complete outputs:',f,flush=True)
    for block,seed in enumerate(SEEDS):
        for rnd in (1,2,3):
            cases=panel[rnd-1:]+panel[:rnd-1]
            if block:cases=cases[::-1]
            for p in cases:
                f=p['family'];idx=panel.index(p)
                names=[n for n in roster if eligible[f,n]]
                off=(idx+3*(rnd-1)+5*block)%max(1,len(names))
                order=names[off:]+names[:off]
                if block:order.reverse()
                for pos,name in enumerate(order):
                    observe(p,name,'random',seed,rnd,pos,timings)
                print('completed warm pairs:',f,seed,rnd,flush=True)
    subprocess.run([sys.executable,str(HERE/'score.py'),out],check=True)

if __name__=='__main__':
    if len(sys.argv)!=5:raise SystemExit('usage: study.py WORK VENDOR OUT CPU')
    engine(*(Path(x).resolve() for x in sys.argv[1:4]), int(sys.argv[4]))
