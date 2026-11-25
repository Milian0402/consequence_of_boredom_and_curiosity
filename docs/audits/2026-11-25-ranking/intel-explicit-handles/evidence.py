"""Read the ordinary/diagnostic/control/independent implementation evidence."""
import hashlib
import json
import math
import re
from collections import Counter

from raw import BASE, DIRECT, HANDLE1, HANDLEP, RETAINED, ROSTER, SEEDS, require, rows
from verify.layout import verify_layout

RELEASE = 'a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6'
TREE = '18f22622de1a4b1b7aebbdb66985741aa38c6e9e'
PROTOCOL = 'b63ad49548f9c8fd0f429427de3c860309d991bc3a791a695c15d5f6623e673e'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return re.fullmatch(r'[0-9a-f]{64}', value) is not None


def index(records, *fields):
    result = {}
    for record in records:
        key = tuple(record[f] for f in fields)
        require(key not in result, f'duplicate identity {key}')
        result[key] = record
    return result


def host(root):
    raw = (root/'host.txt').read_text()
    def field(k):
        match = re.search('^' + re.escape(k) + r':\s*([^\n]+)', raw, re.M)
        return match[1].strip() if match else ''
    model, vendor, flags = field('Model name'), field('Vendor ID'), set(field('Flags').split())
    require(field('Architecture') == 'x86_64' and model and vendor in {'GenuineIntel', 'AuthenticAMD'}, 'host identity')
    isa = 'avx512' if 'avx512f' in flags else 'avx2'
    require({'avx2', 'fma'} <= flags and (root/'grouped-native-features.txt').read_text().split() ==
            ['1', str(int(isa == 'avx512'))], 'runtime ISA selection')
    return model, vendor, isa


def native(root, methods):
    docs=json.loads((root/'grouped-sources.json').read_text())
    require(docs['release']==RELEASE and docs['tree']==TREE and set(docs['methods'])=={BASE},'installed source identity')
    spec=docs['methods'][BASE]
    b,=rows(root/'grouped-builds.csv')
    manifest=sha256(json.dumps(spec['files'],sort_keys=True,separators=(',',':')).encode())
    require(manifest==b['source_manifest_sha256']==spec['sha256']=='c1ec96b0a6fb6fb7f80fa94addc4f4ff20eab39c68873ff0af89030eeafb6679','released production source manifest')
    require(b['source_revision']==RELEASE and b['patch_sha256']=='none' and b['header_sha256']==spec['files']['include/cob_gemm.h'] and
            b['public_symbol']=='cob_sgemm_rowmajor' and b['worker_sha256']==methods[BASE,]['sha256'] and b['worker_sha256']!=b['trace_sha256'] and
            b['archive_sha256']==b['installed_archive_sha256'],'ordinary installed executable')
    require('100% tests passed' in (root/(BASE+'-tests.txt')).read_text(),'release tests')
    flag_file=root/(BASE+'-flags.json'); require(sha256(flag_file.read_bytes())==b['flags_sha256'],'compile options')
    opts=json.loads(flag_file.read_text())
    require(set(opts)=={'public','avx2','avx512'} and all('-O3' in x and '-masm=att' in x for x in opts.values()),'separate optimized ISA paths')
    return {'release':RELEASE,'production_source_manifest':manifest,
            'ordinary':{m[0]:v['sha256'] for m,v in methods.items() if v['status']=='available'},
            'provider':methods['mkl',]['sha256'],'trace':{}}


def peers(root, panel, raw, methods):
    dims = ('m','n','k','lda','ldb','ldc')
    xsmm = index(rows(root/'libxsmm-routes.csv'), 'method','family','layout')
    require(set(xsmm) == {(m,s['family'],s['layout']) for m in ('libxsmm-dispatch','libxsmm-retained') for s in panel},
            'LIBXSMM exact geometry coverage')
    for s in panel:
        for m in ('libxsmm-dispatch','libxsmm-retained'):
            r = xsmm[m,s['family'],s['layout']]
            require([r[k] for k in dims] == [s[k] for k in dims] and int(r['admitted'])==int(int(r['code_bytes'])>0), 'LIBXSMM exact-shape JIT')
            require(r['reason']==('verified_nonreference_jit' if int(r['admitted']) else 'no_verified_nonreference_kernel'), 'JIT qualification')
    key_fields=['method','family','layout','mode','seed','round']
    dnn=index(rows(root/'onednn-routes.csv'),*key_fields)
    raws = index([r for r in raw if r['method'].startswith('onednn-')],*key_fields)
    require(set(raws)==set(dnn), 'per-process measured oneDNN provider evidence')
    implementations = Counter()
    for key,r in dnn.items():
        cached = int(r['method'] == 'onednn-cached-plan')
        require(int(r['cached'])==cached and int(r['plans'])==(1 if cached else 1+int(raws[key]['calls'])) and r['implementation'].strip(), 'primitive and cache policy')
        implementations[r['implementation']]+=1
    for n in ('oneshot', 'cached-plan'):
        if methods[('onednn-'+n,)]['status']=='available':
            text=(root/('onednn-qualified-'+n+'.log')).read_text()
            require('strict_fp32_residual=0x1p-23 subnormal=0x1p-149' in text and 'PASS full_fp64_outputs=73096' in text, 'native sequential strict oneDNN admission')
    return {'xsmm_records':len(xsmm),'dnn_identities':dict(implementations)}, xsmm


def validate(root, acceptance):
    from collections import defaultdict
    from raw import PANEL_SHA, qualify_samples
    from handles import verify_handles
    from field import rank_fields
    run=json.loads((root/'run.json').read_text())
    require(run['production']==RELEASE and len(run['benchmark'])==40, 'native run identity')
    require(all(sha256((root/f).read_bytes())==h for f,h in run['source_files'].items()), 'source and method digests')
    require(run['source_files']['panel.csv']==PANEL_SHA and run['source_files']['protocol.md']==PROTOCOL, 'changed registration')
    require((run['run'],run['benchmark'],run['replica'],run['attempt'])==
            (acceptance['run'],acceptance['benchmark_sha'],acceptance['replica'],acceptance['attempt']),'hosted/native identity mismatch')
    page=int((root/'page-size.txt').read_text())
    require(page==run['page'] and page>=4096 and page&(page-1)==0,'native page size')
    model,vendor,isa=host(root)
    ordered=rows(root/'methods.csv')
    methods=index(ordered,'method')
    require(tuple(r['method'] for r in ordered)==ROSTER,'native roster')
    for (m,),r in methods.items():
        require(r['role']==('standalone' if m==BASE else 'independent') and
                r['policy']==('retained' if m in (*RETAINED,HANDLEP) else 'one-shot') and r['status'] in ('available','unavailable') and
                (r['status']=='unavailable' or digest(r['sha256'])),'provider identity or call policy')
    ident=native(root,methods)
    panel,raw=rows(root/'panel.csv'),rows(root/'raw.csv')
    incumbent=index(rows(root/'incumbent-routes.csv'),'family','layout')
    require(set(incumbent)=={(s['family'],s['layout']) for s in panel},'every standalone witness')
    for s in panel:
        c=incumbent[s['family'],s['layout']]
        require(c['isa']==isa and int(c['checked'])==int(s['m'])*int(s['n']) and int(c['cpu'])==run['cpu'] and
                all(c[k]=='1' for k in ('kernel_calls','public_calls','forwarded_args','workspace_aligned','passed')),'ordinary COB executed native kernel')
    details,jit=peers(root,panel,raw,methods)
    handle_proof=verify_handles(root,panel,raw,ordered,run['cpu'])
    handle_rows=index(rows(root/'handle-observations.csv'),'method','family','layout','mode','seed','round')
    admission=index(rows(root/'admissions.csv'),'family','layout','method')
    expect={(s['family'],s['layout'],m) for s in panel for m in ROSTER}
    require(set(admission)==expect,'missing declared case/method')
    bycell=defaultdict(list)
    for r in raw:
        key=r['family'],r['layout'],r['method']
        require(key in expect and int(r['cpu'])==run['cpu'],'unregistered method or changed logical CPU')
        bycell[key].append(r)
    failures=index(rows(root/'failures.csv'),'family','layout','method')
    require(set(failures)<=expect,'unregistered failure')
    cells=[]
    for s in panel:
        good,bad={},{}
        for m in ROSTER:
            key=s['family'],s['layout'],m
            a=admission[key];rr=bycell[key]
            require(a['qualified'] in ('0','1') and a['reason'] and ((a['qualified']=='1')==(a['reason']=='qualified')),
                    'unexplained omission')
            require([sum((int(r['round'])>=0)==warm for r in rr) for warm in (False,True)]==
                    [int(a[k]) for k in ('accuracy_records','timing_records')],'observed/declared counts')
            if key in failures: require(a['qualified']=='0' and a['reason']==failures[key]['reason'],'ignored failure')
            if methods[m,]['status']=='unavailable':require(a['qualified']=='0','unavailable binary claimed timing')
            if m.startswith('libxsmm-') and jit[m,key[0],key[1]]['admitted']=='0':require(a['qualified']=='0','unverified LIBXSMM JIT')
            val,why=qualify_samples(s,rr,page,[] if a['qualified']=='1' else [a['reason']])
            require((val is not None)==(a['qualified']=='1'),'independent qualification contradicts native')
            if val is None:bad[m]=why;continue
            val['prep']=[0.0]*6
            val['api_status']=''
            if m in (HANDLE1,HANDLEP):
                paired_rows=[handle_rows[(m,key[0],key[1],'random',str(SEEDS[j//3]),str(j))] for j in range(6)]
                cf={r[k] for r in paired_rows for k in ('first_status','last_status')}
                require(len(cf)==1 and cf<={'0','1'},'mixed or invalid measured API output')
                val['api_status']='MKL_JIT_SUCCESS' if cf=={'0'} else 'MKL_NO_JIT'
                val['prep']=[float(r['prepare_s']) for r in paired_rows]
            good[m]=val
        cells.append({'layout':s,'methods':good,'exclusions':bad})
    canonical=[]
    for j in range(6):
        seq=panel[j%3:]+panel[:j%3]
        if j>=3:seq.reverse()
        for s in seq:
            shift=(panel.index(s)+2*(j%3))%len(ROSTER)
            ms=list(ROSTER[shift:]+ROSTER[:shift])
            if j>=3:ms.reverse()
            for place,m in enumerate(ms):canonical.append((j,s['family'],s['layout'],m,place))
    actual=[(int(r['round']),r['family'],r['layout'],r['method'],int(r['place'])) for r in raw if int(r['round'])>=0]
    present=set(actual)
    require(len(present)==len(actual) and [r for r in canonical if r in present]==actual,'method/layout order')
    rankings=rank_fields(cells)
    hosted=root/'handle-score.json'
    if hosted.exists():
        reported=json.loads(hosted.read_text())
        tier_map={'original_one_shot':'original-one-shot','original_including_retained':'original-including-retained','extended_one_shot':'expanded-one-shot','extended_including_retained':'expanded-including-retained'}
        for regime,tiers in rankings['aggregates'].items():
            remote_metric={'warm':'warm','first':'first','first_with_setup':'first-with-preparation'}[regime]
            for name,merged in tiers.items():
                key=tier_map[name]+' '+remote_metric
                if not merged or merged['geomean'] is None: continue
                recorded=reported['summary'][key]
                require(math.isclose(merged['geomean'],recorded['geomean'],rel_tol=1e-12) and
                        merged['paired_wins']==recorded['layout_wins'] and merged['round_wins']==recorded['paired_wins'],'hosted / independent score '+key)
                for l in panel:
                    case=rankings['cases'][l['family']][regime][name]
                    remote=reported['layouts'][l['family']]['tiers'].get(tier_map[name],{}).get(remote_metric)
                    if case['comparison'] is None: continue
                    require(remote and case['method']==remote['leader'] and case['complete']==remote['complete'],'native fastest method '+key)
    return dict(cells=cells,rankings=rankings,model=model,vendor=vendor,isa=isa,run=run,receipt=acceptance,
                identity=ident,handle_witnesses=handle_proof,peers=details,
                qualified_count=sum(len(c['methods']) for c in cells),
                numerical={'raw_fp64_outputs':sum(int(r['checked']) for r in raw),'four_fixture_fp64_outputs':sum(int(r['checked']) for r in raw if int(r['round'])<0),
                           'checked_parent_storage':sum(int(r['guards']) for r in raw),'max_absolute_error':max(float(r['error']) for r in raw),
                           'complete_warm_calls':sum(int(r['calls']) for r in raw),'warm_batches':len(actual),'sampled_fresh_workers':len(raw)})
