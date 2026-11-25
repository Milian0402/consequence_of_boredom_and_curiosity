"""Read the ordinary/diagnostic/control/independent implementation evidence."""
import hashlib
import json
import math
import re
from collections import Counter

from score import BASE, DIRECT, NEW, FORWARD, SPLIT, RETAINED, ROSTER, SEEDS, require, rows
from verify.layout import verify_layout

RELEASE = 'a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6'
TREE = '18f22622de1a4b1b7aebbdb66985741aa38c6e9e'
PROTOCOL = '02581f0e088da5d11c0308b3583b9ef5f87e239999268ff6430952596db57c60'


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
    from score import evaluate, PANEL_SHA
    run=json.loads((root/'run.json').read_text())
    require(run['production']==RELEASE and len(run['benchmark'])==40, 'native run identity')
    require(all(sha256((root/f).read_bytes())==h for f,h in run['source_files'].items()), 'run-bound source and method digests')
    require(run['source_files']['panel.csv']==PANEL_SHA and run['source_files']['protocol.md']==PROTOCOL, 'changed registration')
    if acceptance:
        require((run['run'],run['benchmark'],run['replica'],run['attempt']) ==
                (acceptance['run'],acceptance['benchmark_sha'],acceptance['replica'],acceptance['attempt']), 'hosted/native identity mismatch')
    page=int((root/'page-size.txt').read_text())
    require(page==run['page'] and page>=4096 and page&(page-1)==0, 'native page size')
    model,vendor,isa=host(root)
    ordered=rows(root/'methods.csv')
    methods=index(ordered,'method')
    require(tuple(r['method'] for r in ordered)==ROSTER, 'incomplete/out-of-order native roster')
    for (method,),row in methods.items():
        require(row['role']==('standalone' if method==BASE else 'vendor-assisted' if method in (NEW,FORWARD,SPLIT) else 'independent') and
                row['policy']==('retained' if method in RETAINED else 'one-shot') and row['status'] in ('available','unavailable') and
                (row['status']=='unavailable' or digest(row['sha256'])), 'provider binary identity/field')
    source=native(root,methods)
    panel,raw=rows(root/'panel.csv'),rows(root/'raw.csv')
    from routes import verify_routes
    cob_routes=index(rows(root/'incumbent-routes.csv'),'family','layout')
    require(set(cob_routes)=={(s['family'],s['layout']) for s in panel},'every incumbent route')
    for s in panel:
        c=cob_routes[s['family'],s['layout']]
        require(c['isa']==isa and int(c['checked'])==int(s['m'])*int(s['n']) and int(c['cpu'])==run['cpu'] and
                all(c[k]=='1' for k in ('kernel_calls','public_calls','forwarded_args','workspace_aligned','passed')),'ordinary incumbent public trace')
    trace=verify_routes(root,panel,run['cpu'],ordered)
    trace['incumbent_routes']=len(cob_routes)
    source['trace']={s['method']:s['trace_sha256'] for s in rows(root/'prefix-builds.csv')}
    details,xsmm=peers(root,panel,raw,methods)
    admission=index(rows(root/'admissions.csv'),'family','layout','method')
    keys={(s['family'],s['layout'],m) for s in panel for m in ROSTER}
    require(set(admission)==keys,'missing registered admissions')
    failures=index(rows(root/'failures.csv'),'family','layout','method')
    require(set(failures)<=keys,'unexpected failures')
    observed=Counter((r['family'],r['layout'],r['method'],int(r['round'])>=0) for r in raw)
    reasons={}
    for key,info in admission.items():
        family,layout,m=key
        require(info['qualified'] in ('1','0') and info['reason'] and ((info['qualified']=='1')==(info['reason']=='qualified')),
                'unexplained excluded method')
        require([int(info[c]) for c in ('accuracy_records','timing_records')]==[observed[(*key,warm)] for warm in (False,True)],'raw/admission count mismatch')
        if key in failures: require(info['qualified']=='0' and info['reason'] == failures[key]['reason'], 'ignored worker failure')
        if m.startswith('libxsmm-') and xsmm[m,family,layout]['admitted']=='0': require(info['qualified']=='0','unadmitted JIT kernel timed')
        if methods[m,]['status']=='unavailable': require(info['qualified']=='0','unavailable provider timed')
        if info['qualified']=='0': reasons[key]=[info['reason']]
    ranks=evaluate(panel,raw,page,reasons,ROSTER)
    require(ranks['cpus']==[int(run['cpu'])], 'mixed worker affinity')
    for cell in ranks['cells']:
        l=cell['layout']
        for m in ROSTER:
            require((m in cell['methods'])==(admission[l['family'],l['layout'],m]['qualified']=='1'), 'offline/native admission contradiction')
    expected_sequence=[]
    for i in range(6):
        ordered_views=panel[i%3:]+panel[:i%3]
        if i>=3: ordered_views.reverse()
        for s in ordered_views:
            shift=(panel.index(s)+2*(i%3))%len(ROSTER)
            ms=list(ROSTER[shift:]+ROSTER[:shift])
            if i>=3: ms.reverse()
            for place,m in enumerate(ms):
                expected_sequence.append((i,s['family'],s['layout'],m,place))
    actual=[(int(r['round']),r['family'],r['layout'],r['method'],int(r['place'])) for r in raw if int(r['round'])>=0]
    actual_set=set(actual)
    require(len(actual_set)==len(actual) and [x for x in expected_sequence if x in actual_set]==actual,'matched native order or place')
    ranks['promotion']=ranks['meets_warm_timing_criteria'] if vendor=='GenuineIntel' else None
    summary_file=root/'prefix-score.json'
    if summary_file.is_file():
        reported=json.loads(summary_file.read_text())
        pr=reported['primary']
        require(pr['eligible_cpu']==(vendor=='GenuineIntel') and pr['complete']==ranks['required_numerical_complete'] and
                pr['passed']==(ranks['promotion'] is True),'primary decision disagrees with native results')
        if pr['complete']:
            require(pr['qualifying_fresh']==len(ranks['qualifying_fresh_layouts']),'joint layout count')
            for method in ('mkl',FORWARD,SPLIT):
                m=pr['comparisons'][method]
                warm=ranks['aggregates']['warm'][NEW+'/'+method]
                cold=ranks['aggregates']['first'][NEW+'/'+method]
                require(math.isclose(m['geomean'],warm['geomean'],rel_tol=1e-12) and
                        math.isclose(m['first_geomean'],cold['geomean'],rel_tol=1e-12) and
                        warm['wins_of_36']==m['paired_wins'] and warm['median_wins']==m['layout_wins'],'matched cold/warm results')
        for c in ranks['cells']:
            family=c['layout']['family']
            for tier,label in [('one_shot','one-shot'),('including_retained','including-retained'),('whole','all')]:
                require(c['winners'].get(tier)==reported['layouts'][family]['tiers'].get(label,{}).get('method'),'rank winner')
    probe=root/'explicit-jit-probe-status.txt'
    ranks['explicit_jit']={'status':probe.read_text().strip() if probe.exists() else 'absent','panel_counts':{}}
    if ranks['explicit_jit']['status']=='pass':
        checks=index(rows(root/'explicit-jit.csv'),'family','layout')
        expected={(s['family'],s['layout']):s for s in panel}
        require(set(checks)==set(expected)|{('probe-small-16','compact')},'explicit MKL JIT admission panel')
        counts=Counter()
        for k,c in checks.items():
            shape=expected.get(k)
            if shape:require(all(c[key]==v for key,v in shape.items()),'explicit JIT complete physical layout')
            status=c['status']
            require(int(c['cpu'])==run['cpu'] and (status,c['code']) in {('MKL_JIT_SUCCESS','0'),('MKL_NO_JIT','1'),('MKL_JIT_ERROR','2')},'genuine native explicit API status')
            if status!='MKL_JIT_ERROR':
                require(int(c['called'])==1 and int(c['checked'])==int(c['m'])*int(c['n']) and int(c['guards'])>0,'unranked explicit API per-output result')
            if shape:counts[status]+=1
        ranks['explicit_jit'].update(panel_counts=dict(counts),tiny_status=checks[('probe-small-16','compact')]['status'])
    ranks.update(model=model,vendor=vendor,isa=isa,run=run,receipt=acceptance,identity=source,
                 routes=trace,peers=details,qualified_count=sum(ranks['qualified_method_layout_counts'].values()),
                 numerical={'raw_fp64_outputs':sum(int(r['checked']) for r in raw),'four_fixture_fp64_outputs':sum(int(r['checked']) for r in raw if int(r['round'])<0),
                            'checked_parent_storage':sum(int(r['guards']) for r in raw),
                            'max_absolute_error':max(float(r['error']) for r in raw),'complete_warm_calls':sum(int(r['calls']) for r in raw),
                            'warm_batches':len(actual),'sampled_fresh_workers':len(raw)})
    return ranks
