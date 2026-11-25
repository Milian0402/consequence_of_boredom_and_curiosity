"""Read the ordinary/diagnostic/control/independent implementation evidence."""
import hashlib
import json
import math
import re
from collections import Counter

from score import BASE, DIRECT, NEW, RETAINED, ROSTER, SEEDS, require, rows
from verify.layout import verify_layout

RELEASE = 'a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6'
TREE = '18f22622de1a4b1b7aebbdb66985741aa38c6e9e'
PATCH = 'd3c5ac7caffa8d00a0d6a0df5a617d66b17f7649b738c5b3e364858e6fd0449f'
PROTOCOL = 'a36cfd1999ebedf902a649f982ff30e498c6b8d8edb69e09ff7862f233f778a2'


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
    docs = json.loads((root/'grouped-sources.json').read_text())
    require([docs['release'], docs['tree'], docs['patch_sha256']] == [RELEASE, TREE, PATCH], 'release/patch identity')
    require(sha256((root/'protocol.md').read_bytes()) == PROTOCOL and
            (root/'candidate.sha256').read_text().split()[0] == PATCH, 'protocol or candidate changed')
    base, new = (docs['methods'][n]['files'] for n in (BASE, NEW))
    changed = sorted(p for p in base.keys() | new.keys() if base.get(p) != new.get(p))
    require(changed == docs['changed_paths'] == ['src/cob_stream_b_dispatch.inc', 'src/cob_stream_b_group.h', 'src/cob_stream_b_kernel.c'], 'production delta')
    builds = index(rows(root/'grouped-builds.csv'), 'implementation')
    require(set(builds) == {(BASE,), (NEW,)}, 'both standalone builds')
    flags = {}
    for method in (BASE, NEW):
        spec = docs['methods'][method]
        b = builds[(method,)]
        actual_manifest = sha256(json.dumps(spec['files'], sort_keys=True, separators=(',', ':')).encode())
        require(actual_manifest == b['source_manifest_sha256'] == spec['sha256'], 'source inventory')
        f = root/(method+'-flags.json')
        require(sha256(f.read_bytes()) == b['flags_sha256'], 'compiler flag digest')
        flags[method] = json.loads(f.read_text())
        require(b['archive_sha256'] == b['installed_archive_sha256'] and
                b['worker_sha256'] == methods[(method,)]['sha256'] and b['trace_sha256'] != b['worker_sha256'] and
                all(digest(b[k]) for k in ('archive_sha256', 'worker_sha256', 'trace_sha256', 'header_sha256')) and
                b['header_sha256'] == spec['files']['include/cob_gemm.h'] and b['public_symbol'] == 'cob_sgemm_rowmajor' and
                b['source_revision'] == RELEASE and b['patch_sha256'] == (PATCH if method == NEW else 'none'), 'installed ordinary worker')
        require('100% tests passed' in (root/(method+'-tests.txt')).read_text(), 'baseline library tests')
    require(builds[BASE,]['archive_sha256'] != builds[NEW,]['archive_sha256'], 'distinct archives')
    switch = '-DCOB_ENABLE_X86_GROUPED_B=1'
    for target, opts in flags[NEW].items():
        ref = flags[BASE][target]
        require(opts.count(switch) == 1 and switch not in ref and [o for o in opts if o != switch] == ref,
                'candidate-only compiler switch')
        require('-O3' in ref and '-masm=att' in ref, 'matched optimized native compilation')
    return {'release': RELEASE, 'patch': PATCH, 'source_manifests': {n: docs['methods'][n]['sha256'] for n in (BASE,NEW)}}


def routes(root, panel, page, cpu, isa):
    by_id = {(s['family'], s['layout']): s for s in panel}
    regular = rows(root/'grouped-routes.csv')
    rescued = rows(root/'grouped-fallbacks.csv')
    require(set(index(regular,'family','layout','implementation')) ==
            {(*p, m) for p in by_id for m in (BASE, NEW)}, 'all real and bypass routes')
    require(Counter(r['implementation'] for r in rescued) == Counter((BASE, NEW)), 'allocation rescue per build')
    for r in regular + rescued:
        view = by_id[r['family'],r['layout']]
        m, n, k = (int(view[f]) for f in ('m','n','k'))
        valid = 24 <= m <= 160 and 1024 <= n <= 4096 and 96 <= k <= 384 and n >= 8*m and m*k <= 49152
        grouped = r['implementation'] == NEW
        is_rescue = 'allocation_refusals' in r
        selected = ('grouped_' if grouped else '') + isa
        require(all(r[f] == view[f] for f in view), 'forwarded layout')
        verify_layout(view, dict(r, guards=r['guard_cells']), page)
        require(int(r['trace_checked']) == m*n and all(r[k]=='1' for k in
                ('guards','inputs_unchanged','threads','passed','args_ok','workspace_ok')) and int(r['cpu'])==cpu and r['isa']==isa,
                'complete native route inspection')
        for field in ('avx2','avx512','grouped_avx2','grouped_avx512'):
            require(int(r[field]) == int(valid and not is_rescue and field == selected), 'expected real kernel '+field)
        expect = 'allocation-fallback' if is_rescue else 'bypass' if not valid else 'grouped-b' if grouped else 'incumbent'
        require(r['route'] == expect and int(r['candidate_calls']) == int(grouped and valid and not is_rescue), 'kernel route')
        if valid:
            size = 4*k*(64 if isa=='avx512' else 32)*(4 if grouped else 1)
            require(int(r['first_bytes']) == size and int(r['first_align']) == 64, 'actual promoted scratch footprint')
        if is_rescue: require(valid and r['allocation_refusals']=='1','real failed allocation')
        require(0 <= float(r['error']) <= .002 and 0 <= float(r['trace_error']) <= .002, 'valid route output')
    parity = index(rows(root/'grouped-control-parity.csv'), 'family','layout','mode','seed')
    fixtures = [('random', SEEDS[0]), ('random', SEEDS[1]), ('cancel', SEEDS[0]), ('scaled', SEEDS[1])]
    expected = {(*key, mode, str(seed)) for key in by_id if key[0].startswith('control-') for mode,seed in fixtures}
    require(set(parity) == expected, 'two full bypasses and all four fixtures')
    for key,r in parity.items():
        s = by_id[key[:2]]
        require(int(r['checked']) == int(s['m'])*int(s['n']) and r['passed']=='1' and digest(r['candidate_sha256']) and
                r['candidate_sha256'] == r['incumbent_sha256'], 'complete output bit parity')
    return {'normal': len(regular), 'allocation_failure': len(rescued), 'bypass_bitwise_fixtures': len(parity)}


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
        require(row['role']==('standalone' if method in (BASE,NEW) else 'independent') and
                row['policy']==('retained' if method in RETAINED else 'one-shot') and row['status'] in ('available','unavailable') and
                (row['status']=='unavailable' or digest(row['sha256'])), 'provider binary identity/field')
    source=native(root,methods)
    panel,raw=rows(root/'panel.csv'),rows(root/'raw.csv')
    trace=routes(root,panel,page,run['cpu'],isa)
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
    hosted_score=root/'grouped-score.json'
    if hosted_score.exists():
        native_score=json.loads(hosted_score.read_text())
        primary=native_score['primary']
        matched=ranks['aggregates'][NEW+'/'+BASE]
        if primary.get('complete'):
            require(matched and math.isclose(primary['geomean'],matched['geomean'],rel_tol=1e-12) and
                    primary['layout_wins']==matched['median_wins'] and primary['paired_wins']==matched['wins_of_48'] and
                    primary['passed']==ranks['standalone_timing_gate'],'hosted standalone gates differ')
        for label,tier in [('one-shot','one_shot'),('including-retained','including_retained')]:
            ext=native_score['external'][label]
            ours=ranks['aggregates'][NEW+'/'+tier]
            if ours:
                require(ranks['tier_complete'][tier]==ext['complete'] and math.isclose(ours['geomean'],ext['geomean'],rel_tol=1e-12) and
                        ours['wins_of_48']==ext['paired_wins'] and ours['median_wins']==ext['layout_wins'] and
                        ext['stable_lead']==(ranks['tier_complete'][tier] and ours['repeatable_layout_leads']==8),'hosted independent gates differ')
        for cell in ranks['cells']:
            r=native_score['layouts'][cell['layout']['family']]
            for label,tier in [('one-shot','one_shot'),('including-retained','including_retained'),('whole-field','whole')]:
                require(cell['winners'].get(tier)==r['tiers'].get(label,{}).get('method'),'raw-median winner differs from online table')
    ranks.update(model=model,vendor=vendor,isa=isa,run=run,receipt=acceptance,identity=source,
                 routes=trace,peers=details,qualified_count=sum(ranks['qualified_method_layout_counts'].values()),
                 numerical={'raw_fp64_outputs':sum(int(r['checked']) for r in raw),'four_fixture_fp64_outputs':sum(int(r['checked']) for r in raw if int(r['round'])<0),
                            'checked_parent_storage':sum(int(r['guards']) for r in raw),
                            'max_absolute_error':max(float(r['error']) for r in raw),'complete_warm_calls':sum(int(r['calls']) for r in raw),
                            'warm_batches':len(actual),'sampled_fresh_workers':len(raw)})
    return ranks
