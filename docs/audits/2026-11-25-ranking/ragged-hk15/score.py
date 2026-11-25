"""Replay the registered ragged HK15 timings and per-host gates. Run with OUT."""
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median

BASE = 'released-hk15'
CAND = 'ragged-hk15'
SEEDS = (154983127, 720184093)
FIXTURES = {('random', SEEDS[0]), ('random', SEEDS[1]), ('cancel', SEEDS[0]), ('scaled', SEEDS[1])}
BLOCKS = tuple((s, r) for s in SEEDS for r in (1, 2, 3))
PANEL_SHA = '6d9d898375f2f57eea76ae42d6ec04b0792d5440967da260b52c47b59c77dc42'
PROTOCOL_SHA = '51ac3e957ad14d06eadcbea25ba17a60c75d5a237e2c2c64cf8d951ceacc5460'
ROLES = {'baseline', 'candidate', 'vendor-assisted', 'independent-oneshot', 'independent-retained'}


def require(test, message):
    if not test:
        raise ValueError(message)


def read(root, name):
    with (root / name).open(newline='') as f:
        return list(csv.DictReader(f))


def unique(rows, index):
    result = {}
    for row in rows:
        k = index(row)
        require(k not in result, f'duplicate record {k}')
        result[k] = row
    return result


def is_hash(value):
    return re.fullmatch('[0-9a-f]{64}', value) is not None


def comparison(peer, candidate):
    ratios = [float(peer[k]['warm_s']) / float(candidate[k]['warm_s']) for k in BLOCKS]
    return {'median': median(ratios), 'ratios': ratios, 'wins': sum(v > 1 for v in ratios),
            'ties': sum(v == 1 for v in ratios), 'seed_medians': [median(ratios[:3]), median(ratios[3:])]}


def score(root):
    for name, digest in [('panel.csv', PANEL_SHA), ('protocol.md', PROTOCOL_SHA)]:
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, 'changed '+name)
    panel = unique(read(root, 'panel.csv'), lambda r: r['family'])
    hosts = (root/'lscpu.txt').read_text().splitlines()
    host = {a.strip(): b.strip() for x in hosts if ':' in x for a, b in [x.split(':',1)]}
    arch = host['Architecture']
    require(arch in ('aarch64','x86_64'), 'unexpected architecture')
    provider = 'armpl' if arch == 'aarch64' else 'mkl'
    required_peers = {'openblas', provider, 'blasfeo', 'fastmmw-768', 'fastmmw-1536'}
    if arch == 'x86_64':
        required_peers.update(('mkl-jit','aocl'))
    required_assisted = {f'release-{p}-aligned-two' for p in ('openblas',provider)}
    cpu = int((root/'pinned-cpu.txt').read_text())
    page = int((root/'page-bytes.txt').read_text())
    methods = unique(read(root, 'sources/methods.csv'), lambda r:r['method'])
    role = {m:v['role'] for m,v in methods.items()}
    require(set(role.values()) <= ROLES and role[BASE]=='baseline' and role[CAND]=='candidate', 'method identities')
    require(all(role[m]=='independent-oneshot' for m in required_peers) and
            all(role[m]=='vendor-assisted' for m in required_assisted), 'missing required method identity')
    for m,v in methods.items():
        if v['available']=='1':
            require(is_hash(v['worker_sha256']), 'missing worker digest '+m)
    gold = unique(read(root,'goldens.csv'), lambda r:(r['family'],r['mode'],int(r['seed'])))
    admissions = unique(read(root,'qualified.csv'), lambda r:(r['family'],r['method']))
    routes = unique(read(root,'routes.csv'), lambda r:(r['family'],r['method']))
    require(set(admissions)=={(f,m) for f in panel for m in methods},'incomplete admissions')
    require(set(gold)=={(f,mode,seed) for f in panel for mode,seed in FIXTURES},'incomplete references')
    accuracy, timings = defaultdict(dict), defaultdict(dict)
    for target, name in [(accuracy,'accuracy.csv'),(timings,'timings.csv')]:
        for r in read(root,name):
            key=r['family'],r['method']
            require(key in admissions and r['role']==role[key[1]], 'unknown record')
            k=(r['mode'],int(r['seed'])) if target is accuracy else (int(r['seed']),int(r['round']))
            require(k not in target[key], 'duplicate measurement')
            target[key][k]=r
    cases={}; pairs_out=[]
    for f,p in panel.items():
        m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc=(int(p[x]) for x in 'm n k lda ldb ldc ar ac br bc cr cc'.split())
        cells=m*n; guards=((cr+m+2)*lc*4+page-1)//page*page//4-cells
        residues=[4*x%64 for x in (ar*la+ac,br*lb+bc,cr*lc+cc)]
        for mode,seed in FIXTURES:
            g=gold[f,mode,seed]
            require(int(g['bytes'])==8*cells and int(g['checked'])==cells and int(g['scalar_spots'])==23 and
                    0<=float(g['max_spot_error'])<=1e-9 and int(g['cpu'])==cpu and int(g['threads'])==1 and
                    all(is_hash(g[key]) for key in ('a_sha256','b_sha256','reference_sha256')), 'reference '+f)
        participating=[method for method in methods if timings[f,method]]
        for key in BLOCKS:
            seed,r=key
            if not participating: break
            shift=(r-1+list(panel).index(f)) % len(participating)
            order=participating[shift:]+participating[:shift]
            if seed==SEEDS[1]: order=order[::-1]
            for pos,method in enumerate(order,1):
                require(key in timings[f,method] and int(timings[f,method][key]['position'])==pos,
                        'counterbalanced order '+str((f,method,key)))
        times={}; cold={}; errors={}; exclusions={}; pairs={}
        for method, source in methods.items():
            key=f,method
            q=admissions[key]
            require(q['role']==role[method] and q['qualified'] in ('0','1'), 'admission role')
            if q['qualified']=='0':
                exclusions[method]=q['reason']
                continue
            a=accuracy[key]; t=timings[key]
            require(source['available']=='1' and set(a)==FIXTURES and set(t)==set(BLOCKS),'admitted coverage '+str(key))
            for r in [*a.values(),*t.values()]:
                golden=gold[f,r['mode'],int(r['seed'])]
                require(all(p[col]==r[col] for col in p) and r['passed']=='1' and int(r['checked'])==cells and
                        int(r['guards'])==guards and [int(r[x]) for x in ('a_mod64','b_mod64','c_mod64')]==residues and
                        int(r['cpu'])==cpu and int(r['threads'])==1 and 0<=float(r['max_abs'])<=.002 and
                        r['a_sha256']==golden['a_sha256'] and r['b_sha256']==golden['b_sha256'] and
                        is_hash(r['c_sha256']), 'admitted evidence '+str(key))
                require(all(math.isfinite(float(r[z])) and float(r[z])>0 for z in ('first_s','first_cpu_s')), 'first time')
            for b,r in t.items():
                require(r['mode']=='random' and int(r['position'])>0 and
                        all(math.isfinite(float(r[z])) and float(r[z])>0 for z in ('warm_s','warm_cpu_s')), 'warm time')
            if source['trace_worker_sha256']:
                route=routes[key]
                require(route['verified']=='1' and route['ordinary_sha256']==route['trace_sha256']==
                        a['random',SEEDS[0]]['c_sha256'],'route '+str(key))
            times[method]=median(float(t[b]['warm_s']) for b in BLOCKS)
            cold[method]=median(float(a['random',s]['first_s']) for s in SEEDS)
            errors[method]=max(float(r['max_abs']) for r in [*a.values(),*t.values()])
        if CAND in times:
            for peer in times:
                if peer==CAND:continue
                pair=comparison(timings[f,peer],timings[f,CAND]);pairs[peer]=pair
                pairs_out.append([f,peer,role[peer],*pair['ratios'],pair['median'],pair['wins'],pair['ties'],*pair['seed_medians']])
        group={
            'independent_oneshot': lambda m:role[m]=='independent-oneshot',
            'independent_any': lambda m:role[m].startswith('independent-'),
            'vendor_assisted': lambda m:role[m]=='vendor-assisted',
            'whole_oneshot': lambda m:role[m]!='independent-retained',
            'whole_any': lambda m:True}
        tiers={}
        for tier,member in group.items():
            available=[v for v in times if member(v)]
            others=[v for v in available if v!=CAND]
            leader=min(available,key=lambda v:(times[v],v)) if available else None
            rival=min(others,key=lambda v:(times[v],v)) if others else None
            mandatory = (required_peers if tier.startswith('independent') else required_assisted if tier=='vendor_assisted'
                         else required_peers|required_assisted|{BASE})
            mandatory = set(mandatory)
            if (tier.startswith('independent') or tier.startswith('whole')) and methods.get('onednn-oneshot',{}).get('available')=='1':
                mandatory.add('onednn-oneshot')
            if tier.endswith('_any'):
                mandatory.add('onednn-cached-plan')
            # Unaligned two-level release configurations enter the field whenever they built.
            if tier=='vendor_assisted' or tier.startswith('whole'):
                mandatory.update(v for v in methods if v.startswith('release-') and methods[v]['available']=='1')
            missing=sorted(mandatory-set(times))
            metric=pairs.get(rival)
            tiers[tier]={'leader':leader,'fastest_other':rival,'missing_required':missing,
                         'excluded_or_unavailable':{v:exclusions[v] for v in exclusions if member(v)},
                         'candidate_lower_than_fastest_rival':bool(rival and CAND in times and times[CAND]<times[rival]),
                         'candidate_tied_with_fastest_rival':bool(rival and CAND in times and times[CAND]==times[rival]),
                         'paired_median':metric['median'] if metric else None,
                         'paired_wins':metric['wins'] if metric else None,
                         'passes':bool(metric and metric['median']>1.02 and metric['wins']>=5 and not missing)}
        cases[f]=dict(shape=[m,n,k],qualified=list(times),exclusions=exclusions,
                      warm_median_s=times,first_median_s=cold,max_abs=errors,tiers=tiers,pairs=pairs)
    fresh=[f for f in panel if f.startswith('fresh-')];control='repeat-divisible-winner'
    matched=[cases[f]['pairs'].get(BASE) for f in fresh]
    all_valid=all(BASE in c['qualified'] and CAND in c['qualified'] for c in cases.values())
    parity=all(accuracy[control,CAND].get(x,{}).get('c_sha256') and
               accuracy[control,CAND][x]['c_sha256']==accuracy[control,BASE].get(x,{}).get('c_sha256') for x in FIXTURES)
    control_ratio=cases[control]['pairs'].get(BASE,{}).get('median')
    primary={'complete':all_valid,'control_bit_identical':bool(parity),'control_paired_median':control_ratio,
             'passes':False}
    if all(matched):
        med=[q['median'] for q in matched]
        blocks=[geometric_mean(q['seed_medians'][j] for q in matched) for j in range(2)]
        gm=geometric_mean(med);win=sum(v>1 for v in med);minimum=min(med)
        primary.update(geomean=gm,layout_wins=win,minimum=minimum,seed_geomeans=blocks,
                       paired_wins=sum(q['wins'] for q in matched),paired_count=6*len(fresh),
                       passes=bool(all_valid and parity and control_ratio is not None and control_ratio>=.97
                                   and gm>=1.03 and win>=2 and minimum>=.98 and min(blocks)>1))
    fields={}
    for tier in cases[control]['tiers']:
        t=[cases[f]['tiers'][tier] for f in fresh];v=[q['paired_median'] for q in t]
        fields[tier]={'passes':all(q['passes'] for q in t),'layouts_meeting_criterion':sum(q['passes'] for q in t),
                      'all_three_geomean':geometric_mean(v) if all(x is not None for x in v) else None,
                      'candidate_lowest_raw_medians':sum(q['candidate_lower_than_fastest_rival'] for q in t),
                      'missing_required':{f:q['missing_required'] for f,q in zip(fresh,t) if q['missing_required']}}
    with (root/'paired.csv').open('w',newline='') as file:
        wr=csv.writer(file,lineterminator='\n')
        wr.writerow(['family','peer','role',*[f'seed{j}_r{i}' for j in (1,2) for i in (1,2,3)],
                     'median','wins','ties','seed1_median','seed2_median'])
        wr.writerows(pairs_out)
    result=dict(architecture=arch,model=host.get('Model name'),cpu=cpu,panel_sha256=PANEL_SHA,
                methods=role,primary=primary,field=fields,cases=cases)
    (root/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'primary':primary,'field':fields},allow_nan=False))


if __name__=='__main__':
    score(Path(sys.argv[1]))
