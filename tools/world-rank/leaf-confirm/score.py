"""Replay the registered leaf confirmation timings and per-host gates. Run with OUT."""
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
SEEDS = (197906831, 884039527)
FIXTURES = {('random', SEEDS[0]), ('random', SEEDS[1]), ('cancel', SEEDS[0]), ('scaled', SEEDS[1])}
BLOCKS = tuple((s, r) for s in SEEDS for r in (1, 2, 3))
PANEL_SHA = '204a37e5fd61c9530f01ec7a95e01ea2d09acf719784f198af6cdd6221609eff'
PROTOCOL_SHA = '81d8142ae0feb265c092e9ed3ed26b523b777deb8a4b9ca9fd70a6cb563114ee'
ROLES = {'baseline', 'candidate', 'vendor-assisted', 'hk15-vendor-assisted', 'independent-oneshot', 'independent-retained'}


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
    delta_ms = [1000*(float(peer[k]['warm_s'])-float(candidate[k]['warm_s'])) for k in BLOCKS]
    leave_one_out = [median(ratios[:j]+ratios[j+1:]) for j in range(len(ratios))]
    return {'median': median(ratios), 'ratios': ratios, 'delta_ms':delta_ms,
            'median_delta_ms':median(delta_ms),'leave_one_out_range':[min(leave_one_out),max(leave_one_out)],
            'wins': sum(v > 1 for v in ratios),
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
    required_assisted = {f'release-{p}-{policy}' for p in ('openblas',provider) for policy in ('two','aligned-two')}
    contenders = (CAND,'hk15-openblas-leaf',f'hk15-{provider}-leaf')
    direct = required_peers-{'blasfeo','fastmmw-768','fastmmw-1536'}
    cpu = int((root/'pinned-cpu.txt').read_text())
    page = int((root/'page-bytes.txt').read_text())
    methods = unique(read(root, 'sources/methods.csv'), lambda r:r['method'])
    role = {m:v['role'] for m,v in methods.items()}
    require(set(role.values()) <= ROLES and role[BASE]=='baseline' and role[CAND]=='candidate', 'method identities')
    require(all(role[m]=='independent-oneshot' for m in required_peers) and
            all(role[m]=='vendor-assisted' for m in required_assisted), 'missing required method identity')
    if arch=='x86_64':
        require(role.get('mkl-handle-oneshot')=='independent-oneshot' and
                role.get('mkl-handle-prepared')=='independent-retained','explicit Intel handle roles')
    for contender in contenders[1:]:
        require(role.get(contender)=='hk15-vendor-assisted', 'unregistered leaf role '+contender)
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
            if method.startswith('onednn-'):
                retained=method=='onednn-cached-plan'
                for r in [*a.values(),*t.values()]:
                    require(r['implementation'] and not r['implementation'].lower().startswith('ref') and
                            int(r['cached'])==int(retained) and int(r['plans'])==(1 if retained or r['round']=='0' else 2),
                            'oneDNN plan identity '+str(key))
            if method.startswith('mkl-handle-'):
                retained=method=='mkl-handle-prepared'
                for r in [*a.values(),*t.values()]:
                    require(r['implementation'] in ('explicit-generated','explicit-fallback') and
                            int(r['cached'])==int(retained) and int(r['plans'])==(1 if retained or r['round']=='0' else 2),
                            'explicit Intel handle identity '+str(key))
            if source['trace_worker_sha256']:
                route=routes[key]
                require(route['verified']=='1' and route['ordinary_sha256']==route['trace_sha256']==
                        a['random',SEEDS[0]]['c_sha256'],'route '+str(key))
            times[method]=median(float(t[b]['warm_s']) for b in BLOCKS)
            cold[method]=median(float(a['random',s]['first_s']) for s in SEEDS)
            errors[method]=max(float(r['max_abs']) for r in [*a.values(),*t.values()])
        missing_by_tier = {}
        sets = {
            'old_standalone':{BASE},
            'direct':direct,
            'independent_oneshot':{m for m,r in role.items() if r=='independent-oneshot'},
            'independent_any':{m for m,r in role.items() if r.startswith('independent-')},
            'released_winograd':required_assisted,
            'whole_oneshot':{m for m,r in role.items() if r!='independent-retained'},
            'whole_any':set(methods),
        }
        required_one=set(required_peers)
        if methods.get('onednn-oneshot',{}).get('available')=='1':
            required_one.add('onednn-oneshot')
        required={
            'old_standalone':{BASE},'direct':direct,'independent_oneshot':required_one,
            'independent_any':required_one|{'onednn-cached-plan'},'released_winograd':required_assisted,
            'whole_oneshot':required_one|required_assisted|{CAND,BASE},
            'whole_any':required_one|required_assisted|{CAND,BASE,'onednn-cached-plan'},
        }
        for tier in sets:
            missing_by_tier[tier]=sorted(required[tier]-set(times))
        contenders_out={}
        for contender in contenders:
            peers={}
            if contender in times:
                for peer in times:
                    if peer==contender:continue
                    record=comparison(timings[f,peer],timings[f,contender])
                    peers[peer]=record
                    pairs_out.append([f,contender,peer,role[peer],*record['ratios'],*record['delta_ms'],
                                      record['median'],record['median_delta_ms'],record['wins'],record['ties'],
                                      *record['seed_medians'],*record['leave_one_out_range']])
            outcomes={}
            for tier,members in sets.items():
                choices=sorted((m for m in members if m in times and m!=contender),key=lambda m:(times[m],m))
                rival=choices[0] if choices else None
                including=sorted([*choices,*([contender] if contender in times else [])],key=lambda m:(times[m],m))
                stats=peers.get(rival)
                faster=bool(stats and times[contender]<times[rival])
                outcomes[tier]={
                    'fastest_other':rival,'raw_leader':including[0] if including else None,
                    'raw_runner_up':including[1] if len(including)>1 else None,
                    'raw_winner_ties':[v for v in including if times[v]==times[including[0]]] if including else [],
                    'candidate_strictly_fastest':faster,'missing_required':missing_by_tier[tier],
                    'paired_median':stats['median'] if stats else None,'wins':stats['wins'] if stats else None,
                    'passes':bool(stats and not missing_by_tier[tier] and faster and stats['median']>1.02 and stats['wins']>=5)}
            own_provider=contender.split('-')[1] if contender!=CAND else None
            named={
                'versus_standalone':peers.get(CAND),
                'versus_own_direct':peers.get(own_provider),
                'versus_own_released_aligned':peers.get(f'release-{own_provider}-aligned-two'),
                'versus_own_released_unaligned':peers.get(f'release-{own_provider}-two')
            } if own_provider else {}
            contenders_out[contender]=dict(admitted=contender in times,tiers=outcomes,pairs=peers,named=named)
        implementations={
            x:sorted({r['implementation'] for r in timings[f,x].values() if r.get('implementation')})
            for x in times if x.startswith(('mkl-handle-','onednn-'))}
        cases[f]=dict(measured_implementations=implementations,shape=[m,n,k],layout=p['layout'],qualified=list(times),exclusions=exclusions,
                      warm_median_s=times,first_median_s=cold,max_abs=errors,entries=contenders_out)
    groups={tag:[f for f in panel if f.startswith(tag+'-')] for tag in ('repeat','fresh','anchor')}
    require([len(groups[k]) for k in ('repeat','fresh','anchor')]==[3,3,1], 'wrong cohort sizes')
    aggregate={}
    for contender in contenders:
        aggregate[contender]={}
        for group in ('repeat','fresh'):
            results={}
            for tier in sets:
                entries=[cases[f]['entries'][contender] for f in groups[group]]
                selected=[e['tiers'][tier] for e in entries]
                pair_records=[e['pairs'][x['fastest_other']] for e,x in zip(entries,selected)
                              if x['paired_median'] is not None]
                full=len(pair_records)==3
                block_gm=[geometric_mean(p['seed_medians'][j] for p in pair_records) for j in range(2)] if full else None
                results[tier]={
                    'passes':bool(full and all(x['passes'] for x in selected) and min(block_gm)>1),
                    'paired_geomean':geometric_mean(p['median'] for p in pair_records) if full else None,
                    'seed_geomeans':block_gm,'ratio_median_wins':sum(p['median']>1 for p in pair_records),
                    'raw_layout_wins':sum(x['candidate_strictly_fastest'] for x in selected),
                    'pairs_won':sum(p['wins'] for p in pair_records),'pairs_available':6*len(pair_records),
                    'layouts_meeting_criterion':sum(x['passes'] for x in selected),
                    'missing_required':{f:x['missing_required'] for f,x in zip(groups[group],selected) if x['missing_required']}}
            aggregate[contender][group]=results
        aggregate[contender]['full_field_passes_both']=all(aggregate[contender][g]['whole_any']['passes'] for g in ('repeat','fresh'))
    with (root/'paired.csv').open('w',newline='') as file:
        wr=csv.writer(file,lineterminator='\n')
        select=[f'seed{i}_r{j}' for i in (1,2) for j in (1,2,3)]
        wr.writerow(['family','contender','peer','peer_role',*[x+'_ratio' for x in select],*[x+'_delta_ms' for x in select],
                     'paired_median','median_delta_ms','wins','ties','seed1_median','seed2_median','loo_median_min','loo_median_max'])
        wr.writerows(pairs_out)
    result=dict(architecture=arch,model=host.get('Model name'),cpu=cpu,primary=CAND,panel_sha256=PANEL_SHA,
                methods=role,cohorts=groups,gates=aggregate,cases=cases)
    (root/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:{'both_full':v['full_field_passes_both'],'repeat':v['repeat']['whole_any'],'fresh':v['fresh']['whole_any']}
                      for m,v in aggregate.items()},allow_nan=False))


if __name__=='__main__':
    score(Path(sys.argv[1]))
