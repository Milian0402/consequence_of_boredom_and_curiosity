"""Validate every stream-B worker observation and score the frozen five-shape gate."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median
import re
import sys

SEEDS = (348271759, 710438957)
PAIRS = [(i, j) for i in (1, 2) for j in (1, 2, 3)]
FIXTURES = {('random', 1, SEEDS[0]), ('random', 2, SEEDS[1]),
            ('cancel', 0, SEEDS[0]), ('scaled', 0, SEEDS[0])}
FROZEN = '466c2b51ee6df5fd737f3f4b1977c5b4ad75cdb7ae61bbaec7bc5919530f18fd'
BUILDS = {'stream-native': 'stream-incumbent', 'stream-avx2': 'stream-avx2-incumbent',
          'stream-avx512': 'stream-avx512-incumbent'}
CACHED = 'onednn-cached-plan'


def need(ok, what):
    if not ok:
        raise ValueError(what)


def read(root, filename):
    with (root / filename).open(newline='') as stream:
        return list(csv.DictReader(stream))


def shape(r):
    return tuple(int(r[f]) for f in ('m', 'n', 'k', 'pad'))


def score(root):
    need(hashlib.sha256((root/'panel.csv').read_bytes()).hexdigest() == FROZEN, 'changed panel')
    panel = {r['family']: shape(r) for r in read(root, 'panel.csv')}
    fresh = [f for f in panel if f.startswith('fresh-')]
    inventory = read(root, 'eligibility.csv')
    roles = {r['implementation']: r['role'] for r in inventory if r['status'] == 'included'}
    external = {n for n in roles if n not in {*BUILDS, *BUILDS.values()}}
    need(len(fresh) == 5 and {'stream-native', 'stream-incumbent', 'openblas', 'mkl', 'mkl-jit', 'blasfeo', 'onednn-oneshot', CACHED}.issubset(roles), 'missing native pair or required peer')
    build_rows = read(root, 'builds.csv')
    builds = {r['implementation']:r for r in build_rows}
    need(len(build_rows)==len(builds)==6 and set(builds)=={*BUILDS,*BUILDS.values()}, 'build inventory')
    for n,r in builds.items():
        need(r['status'] in ('qualified','unsupported') and (n in roles)==(r['status']=='qualified'), f'build admission {n}')
        if n in roles:
            need(r['isa'] in ('avx2','avx512') and r['variant']==('candidate' if n in BUILDS else 'incumbent') and all(re.fullmatch('[0-9a-f]{64}',r[k]) for k in ('worker_sha256','trace_worker_sha256','source_sha256','object_sha256','trace_source_sha256')), f'build identity {n}')
    for c, old in BUILDS.items():
        need((c in roles) == (old in roles), f'unmatched ISA build {c}')
    eligible = { (r['family'], r['implementation']):r['qualified'] == '1' for r in read(root, 'case-eligibility.csv') }
    expected = {(f,n) for f in panel for n in roles}
    need(set(eligible) == expected, 'case-eligibility coverage')
    ranking = json.loads((root/'ranking.json').read_text())
    need(ranking['phase'] == 'stream-b' and ranking['seeds'] == list(SEEDS) and ranking['rounds_per_seed'] == 3 and set(ranking['implementations']) == set(roles), 'study protocol')
    cpus = set()
    trace = {}
    for r in read(root, 'routes.csv'):
        f,n = key = (r['family'],r['implementation']); m,w,k,_ = shape(r)
        need(key not in trace and panel[f] == shape(r) and n in roles and n not in external, f'route identity {key}')
        route = n in BUILDS and f != 'outside-width'
        old = not route and m >= 64 and k >= 128
        blocked = not route and not old
        mr,nr,packs,tiles = (int(r[a]) for a in ('mr','nr','packs','tiles'))
        need(int(r['selected']) == route and r['passed'] == '1' and int(r['old_medium']) == int(r['expected_old_medium']) == old and int(r['blocked']) == int(r['expected_blocked']) == blocked, f'dispatch {key}')
        if route:
            need(r['isa']==builds[n]['isa'] and (mr,nr)==((3,32) if r['isa']=='avx2' else (6,64)) and (n != 'stream-avx2' or mr == 3) and (n != 'stream-avx512' or mr == 6) and packs == (w+nr-1)//nr and tiles == packs*((m+mr-1)//mr), f'tile counts {key}')
        else:
            need((mr,nr,packs,tiles) == (0,0,0,0), f'bypass counts {key}')
        need(int(r['linked_checked']) == int(r['trace_checked']) == m*w and r['threads'] == '1' and r['inputs_unchanged'] == r['guards'] == '1', f'fixture mismatch {key}')
        cpus.add(r['cpu']); trace[key] = True
    need(set(trace) == expected-{k for k in expected if k[1] in external}, 'missing project route')
    checks, times, batches = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for r in read(root, 'accuracy.csv'):
        f,n = key = r['family'],r['implementation']; token=r['mode'],int(r['block']),int(r['seed'])
        d = dict(x.split('=',1) for x in r['detail'].split(',') if '=' in x)
        e = float(r['max_abs_error']); m,w,_,_=panel[f]
        need(key in expected and shape(r)==panel[f] and token in FIXTURES and token not in checks[key], f'fixture identity {key}')
        need(r['threads']==d['threads']=='1' and r['cpu']==d['cpu'] and int(r['checked'])==m*w and r['guard_changes']==r['nonfinite']==d['input_changes']=='0' and math.isfinite(e) and e>=0 and ((r['verified']=='1') == (e<=0.002)), f'fixture integrity {key}')
        need(math.isfinite(float(d['first_sgemm_seconds'])) and float(d['first_sgemm_seconds'])>0 and (not n.startswith('onednn-') or d.get('onednn_impl')), f'first call or oneDNN route {key}')
        checks[key][token]=e; cpus.add(r['cpu'])
    for r in read(root, 'timings.csv'):
        f,n = key = r['family'],r['implementation']; b,j=int(r['block']),int(r['round']); t=float(r['seconds'])
        need(key in expected and shape(r)==panel[f] and (b,j) in PAIRS and int(r['seed'])==SEEDS[b-1] and (b,j) not in times[key] and r['threads']=='1' and int(r['calls'])>0 and math.isfinite(t) and t>0, f'timing integrity {key}')
        times[key][b,j]=t; batches[key][b,j]=int(r['calls']); cpus.add(r['cpu'])
    need(len(cpus)==1 and next(iter(cpus)).isdigit(), 'not one pinned CPU')
    for key in sorted(expected):
        complete=set(checks[key])==FIXTURES and max(checks[key].values())<=0.002 and set(times[key])==set(PAIRS)
        need(complete==eligible[key],f'qualification {key}')
    output=[]; decisions={}
    for c,old in BUILDS.items():
        if c not in roles: continue
        primaries=[]
        for f in panel:
            ok={n for n in roles if eligible[f,n]}
            need({c,old}.issubset(ok) and external.intersection(ok)-{CACHED},f'missing qualified pair {c} {f}')
            elapsed={n:median(times[f,n].values()) for n in ok}
            direct=external.intersection(ok)-{CACHED}
            best=min(direct,key=elapsed.get); best_any=min(external.intersection(ok),key=elapsed.get)
            for n in [old]+sorted(external.intersection(ok)):
                ratios=[times[f,n][key]/times[f,c][key] for key in PAIRS]
                need(batches[f,n]==batches[f,c],f'unmatched calls {f} {n} {c}')
                row=dict(family=f,m=panel[f][0],n=panel[f][1],k=panel[f][2],pad=panel[f][3],candidate=c,peer=n,candidate_ms=1000*elapsed[c],peer_ms=1000*elapsed[n],paired_speedup=median(ratios),wins=sum(x>1 for x in ratios),seed1_median=median(ratios[:3]),seed2_median=median(ratios[3:]),matched_incumbent=int(n==old),fastest_independent_oneshot=int(n==best),fastest_independent_any=int(n==best_any),ratios=json.dumps(ratios,separators=(',',':')))
                output.append(row)
                if f in fresh and n == old: primaries.append(row)
        gm=geometric_mean(x['paired_speedup'] for x in primaries); blocks=[geometric_mean(x[k] for x in primaries) for k in ('seed1_median','seed2_median')]; count=sum(x['paired_speedup']>1 for x in primaries); floor=min(x['paired_speedup'] for x in primaries)
        decisions[c]=dict(fresh_shapes=len(primaries),geometric_mean=gm,seed_geometric_means=blocks,shape_wins=count,minimum_paired_speedup=floor,gate_passed=gm>=1.03 and min(blocks)>1 and count>=4 and floor>=0.98)
    with (root/'stream-summary.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=output[0]);w.writeheader();w.writerows(output)
    payload=dict(panel_sha256=FROZEN,cpu=int(next(iter(cpus))),primary='stream-native',seeds=SEEDS,decisions=decisions,unqualified=[list(k) for k in sorted(expected) if not eligible[k]],omitted=[r for r in inventory if r['status']!='included'])
    (root/'stream-score.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')

if __name__ == '__main__':
    score(Path(sys.argv[1]))
