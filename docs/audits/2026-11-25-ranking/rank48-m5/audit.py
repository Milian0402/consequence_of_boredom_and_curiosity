"""Verify the native Apple paired measurements and component timings from raw CSVs."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

def rows(f):
    with f.open(newline='') as io:return list(csv.DictReader(io))
def need(test,msg):
    if not test:raise ValueError(msg)
def audit(root):
    seeds=(35485721,93392083);methods=('accelerate','cob-release','form48','form49')
    need(hashlib.sha256((root/'panel.csv').read_bytes()).hexdigest()=='21aca48602498f5413ffea99197d078eec5fcede5580253dcbf9068dfdd65aeb','wrong panel')
    panel={x['family']:tuple(int(x[k]) for k in ('m','n','k','pad')) for x in rows(root/'panel.csv')}
    need(len(panel)==4,'four layouts')
    cert=(root/'certificates.txt').read_text().splitlines()
    need(len(cert)==2 and all('verified all 4096 bilinear tensor coefficients over the rationals' in x for x in cert),'rational certificates')
    route=rows(root/'routes.csv')
    need(len(route)==8 and len({(r['rank'],r['m'],r['n'],r['k']) for r in route})==8,'route coverage')
    for r in route:
        m,n,k,p=(int(r[x]) for x in ('m','n','k','pad'))
        need((m,n,k,p) in panel.values() and int(r['core_calls'])==int(r['rank']) and int(r['rank']) in [48,49] and
             int(r['fringe_calls'])==(m%4!=0)+(n%4!=0) and int(r['checked'])==m*n and
             r['errors']=='0' and r['inputs_unchanged']==r['route_verified']=='1','observed route')
    a=rows(root/'accuracy.csv');t=rows(root/'timings.csv')
    qual=defaultdict(dict);times=defaultdict(dict);fail=[]
    cells=0
    for r in a:
        key=(r['family'],r['implementation']);m,n,k,pad=panel[key[0]]
        details=dict(z.split('=',1) for z in r['detail'].split(','))
        mode=r['mode'],int(r['seed']); value=float(r['max_abs_error'])
        need(key[1] in methods and mode not in qual[key] and math.isfinite(value) and 0<=value and
             int(r['checked'])==m*n and r['nonfinite']==r['guard_changes']==details['input_changes']=='0' and
             r['cpu']=='unpinned' and r['threads']==details['threads']=='1' and
             math.isclose(max(abs(float(details[q])) for q in ('signed_min_error','signed_max_error')),value,rel_tol=1e-10),'incorrect accuracy row')
        need(int(r['verified'])==int(value<=.002),'incorrect accuracy decision')
        qual[key][mode]=value
        cells+=m*n
        if value>.002:fail.append(dict(family=key[0],method=key[1],fixture=mode[0],seed=mode[1],max_abs_error=r['max_abs_error']))
    for r in t:
        key=(r['family'],r['implementation']);group=int(r['block']),int(r['round'])
        sec=float(r['seconds']);caller=float(r['caller_cpu_seconds']);proc=float(r['process_cpu_seconds'])
        need(key[0] in panel and key[1] in methods and group not in times[key] and r['calls']=='1' and
             int(r['seed'])==seeds[group[0]-1] and r['cpu']=='unpinned' and r['threads']=='1' and
             all(0<x<math.inf for x in (sec,caller,proc)),'malformed timing')
        times[key][group]=sec
    result=[]
    fixtures={('random',seeds[0]),('random',seeds[1]),('cancel',seeds[0]),('scaled',seeds[0])}
    groups=sorted((b,r) for b in (1,2) for r in (1,2,3))
    for f in panel:
        med={}; valid={}
        for name in methods:
            ws=times[f,name];v=qual[f,name]
            need(set(ws)==set(groups) and set(v)==fixtures,'incomplete calls/fixtures')
            med[name]=1000*median(ws.values());valid[name]=max(v.values())<=.002
        pairs={}
        for other in ('form49','accelerate','cob-release'):
            rs=[times[f,other][g]/times[f,'form48'][g] for g in groups]
            pairs[other]=dict(median=median(rs),wins=sum(x>1 for x in rs),ratios=rs,qualified=valid[other] and valid['form48'])
        result.append(dict(family=f,shape=panel[f],median_ms=med,qualified=valid,form48_vs=pairs))
    cpu={k:[min(float(r[k])/float(r['seconds']) for r in t),max(float(r[k])/float(r['seconds']) for r in t)] for k in ('caller_cpu_seconds','process_cpu_seconds')}
    components={}
    if (root/'components.csv').exists():
        raw=rows(root/'components.csv')
        need(len(raw)==16 and len({(r['series'],r['call']) for r in raw})==16,'profile coverage')
        for rank in (48,49):
            warm=[r for r in raw if int(r['rank'])==rank and int(r['call'])>0];need(len(warm)==6,'warm component coverage')
            for r in warm:
                need(abs(sum(float(r[k]) for k in ('alloc_seconds','input_transforms_seconds','core_leaf_seconds','recombine_seconds','free_and_tails_seconds'))-float(r['wall_seconds']))<=2e-8,'phase sum')
            components[rank]={k:1000*median(float(r[k]) for r in warm) for k in ('input_transforms_seconds','core_leaf_seconds','recombine_seconds','free_and_tails_seconds','wall_seconds')}
    return dict(cases=result,failures=fail,checked=cells,timings=len(t),fixtures=len(a),cpu_ratios=cpu,components=components)
if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1])),indent=2))
