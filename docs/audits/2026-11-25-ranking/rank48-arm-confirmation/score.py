"""Rebuild all numerical admissions, actual routes and six-round decisions."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

DIMS=('m','n','k','pad'); SEEDS=(73918261,498113807)
CASES={'fresh-deep-odd':(2333,3119,3413,7),'fresh-wide-odd':(3019,2657,3347,9),'compact-control':(3136,3200,3072,0)}
METHODS=['openblas','armpl','form48','form49','release-openblas','fastmmw-768','fastmmw-1536']
PRIMARY=set(METHODS)-{'form48','form49'}
FIX={('random',1,SEEDS[0]),('random',2,SEEDS[1]),('cancel',0,SEEDS[0]),('scaled',0,SEEDS[0])}
PAIRS=[(b,j) for b in (1,2) for j in (1,2,3)]

def read(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))

def need(ok,why):
    if not ok: raise ValueError(why)

def dims(r):return tuple(int(r[d]) for d in DIMS)

def score(path):
    need({r['family']:dims(r) for r in read(path/'panel.csv')}==CASES,'panel')
    need(hashlib.sha256((path/'protocol.md').read_bytes()).hexdigest()=='d7308bdb31d8876e2fe125ad9e5a454c2bb5122c7cba619df65c98193210b588','frozen scoring protocol')
    expected=set(CASES.values()); cpus=set(); required=set()
    for r in read(path/'routes.csv'):
        m,n,k,pad=dims(r);rank=int(r['rank']);key=dims(r),rank
        need(key not in required and rank in (48,49) and int(r['core_calls'])==rank and int(r['fringe_calls'])==(m%4!=0)+(n%4!=0) and int(r['checked'])==m*n and r['errors']=='0' and r['inputs_unchanged']==r['route_verified']=='1','finite-rank route');required.add(key)
    need(required=={(s,r) for s in expected for r in (48,49)},'finite route coverage')
    required=set()
    for r in read(path/'fastmmw-routes.csv'):
        m,n,k,pad=d=dims(r);name=r['implementation'];key=d,name;cutoff=int(name.split('-')[1]);factor=1;call_count=1
        while (min(m,n,k)+factor-1)//factor>cutoff: factor*=2;call_count*=7
        need(key not in required and name in ('fastmmw-768','fastmmw-1536') and r['passed']=='1' and int(r['leaf_calls'])==call_count and int(r['padded'])==any(s%factor for s in (m,n,k)) and r['leaf']=='openblas','FastMMW route');required.add(key)
    need(required=={(s,n) for s in expected for n in ('fastmmw-768','fastmmw-1536')},'FastMMW coverage')
    required=set()
    for r in read(path/'release-routes.csv'):
        m,n,k,pad=d=dims(r);need(d not in required and int(r['core_calls'])==int(r['core_overwrites'])==49 and int(r['calls'])==49+(m%64!=0)+(n%64!=0)+(k%32!=0) and int(r['checked'])==m*n and all(r[f]=='1' for f in ('output_exact','inputs_unchanged','guards','threads','edges_ok','quadrants_ok','provider_ok','passed')),'release route');cpus.add(r['cpu']);required.add(d)
    need(required==expected,'release coverage')
    cert=(path/'certificates.txt').read_text();need(cert.count('4096 bilinear tensor coefficients over the rationals')==2,'tensor certification')
    times=defaultdict(dict);errors=defaultdict(dict);fail=[]
    for r in read(path/'accuracy.csv'):
        f,n=key=r['family'],r['implementation'];token=r['mode'],int(r['block']),int(r['seed']);d=dict(v.split('=',1) for v in r['detail'].split(',') if '=' in v); e=float(r['max_abs_error']);first=float(d['first_sgemm_seconds'])
        need(r['phase']=='rank48-arm-confirm' and dims(r)==CASES[f] and n in METHODS and token in FIX and token not in errors[key] and int(r['checked'])==int(r['m'])*int(r['n']) and r['threads']==d['threads']=='1' and r['cpu']==d['cpu'] and r['guard_changes']==r['nonfinite']==d['input_changes']=='0' and math.isfinite(e) and e>=0 and math.isfinite(first) and first>0 and ((r['verified']=='1')==(e<=.002)),'FP64 integrity '+str((key,token)))
        cpus.add(r['cpu']); errors[key][token]=e
        if e>.002: fail.append(dict(family=f,method=n,mode=token[0],seed=token[2],max_abs_error=e))
    for r in read(path/'timings.csv'):
        f,n=key=r['family'],r['implementation'];b,j=pair=int(r['block']),int(r['round']); t=float(r['seconds']);pos=int(r['position']);shift=(list(CASES).index(f)+j-1)%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
        if (b+j-2)%2:order.reverse()
        need(dims(r)==CASES[f] and r['phase']=='rank48-arm-confirm' and n in METHODS and pair in PAIRS and pair not in times[key] and int(r['seed'])==SEEDS[b-1] and order[pos-1]==n and r['threads']=='1' and r['calls']=='1' and math.isfinite(t) and t>0,'warm-call integrity');cpus.add(r['cpu']);times[key][pair]=t
    need(len(cpus)==1 and bool(fail)==bool(int((path/'study-status.txt').read_text())),'unexplained study failure')
    summary=[];gates=[];qual={};ranks={};primary_complete=True
    for f,shape in CASES.items():
        eligible={}
        for n in METHODS:
            need(set(errors[f,n])==FIX,'missing numerical reference '+str((f,n)));qual[f,n]=max(errors[f,n].values())<=.002
            if qual[f,n]:
                need(set(times[f,n])==set(PAIRS),'missing six warm calls '+str((f,n)));eligible[n]=median(times[f,n].values())
        primary_complete &= {'form48',*PRIMARY}.issubset(eligible)
        ranks[f]=sorted(eligible,key=eligible.get)
        for n in METHODS:
            if n=='form48':continue
            xs=[times[f,n][p]/times[f,'form48'][p] for p in PAIRS] if n in eligible and 'form48' in eligible else [];wins=sum(x>1 for x in xs) if xs else None
            gain=median(xs) if xs else None
            summary.append(dict(family=f,m=shape[0],n=shape[1],k=shape[2],pad=shape[3],candidate='form48',peer=n,rank48_ms=1000*eligible['form48'] if 'form48' in eligible else '',peer_ms=1000*eligible[n] if n in eligible else '',rank48_max_abs=max(errors[f,'form48'].values()),peer_max_abs=max(errors[f,n].values()),pairs=len(xs),paired_speedup=gain,wins=wins,ratios=json.dumps(xs),primary=n in PRIMARY,fastest_qualified=ranks[f][0]))
            if f.startswith('fresh-') and n in PRIMARY:gates.append(bool(xs) and gain>1 and wins>=5)
    with (path/'scores.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    decision=dict(primary_gate=primary_complete and len(gates)==10 and all(gates),primary_complete=primary_complete,qualifying_primary_head_to_heads=sum(gates),primary_head_to_heads=len(gates),pinned_cpu=int(next(iter(cpus))),first_to_last=ranks,fp64_fixtures=sum(map(len,errors.values())),fp64_outputs=sum(CASES[f][0]*CASES[f][1]*len(es) for (f,n),es in errors.items()),warm_calls=sum(map(len,times.values())),numeric_exclusions=fail)
    (path/'decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n'); print(json.dumps(decision,allow_nan=False))

if __name__=='__main__': score(Path(sys.argv[1]))
