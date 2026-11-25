"""Recompute the finite-schedule screen from per-observation records (Python stdlib)."""
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

SEEDS=(35485721,93392083)
KEYS=[(b,j) for b in (1,2) for j in (1,2,3)]
FIX={('random',1,SEEDS[0]),('random',2,SEEDS[1]),('scaled',0,SEEDS[0]),('cancel',0,SEEDS[0])}
DIMS=('m','n','k','pad')
PANELS={'aligned-square':(2048,2048,2048,0),'deep-odd-repeat':(2039,2969,3367,11),'rectangular-screen':(2869,2423,3209,3),'large-square':(3328,3328,3328,0)}


def need(value, message):
    if not value: raise ValueError(message)


def read(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))


def dims(r): return tuple(int(r[f]) for f in DIMS)


def paired(family, peer, times):
    ratios=[times[family,peer][k]/times[family,'form48'][k] for k in KEYS]
    return median(ratios),sum(r>1 for r in ratios),ratios


def audit(root, arch):
    panel=read(root/'panel.csv');need(dict((r['family'],dims(r)) for r in panel)==PANELS,'frozen panel changed')
    provider=(root/'leaf.txt').read_text().strip(); peers=['mkl','mkl-jit','openblas','aocl'] if arch=='x86_64' else ['openblas','armpl']
    need(provider==('mkl' if arch=='x86_64' else 'openblas'),'leaf mismatch')
    methods=['form48','form49',*peers]; cpu=set(); coverage={}; shape_to_name={v:k for k,v in PANELS.items()}
    for r in read(root/'routes.csv'):
        m,n,k,pad=geom=dims(r);rank=int(r['rank']);f=shape_to_name[geom];key=f,rank
        need(key not in coverage and rank in (48,49) and int(r['core_calls'])==rank and int(r['fringe_calls'])==int(m%4>0)+int(n%4>0) and int(r['checked'])==m*n and r['errors']=='0' and r['inputs_unchanged']==r['route_verified']=='1','finite route or exact output '+str(key));coverage[key]=r
    need(len(coverage)==8,'incomplete finite routes')
    certificate=(root/'certificates.txt').read_text();need(certificate.count('4096 bilinear tensor coefficients over the rationals')==2 and '48_201-balanced' in certificate and '49_156' in certificate,'runner tensor certificate')
    fixtures=defaultdict(dict);times=defaultdict(dict);first=defaultdict(dict);positions=defaultdict(set);failures=[];cpu_model=re.search(r'^Model name:\s*(.+)$',(root/'cpu.txt').read_text(),re.M)[1].strip()
    for r in read(root/'accuracy.csv'):
        f,n=key=(r['family'],r['implementation']);shape=PANELS[f];token=r['mode'],int(r['block']),int(r['seed']);d=dict(s.split('=',1) for s in r['detail'].split(',') if '=' in s)
        need(r['phase']=='finite-ranks' and dims(r)==shape and n in methods and token in FIX and token not in fixtures[key] and int(r['stress'])==(token[0]!='random'),'unexpected accuracy record')
        err=float(r['max_abs_error']); verified=int(r['verified']);t=float(d['first_sgemm_seconds'])
        need(r['threads']==d['threads']=='1' and d['cpu']==r['cpu'] and r['nonfinite']==r['guard_changes']==d['input_changes']=='0' and int(r['checked'])==shape[0]*shape[1] and math.isfinite(err) and err>=0 and math.isfinite(t) and t>0 and verified==(err<=0.002),'invalid or inconsistent FP64/guard/worker record '+str((key,token)))
        fixtures[key][token]=err; first[key][token]=t; cpu.add(r['cpu'])
        if not verified: failures.append((f,n,*token,err))
    roster=['openblas',*(n for n in peers if n!='openblas'),'form48','form49']
    for r in read(root/'timings.csv'):
        f,n=key=r['family'],r['implementation'];pair=int(r['block']),int(r['round']);pos=int(r['position']);t=float(r['seconds']);cp=r['cpu']
        shift=(list(PANELS).index(f)+pair[1]-1)%len(roster); order=roster[shift:]+roster[:shift]
        if (pair[1]+pair[0]-2)%2: order.reverse()
        need(1<=pos<=len(order) and n==order[pos-1], 'unbalanced method order '+str((key,pair,pos)))
        need(key in fixtures and dims(r)==PANELS[f] and r['phase']=='finite-ranks' and pair in KEYS and pair not in times[key] and int(r['seed'])==SEEDS[pair[0]-1] and int(r['calls'])==1 and r['mode']=='random' and r['stress']=='0' and r['threads']=='1' and math.isfinite(t) and t>0 and pos not in positions[f,pair] and 1<=pos<=len(methods),'invalid warm call '+str((key,pair)));times[key][pair]=t;cpu.add(cp);positions[f,pair].add(pos)
    need(len(cpu)==1 and next(iter(cpu)).isdigit(),'CPU pinning');cpu=next(iter(cpu));summary=[];comparisons=[];per_method=[]
    for f,shape in PANELS.items():
        eligible={}; worst={}; missing=[]
        for n in methods:
            key=f,n;need(set(fixtures[key])==FIX,f'incomplete full-output numerical coverage {key}');e=worst[n]=max(fixtures[key].values());
            if e<=.002:
                need(set(times[key])==set(KEYS),f'incomplete warm timings for numerically eligible method {key}');eligible[n]=median(times[key].values())
            elif n in peers: missing.append(n)
            per_method.append(dict(arch=arch,model=cpu_model,family=f,method=n,qualified=int(n in eligible),max_abs_error=e,checked=shape[0]*shape[1]*4,warm_pairs=len(times[key]),median_ms=1000*eligible[n] if n in eligible else '',first_seed1_ms=1000*first[key]['random',1,SEEDS[0]],first_seed2_ms=1000*first[key]['random',2,SEEDS[1]]))
        best=min((n for n in peers if n in eligible),key=eligible.get,default='');row=dict(arch=arch,model=cpu_model,provider=provider,family=f,**dict(zip(DIMS,shape)),rank48_ms=1000*eligible['form48'] if 'form48' in eligible else '',rank49_ms=1000*eligible['form49'] if 'form49' in eligible else '',rank48_max_abs=worst['form48'],rank49_max_abs=worst['form49'],best_direct=best,best_direct_ms=1000*eligible[best] if best else '',direct_unqualified='|'.join(missing))
        for label,peer in (('rank49','form49'),('best_direct',best)):
            ratio=wins=''
            if 'form48' in eligible and peer in eligible and (not missing or label=='rank49'):
                ratio,wins,_=paired(f,peer,times)
            row[f'{label}_over_rank48']=ratio;row[f'wins_vs_{label}']=wins
        for peer in ['form49',*peers]:
            ratio=wins=arr=''
            if 'form48' in eligible and peer in eligible:
                ratio,wins,xs=paired(f,peer,times);arr=json.dumps(xs,separators=(',',':'))
            comparisons.append(dict(arch=arch,family=f,candidate='form48',peer=peer,speedup=ratio,wins=wins,paired_ratios=arr))
        summary.append(row)
    meta=dict(arch=arch,model=cpu_model,leaf=provider,pinned_cpu=int(cpu),methods=methods,fp64_fixtures=sum(map(len,fixtures.values())),fp64_cells=sum(r['checked'] for r in per_method),warm_observations=sum(map(len,times.values())),routes=len(coverage),route_cells=sum(int(r['checked']) for r in coverage.values()),numerical_failures=failures)
    meta['rank48_median_wins_vs_49']=sum(r['rank49_over_rank48']!='' and r['rank49_over_rank48']>1 for r in summary)
    meta['gate']=meta['rank48_median_wins_vs_49']>=3 and not any(n=='form48' for _,n,*_ in failures)
    return summary, comparisons, per_method, meta
