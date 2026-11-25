"""Audit the registered Arm experiment from preserved CSVs, without rerunning it."""
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import geometric_mean, median

ROOT = Path(sys.argv[1])
METHODS = ('incumbent','candidate','openblas','armpl','blasfeo','libxsmm-dispatch',
           'libxsmm-retained','onednn-oneshot','onednn-cached-plan')
DIRECT = ('openblas','armpl','blasfeo','libxsmm-dispatch','onednn-oneshot')
RETAINED = ('libxsmm-retained','onednn-cached-plan')
SEEDS = (73194513,38641729)
FIXTURES = [('random',SEEDS[0]),('random',SEEDS[1]),('cancel',SEEDS[0]),('scaled',SEEDS[1])]
DIMS = 'm n k lda ldb ldc'.split()


def check(valid, detail):
    if not valid:
        raise ValueError(detail)


def table(name):
    with (ROOT/name).open(newline='') as f:
        data = csv.DictReader(f)
        check(data.fieldnames and len(set(data.fieldnames))==len(data.fieldnames),name+' header')
        result = list(data)
    check(all(None not in r and None not in r.values() for r in result),name+' cell missing')
    return result


def index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        check(key not in result,'duplicate '+str(key))
        result[key] = row
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(r,field):
    value=float(r[field])
    check(math.isfinite(value),f'nonfinite {field}')
    return value


def ratios(peer,candidate):
    return [x/y for x,y in zip(peer,candidate,strict=True)]


def compare(peer,candidate):
    values=ratios(peer,candidate)
    return dict(paired=values,median=median(values),wins=sum(x>1 for x in values))


def save_csv(path,records,fields):
    with (ROOT/path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(records)


panel_sha='5e470c40de6e63b6debbb96afd146eda58de83e50ef3da1e2f19186a8990b4c4'
check(digest(ROOT/'panel.csv')==panel_sha,'panel fingerprint')
check(digest(ROOT/'protocol.md')=='9e173faf888debe5a124549fd77dde62f82a7202fb234d917a704d03a7bb0d13','protocol fingerprint')
identity=json.loads((ROOT/'source.json').read_text())
check(identity['release']=='a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6' and
      identity['patch_sha256']=='71e1c7a3ff3c3b807c0d76b89f2c067fdbd9d96261309b7e390110dcfa3f42be' and
      identity['panel_sha256']==panel_sha,'wrong source or patch')
host=(ROOT/'host.txt').read_text()
check('aarch64' in host.lower(),'native Arm host')
page=int((ROOT/'page-size.txt').read_text())
check(page>=4096 and page&(page-1)==0,'non-power-of-two page size')
binaries=index(table('binaries.csv'),['method'])
check(set(binaries)=={(m,) for m in METHODS[:2]},'binary roster')
hash64=lambda x:re.fullmatch('[a-f0-9]{64}',x) is not None
for key,r in binaries.items():
    check(all(hash64(r[f]) for f in ('archive_sha256','worker_sha256','trace_sha256')),f'binary digest {key}')
    check(r['worker_sha256']!=r['trace_sha256'],'instrumented ordinary executable')
for f in ('worker_sha256','archive_sha256','trace_sha256'):
    check(binaries['candidate',][f]!=binaries['incumbent',][f],'identical candidates '+f)

panel=table('panel.csv')
families=[s['family'] for s in panel]
fresh=[f for f in families if f.startswith('fresh-')]
controls=[f for f in families if f.startswith('control-')]
check(len(panel)==len(set(families))==12 and len(fresh)==8 and len(controls)==2,'frozen families')
fail=index(table('failures.csv'),['family','method'])
raw=table('raw.csv')
routes=index(table('routes.csv'),['family','method'])
parity=index(table('parity.csv'),['family','mode','seed'])
check(set(routes)=={(f,m) for f in families for m in METHODS[:2]},'route inventory')
permitted={(f,m) for f in families for m in METHODS}
check(set(fail)<=permitted and all(r['reason'].strip() for r in fail.values()),'failure inventory')
check(all((r['family'],r['method']) in permitted for r in raw),'unknown method/layout')
cpus={int(r['cpu']) for r in raw}
check(len(cpus)==1 and min(cpus)>=0,'common pinned CPU')
expected_parity={(f,mode,str(seed)) for f in controls for mode,seed in FIXTURES}
check(set(parity)<=expected_parity,'unexpected parity fixture')
all_parity=set(parity)==expected_parity
admitted={}
audit=[]
summaries=[]
layouts={}
for position,s in enumerate(panel):
    family=s['family']
    dims={key:int(s[key]) for key in DIMS}
    m,n,k=dims['m'],dims['n'],dims['k']
    eligible=24<=m<=160 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and m*k<=49152
    for method in METHODS[:2]:
        trace=routes[family,method]
        check(int(trace['public_calls'])==1 and int(trace['args_ok'])==1 and
              int(trace['stream_calls'])==int(method=='candidate' and eligible),'wrong route '+str((family,method)))
    for mode,seed in FIXTURES:
        if family not in controls: continue
        pr=parity.get((family,mode,str(seed)))
        if pr:
            check(int(pr['bytes'])==4*m*n and hash64(pr['sha256']),'control readout')
            all_parity=all_parity and pr['same']=='1'
    guards=((int(s['cr'])+m+2)*dims['ldc']*4+page-1)//page*(page//4)-m*n
    residues=[4*(int(s[v+'r'])*dims['ld'+v]+int(s[v+'c']))%64 for v in 'abc']
    observations={}
    for method_i,method in enumerate(METHODS):
        key=(family,method)
        records=[r for r in raw if (r['family'],r['method'])==key]
        fixed=[r for r in records if int(r['round'])<0]
        timed=[r for r in records if int(r['round'])>=0]
        check([(r['mode'],int(r['seed'])) for r in fixed]==FIXTURES[:len(fixed)] and len(fixed)<=4 and
              [int(r['round']) for r in timed]==list(range(len(timed))) and len(timed)<=6 and
              (not timed or len(fixed)==4),'missing or duplicate admitted observations '+str(key))
        hashes={}
        for r in records:
            check(all(int(r[f])==v for f,v in dims.items()),'incorrect actual matrix '+str(key))
            check([int(r[f+'_mod64']) for f in 'abc']==residues and int(r['checked'])==m*n and
                  int(r['guards'])==guards,'view address, output or guard coverage '+str(key))
            check(r['threads']=='1' and r['clone_blocked']=='1' and 0<=number(r,'error')<=0.002 and
                  number(r,'first_s')>0,'task, FP64 or first-call admission '+str(key))
            code_bytes,plans=int(r['code_bytes']),int(r['plans'])
            impl=r['implementation']
            calls=int(r['calls']); round_id=int(r['round'])
            if method.startswith('libxsmm-'):
                check(code_bytes>0 and plans==0 and impl=='libxsmm-fp32-jit','LIBXSMM nonreference '+str(key))
            elif method.startswith('onednn-'):
                check(code_bytes==0 and plans==(1 if method=='onednn-cached-plan' else calls+1)
                      and impl.strip() not in ('','direct','libxsmm-fp32-jit'),'oneDNN primitive identity or count '+str(key))
            else:
                check(code_bytes==plans==0 and impl=='direct','direct identity '+str(key))
            if round_id<0:
                check(round_id==-1 and int(r['place'])==-1 and not calls and number(r,'per_call_s')==0,'fixture timing')
            else:
                j=(method_i-position-2*(round_id%3))%len(METHODS)
                check(int(r['place'])==(j if round_id<3 else len(METHODS)-1-j) and r['mode']=='random' and
                      int(r['seed'])==SEEDS[round_id//3] and calls==16 and number(r,'per_call_s')>0,'warm timing order '+str(key))
            h=r['output_hash']; check(re.fullmatch('[a-f0-9]{16}',h) is not None,'output digest')
            test=(r['mode'],r['seed'])
            check(hashes.setdefault(test,h)==h,'repeated deterministic output '+str(key))
        ok=(len(fixed)==4 and len(timed)==6)
        check(ok==(key not in fail),'missing failure/admission reason '+str(key))
        audit.append(dict(family=family,method=method,admitted=int(ok),fixtures=len(fixed),batches=len(timed),
                          reason='' if ok else fail[key]['reason'],max_error=max((float(r['error']) for r in records),default='')))
        if not ok: continue
        observations[method]=dict(warm=[float(r['per_call_s']) for r in timed],first=[float(r['first_s']) for r in timed],
             hashes=hashes,metadata=[dict(implementation=r['implementation'],code_bytes=int(r['code_bytes']),plans=int(r['plans'])) for r in timed])
        admitted[key]=True
    if family in controls and all(method in observations for method in METHODS[:2]):
        equal=observations['candidate']['hashes']==observations['incumbent']['hashes']
        all_parity=all_parity and equal
    fields={}
    for label,roster in [('oneshot',DIRECT),('retained_only',RETAINED),('including_retained',DIRECT+RETAINED)]:
        observed=[m for m in roster if m in observations]
        fields[label]={'missing':[m for m in roster if m not in observed]}
        for period in ('warm','first'):
            rank=sorted(observed,key=lambda m:median(observations[m][period]))
            compared={m:compare(observations[m][period],observations['candidate'][period]) for m in observed} if 'candidate' in observations else {}
            fields[label][period]=dict(ranking=rank,strongest=rank[0] if rank else None,against_candidate=compared)
    baseline={}
    if 'incumbent' in observations and 'candidate' in observations:
        for period in ('warm','first'):
            baseline[period]=compare(observations['incumbent'][period],observations['candidate'][period])
    metrics={m:{p:median(d[p]) for p in ('first','warm')} for m,d in observations.items()}
    for d in observations.values(): del d['hashes']
    layouts[family]=dict(layout=s['layout'],m=m,n=n,k=k,methods=observations,medians_s=metrics,fields=fields,incumbent_over_candidate=baseline)
    summary=dict(family=family,layout=s['layout'],m=m,n=n,k=k,incumbent_over_candidate=baseline.get('warm',{}).get('median',''))
    for field,period in [('oneshot','warm'),('including_retained','warm'),('oneshot','first'),('including_retained','first')]:
        f=fields[field][period]; best=f['strongest']; compared=f['against_candidate'].get(best,{})
        name=field+'_'+period
        summary[name+'_peer']=best or ''
        summary[name+'_over_candidate']=compared.get('median','')
        summary[name+'_candidate_wins']=compared.get('wins','')
    summaries.append(summary)

local_complete=all((f,m) in admitted for f in families for m in METHODS[:2])
complete_all=all((f,m) in admitted for f in families for m in METHODS)
values=[layouts[f]['incumbent_over_candidate']['warm']['paired'] for f in fresh if layouts[f]['incumbent_over_candidate']]
gms=[geometric_mean(median(x[p:p+3]) for x in values) for p in (0,3)] if len(values)==8 else []
medians=[median(x) for x in values]
overall=geometric_mean(medians) if len(values)==8 else None
improved=(local_complete and all_parity and overall is not None and overall>=1.05
          and min(gms)>1 and sum(x>1 for x in medians)>=6 and min(medians)>=0.95)
field_gates={}
for field in ('oneshot','including_retained'):
    below=[]; missing={}
    for f in fresh:
        obj=layouts[f]['fields'][field]
        if obj['missing']: missing[f]=obj['missing']
        for method,d in obj['warm']['against_candidate'].items():
            if d['median']<=1 or d['wins']<4: below.append({'family':f,'peer':method,'ratio':d['median'],'wins':d['wins']})
    field_gates[field]=dict(complete=not missing,missing=missing,counterexamples=below,
                           panel_lead=bool(local_complete and all_parity and not below and not missing))
result=dict(source=identity,cpu=min(cpus),page_size=page,host=host,layouts=layouts,local_complete=local_complete,
            complete_all=complete_all,control_bitwise=all_parity,max_error=max(float(r['error']) for r in raw),
            output_checks=sum(int(r['checked']) for r in raw),complete_method_layouts=len(admitted),
            geometric_mean=overall,seed_geometric_means=gms,fresh_paired_median_wins=sum(x>1 for x in medians),
            worst_fresh_ratio=min(medians,default=None),improvement_gate=improved,fields=field_gates)
(ROOT/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
save_csv('admission.csv',audit,list(audit[0]))
save_csv('summary.csv',summaries,list(summaries[0]))
print(json.dumps({k:result[k] for k in ('local_complete','complete_all','control_bitwise','geometric_mean','improvement_gate')},sort_keys=True))
