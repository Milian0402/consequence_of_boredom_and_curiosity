"""Verify all raw admission and shape data before recomputing matched medians."""
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

def require(ok,reason):
    if not ok: raise ValueError(reason)

def rows(path):
    with path.open() as file: return list(csv.DictReader(file))

out = Path(sys.argv[1])
panel = out/'panel.csv'
require(hashlib.sha256(panel.read_bytes()).hexdigest()=='ab9115f17a698938d14e3fb0010f05e450b214eca8876d7fd6d59e2323b17c80','panel fingerprint')
methods = ('default','stream','packed','openblas','mkl','mkl-jit','aocl')
peers = methods[3:]
shapes = rows(panel)
bins = rows(out/'binaries.csv')
require([r['implementation'] for r in bins]==list(methods),'binary roster')
require(all(len(r['worker'])==64 for r in bins) and all(len(r['archive'])==64 for r in bins[:3]),'binary hashes')
require(len({r['archive'] for r in bins[:3]})==3,'distinct linked archives')
identity=json.loads((out/'frontier-sources.json').read_text())
require(identity['release']=='a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6' and identity['patch_sha256']=='b994dd80639f521ae59a0ca70dcf58b43db6773b4ecf92033f841515624877fa','production identity')
raw = rows(out/'raw.csv')
routes = rows(out/'routes.csv')
has2,has512=map(int,(out/'isa.txt').read_text().split())
require(has2==1 and has512 in (0,1),'x86 features')
isa='avx512' if has512 else 'avx2'
seeds=(73194513,38641729)
page=int((out/'page-size.txt').read_text())
require(page>=4096 and not page&(page-1),'page size')
require(len(raw)==len(shapes)*len(methods)*10 and len(routes)==len(shapes)*2,'complete raw tables')
require(len({r['cpu'] for r in raw})==1,'single pinned CPU across run')
report={}
for s in shapes:
    family,layout=s['family'],s['layout']
    m,n,k=map(int,[s['m'],s['n'],s['k']])
    def matches(r): return (r['family'],r['layout'])==(family,layout)
    views=[(int(s[f'{v}r']),int(s[f'{v}c']),int(s[ld])) for v,ld in zip('abc',['lda','ldb','ldc'])]
    residues=[(r*ld+c)*4%64 for r,c,ld in views]
    require((layout=='aligned' and residues==[0,0,0]) or (layout=='cropped' and len(set(residues))==3 and 0 not in residues),'frozen base alignment')
    cr,cc,ldc=views[-1]
    guards=((cr+m+2)*ldc*4+page-1)//page*(page//4)-m*n
    candidates=[r for r in routes if matches(r)]
    require(sorted(r['method'] for r in candidates)==['packed','stream'],'route roster')
    for r in candidates:
        prefix='packed_' if r['method']=='packed' and m<=96 else ''
        for col in ('avx2','avx512','packed_avx2','packed_avx512'):
            require(int(r[col])==(col==prefix+isa),'real kernel route '+str((family,layout,r)))
        require((r['public_calls'],r['args_ok'],r['alignment_ok'],r['refused'])==('1','1','1','0'),'public route forwarding')
    by_method={}
    for idx,method in enumerate(methods):
        rr=[r for r in raw if matches(r) and r['method']==method]
        require(len(rr)==10,'sample count')
        valid=[r for r in rr if int(r['round'])<0]
        timings=sorted((r for r in rr if int(r['round'])>=0),key=lambda r:int(r['round']))
        require(sorted((r['mode'],int(r['seed'])) for r in valid)==sorted([('random',seeds[0]),('random',seeds[1]),('cancel',seeds[0]),('scaled',seeds[1])]),'four full FP64 fixtures')
        require(all(int(r['round'])==-1 and int(r['calls'])==0 and int(r['place'])==-1 for r in valid),'qualification records')
        require([int(r['round']) for r in timings]==list(range(6)),'paired gate')
        for r in rr:
            require(int(r['checked'])==m*n and int(r['guards'])==guards and int(r['threads'])==1,'whole parent verification')
            require([int(r[key]) for key in ('a_mod64','b_mod64','c_mod64')]==residues,'actual base residues')
            require(0<=float(r['error'])<=0.002 and math.isfinite(float(r['first_s'])) and float(r['first_s'])>0,'numerical/first timing gate')
        for j,r in enumerate(timings):
            rotating=(idx-shapes.index(s)-2*(j%3))%len(methods)
            require(int(r['place'])==(rotating if j<3 else len(methods)-1-rotating) and int(r['calls'])==16 and int(r['seed'])==seeds[j//3] and r['mode']=='random','balanced whole-call roster')
            require(0<float(r['per_call_s'])<math.inf,'positive timing')
        by_method[method]=[float(r['per_call_s']) for r in timings]
    med={method:statistics.median(values) for method,values in by_method.items()}
    best=min(peers,key=med.get)
    def pairs(method,challenger):
        return [a/b for a,b in zip(by_method[method],by_method[challenger])]
    ratios={method:pairs(method,'stream') for method in methods if method!='stream'}
    check=ratios[best]
    report.setdefault(family,{})[layout]=dict(medians_s=med,warm_s=by_method,best_independent=best,peer_over_stream=ratios,
        peer_over_packed={method:pairs(method,'packed') for method in peers},stream_wins_best=sum(x>1 for x in check),
        stream_transfer_leg=med['stream']<med[best] and sum(x>1 for x in check)>=4)
for family,d in report.items():
    d['crop_over_aligned']={method:[a/b for a,b in zip(d['cropped']['warm_s'][method],d['aligned']['warm_s'][method])] for method in methods}
    if family.startswith('repeat-win'):
        d['earlier_direct_lead_transfers']=d['aligned']['stream_transfer_leg'] and d['cropped']['stream_transfer_leg']
result=dict(host=(out/'host.txt').read_text(),max_abs_error=max(float(r['error']) for r in raw),layouts=report)
(out/'view-score.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'summary.csv').open('w') as f:
    w=csv.writer(f); w.writerow(['family','layout','fastest_peer','stream_microseconds','peer_over_stream_paired_median','stream_pairs_won','default_over_stream','packed_over_stream','earlier_direct_lead_transfers'])
    for family,d in report.items():
        for layout in ('aligned','cropped'):
            l=d[layout]; rat=l['peer_over_stream']; best=l['best_independent']
            w.writerow([family,layout,best,1e6*l['medians_s']['stream'],statistics.median(rat[best]),l['stream_wins_best'],statistics.median(rat['default']),statistics.median(rat['packed']),d.get('earlier_direct_lead_transfers','')])
