"""Check the frozen case matrix and summarize numerical and branch outcomes."""
import argparse
import csv
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checks',type=Path)
p.add_argument('fallbacks',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args()
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
r,f=read(a.checks),read(a.fallbacks)
if len(r)!=98 or len(f)!=15:raise ValueError(f'incomplete {len(r)}, {len(f)}')
assert len({(x['layout'],x['fixture'],x['boundary']) for x in r})==len(r)
assert len({x['case'] for x in f})==len(f)
for x in r:
    assert x['pass']==x['canaries_ok']=='1' and x['nonidentical_to_trace']==x['over_forward_bound']=='0'
    assert x['eligible']==x['trace_delta']
    m,n,k,la,lb,lc=(int(x[q]) for q in ('m','n','k','lda','ldb','ldc'))
    selected=(24<=m<=32 and 1024<=n<=4096 and 96<=k<=384 and la==k and lb==lc==n)
    assert int(x['trace_delta'])==int(selected)
    if m in (33,39):assert not selected and int(x['nonidentical_to_base'])==0
    assert int(x['cells'])==int(x['m'])*int(x['n'])
    if x['eligible']=='0':assert x['nonidentical_to_base']=='0'
for x in f:assert x['passed']=='1'
assert sum(int(x['faults']) for x in f)==sum(int(x['selected']) for x in f)==1
assert {x['case'] for x in f if x['faults']=='1'}=={'fail-next-workspace'}
assert {x['case'] for x in f if x['selected']=='1'}=={'normal-after-failure'}
out={'cases':len(r),'cells':sum(int(x['cells']) for x in r),'selected':sum(int(x['eligible']) for x in r),
     'bypassed':sum(x['eligible']=='0' for x in r),'fallback_cases':len(f),'fixtures':{}}
for fixture in sorted({x['fixture'] for x in r}):
    block=[x for x in r if x['fixture']==fixture]
    out['fixtures'][fixture]=dict(cases=len(block),cells=sum(int(x['cells']) for x in block))
    for k in ('max_abs_candidate','max_abs_base','max_abs_nearest'):
        out['fixtures'][fixture][k]=max(float(x[k]) for x in block)
    for k in ('over_base','over_candidate','over_nearest','new_over_candidate','removed_over_candidate','nonidentical_to_base'):
        out['fixtures'][fixture][k]=sum(int(x[k]) for x in block)
a.out.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='fixtures'}))
