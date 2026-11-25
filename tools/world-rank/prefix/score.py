"""Compute the registered per-host prefix-vs-provider and independent decisions."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

OUT=Path(sys.argv[1])
def read(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))
def require(condition,why):
    if not condition: raise ValueError(why)
def pair(values,subject):
    ratios=[a/b for a,b in zip(values,subject,strict=True)]
    return dict(ratios=ratios,median=st.median(ratios),wins=sum(r>1 for r in ratios))
def gm(values):
    return math.exp(st.mean(math.log(x) for x in values))

for name,expected in {"panel.csv":"2b4d6871262a4d038533b6cab819e819c3cff1eaf3ef606678b2ad6e30520ed5",
                      "protocol.md":"02581f0e088da5d11c0308b3583b9ef5f87e239999268ff6430952596db57c60"}.items():
    require(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==expected,"registration "+name)
panel,methods,admissions,rows=map(read,("panel.csv","methods.csv","admissions.csv","raw.csv"))
subject="host-prefix8"
names=[m["method"] for m in methods]
require(len(names)==len(set(names))==13 and len(panel)==11,"registered field/geometry")
require(len(admissions)==len(panel)*len(names),"missing admissions")
independent=[m['method'] for m in methods if m['role']=='independent']
one=[m['method'] for m in methods if m['role']=='independent' and m['policy']=='one-shot']
require(len(independent)==9 and len(one)==7 and sum(m['role']=='vendor-assisted' for m in methods)==3,"roles")
by_family={}
for shape in panel:
    f,layout=shape['family'],shape['layout']
    times,first,medians={}, {}, {}
    for method in names:
        adm=[r for r in admissions if (r['family'],r['layout'],r['method'])==(f,layout,method)]
        require(len(adm)==1,"duplicate admission")
        if adm[0]['qualified']!='1':
            continue
        raw=[r for r in rows if (r['family'],r['layout'],r['method'])==(f,layout,method)]
        warm=sorted((r for r in raw if int(r['round'])>=0),key=lambda r:int(r['round']))
        require(len(raw)==10 and [int(r['round']) for r in warm]==list(range(6)),"matched rounds")
        require(all(r['calls']=='32' and r['threads']=='1' and int(r['checked'])==int(shape['m'])*int(shape['n'])
                    and float(r['error'])<=0.002 and r['mode']=='random' for r in warm),"qualified sample")
        times[method]=[float(r['per_call_s']) for r in warm]
        first[method]=[float(r['first_s']) for r in warm]
        medians[method]=st.median(times[method])
    tiers={}
    for k,cohort in (('one-shot',one),('including-retained',independent),('all',names)):
        eligible=[m for m in cohort if m in times]
        if eligible:
            best=min(eligible,key=medians.get)
            tiers[k]=dict(method=best,complete=len(eligible)==len(cohort),
                          paired=pair(times[best],times[subject]) if subject in times else None)
    by_family[f]=dict(layout=layout,median_s=medians,tiers=tiers,
        paired={m:pair(vals,times[subject]) for m,vals in times.items()} if subject in times else {},
        first_paired={m:pair(vals,first[subject]) for m,vals in first.items()} if subject in first else {})
fresh=[s['family'] for s in panel if s['family'].startswith('fresh-')]
active=[s['family'] for s in panel if not s['family'].startswith('control-')]
require((len(fresh),len(active))==(6,9),'panel roles')
required=('mkl','host-forward','host-split8','host-prefix8')
complete=all(all(m in by_family[s['family']]['paired'] for m in required) for s in panel)
host=(OUT/'host.txt').read_text()
intel='GenuineIntel' in host
routes=read('prefix-routes.csv')
rkeys={(r['method'],r['family']):r for r in routes}
require(len(routes)==len(rkeys)==33 and all(r['passed']=='1' for r in routes),'native wrapper witness')
jit_ready=all(r['intel_avx2']=='1' for r in routes)
require(all(r['cpu_vendor']==('GenuineIntel' if intel else 'AuthenticAMD') for r in routes),'native processor identity')
for s in panel:
    family=s['family']
    eligible=intel and jit_ready and not family.startswith('control-')
    for method in ('host-forward','host-split8','host-prefix8'):
        r=rkeys[method,family]
        expected=(1, 1) if eligible and method=='host-prefix8' else ((0,2) if eligible and method=='host-split8' else (0,1))
        require((int(r['vector_calls']),int(r['provider_calls']))==expected and int(r['eligible'])==int(eligible),'call-boundary proof')
parity=read('prefix-bypass-parity.csv')
require(len(parity)==(2 if intel else 11)*4*4 and all(r['passed']=='1' and r['sha256']==r['mkl_sha256'] for r in parity),'bypass parity')
primary=dict(eligible_cpu=intel and jit_ready,complete=complete,passed=False)
if complete:
    groups={}
    for m in required[:-1]:
        fresh_rows=[by_family[f]['paired'][m] for f in fresh]
        groups[m]=dict(geomean=gm([r['median'] for r in fresh_rows]),
            min_eligible=min(by_family[f]['paired'][m]['median'] for f in active),
            min_fresh=min(r['median'] for r in fresh_rows),layout_wins=sum(r['median']>1 for r in fresh_rows),
            paired_wins=sum(r['wins'] for r in fresh_rows),
            block_geomeans=[gm([st.median(r['ratios'][j:j+3]) for r in fresh_rows]) for j in (0,3)],
            first_geomean=gm([by_family[f]['first_paired'][m]['median'] for f in fresh]))
    joint=sum(all(by_family[f]['paired'][m]['median']>=1.03 for m in ('mkl','host-forward')) for f in fresh)
    primary.update(comparisons=groups,qualifying_fresh=joint,
        passed=intel and jit_ready and joint>=5 and all(groups[m]['geomean']>=1.03 and groups[m]['min_eligible']>=0.97 for m in ('mkl','host-forward')))
report=dict(primary=primary,layouts=by_family)
(OUT/'prefix-score.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(primary,indent=2))
require(complete,'registered candidates or provider controls failed qualification')
