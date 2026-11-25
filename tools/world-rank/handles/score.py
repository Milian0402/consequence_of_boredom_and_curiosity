"""Compare public COB with frozen original peers and explicit oneMKL handle modes."""
import csv, hashlib, json, math, statistics as st, sys
from pathlib import Path

out=Path(sys.argv[1])
def read(name):
    with (out/name).open() as f:
        return list(csv.DictReader(f))
def require(ok,why):
    if not ok: raise ValueError(why)
def compare(peer,subject):
    ratios=[a/b for a,b in zip(peer,subject,strict=True)]
    return dict(paired=ratios,median=st.median(ratios),wins=sum(x>1 for x in ratios))
def gm(items):
    return math.exp(st.mean(math.log(x) for x in items))

panel,raw,admissions,methods,receipts=map(read,('panel.csv','raw.csv','admissions.csv','methods.csv','handle-observations.csv'))
require(hashlib.sha256((out/'panel.csv').read_bytes()).hexdigest()=='2b4d6871262a4d038533b6cab819e819c3cff1eaf3ef606678b2ad6e30520ed5','frozen panel')
require(hashlib.sha256((out/'protocol.md').read_bytes()).hexdigest()=='b63ad49548f9c8fd0f429427de3c860309d991bc3a791a695c15d5f6623e673e','frozen protocol')
require(len(panel)==11 and len(methods)==12 and len(admissions)==132,'registered roster')
subject='stream-incumbent'
oneshot='mkl-handle-oneshot'
prepared='mkl-handle-prepared'
ind=[r['method'] for r in methods if r['role']=='independent']
usual=[m for m in ind if m not in (oneshot,prepared)]
warmone=[r['method'] for r in methods if r['role']=='independent' and r['policy']=='one-shot']
rosters={'original-one-shot':[m for m in warmone if m!=oneshot],'original-including-retained':usual,
         'expanded-one-shot':warmone,'expanded-including-retained':ind}
require([len(v) for v in rosters.values()]==[7,9,8,11],'call policy')
report={}
for s in panel:
    f,layout=s['family'],s['layout']
    active=[p['method'] for p in admissions if p['family']==f and p['qualified']=='1']
    samples={}; status={}; prep={}
    for m in active:
        rr=[r for r in raw if (r['family'],r['layout'],r['method'])==(f,layout,m)]
        warm=sorted((r for r in rr if int(r['round'])>=0),key=lambda r:int(r['round']))
        require(len(rr)==10 and [int(r['round']) for r in warm]==list(range(6)),'matched samples')
        samples[m]={'warm':[float(r['per_call_s']) for r in warm], 'first':[float(r['first_s']) for r in warm]}
        samples[m]['first-with-preparation']=samples[m]['first']
        if m in (oneshot,prepared):
            handles=[r for r in receipts if (r['family'],r['layout'],r['method'])==(f,layout,m)]
            require(len(handles)==10 and len({(r['first_status'],r['last_status']) for r in handles})==1,'uniform handles')
            stage=handles[0]['first_status']
            require(stage in ('0','1') and all(r['last_status']==stage for r in handles),'native MKL status')
            status[m]='MKL_JIT_SUCCESS' if stage=='0' else 'MKL_NO_JIT'
            hr=sorted((r for r in handles if int(r['round'])>=0),key=lambda r:int(r['round']))
            prep[m]=[float(r['prepare_s']) for r in hr]
            require(all((r['prepared_status']==stage and float(r['prepare_s'])>0) if m==prepared else
                        (r['prepared_status']=='-1' and float(r['prepare_s'])==0) for r in handles),'prepared lifecycle')
            samples[m]['first-with-preparation']=[a+b for a,b in zip(samples[m]['first'],prep[m],strict=True)]
    if len(status)==2: require(len(set(status.values()))==1, 'explicit policies selected different MKL dispatch')
    tiers={}
    for name,roster in rosters.items():
        eligible=[m for m in roster if m in samples]
        if not eligible or subject not in samples: continue
        tiers[name]={}
        for metric in ('warm','first','first-with-preparation'):
            winner=min(eligible,key=lambda m:st.median(samples[m][metric]))
            tiers[name][metric]=dict(leader=winner,complete=len(eligible)==len(roster),peer_over_cob=compare(samples[winner][metric],samples[subject][metric]))
    handles_vs_direct={}
    if 'mkl' in samples:
        for m in status:
            handles_vs_direct[m]={metric:compare(samples['mkl'][metric],samples[m][metric]) for metric in samples[m]}
    handles_vs_macro={m:{metric:compare(samples['mkl-jit'][metric],samples[m][metric]) for metric in samples[m]}
                      for m in status if 'mkl-jit' in samples}
    report[f]=dict(layout=layout,status=status,prep_s=prep,tiers=tiers,handles_vs_direct=handles_vs_direct,handles_vs_macro=handles_vs_macro,
                   medians_s={m:{k:st.median(v) for k,v in items.items()} for m,items in samples.items()})
fresh=[f for f in report if f.startswith('fresh-')]
require(len(fresh)==6,'fresh panel')
summary={}
for name in rosters:
    for metric in ('warm','first','first-with-preparation'):
        rs=[report[f]['tiers'][name][metric] for f in fresh if name in report[f]['tiers']]
        if not rs: continue
        summary[name+' '+metric]=dict(complete=len(rs)==6 and all(r['complete'] for r in rs),geomean=gm([r['peer_over_cob']['median'] for r in rs]),
                                      layout_wins=sum(r['peer_over_cob']['median']>1 for r in rs),paired_wins=sum(r['peer_over_cob']['wins'] for r in rs))
by_status={}
for method in (oneshot,prepared):
    for label in ("MKL_JIT_SUCCESS","MKL_NO_JIT"):
        cases=[f for f in fresh if report[f]["status"].get(method)==label and method in report[f]["handles_vs_direct"]]
        if cases:
            by_status[method+" "+label]={}
            for metric in ("warm","first","first-with-preparation"):
                paired=[report[f]["handles_vs_direct"][method][metric] for f in cases]
                by_status[method+" "+label][metric]=dict(cases=cases,geomean=gm([p["median"] for p in paired]),
                    layout_wins=sum(p["median"]>1 for p in paired),paired_wins=sum(p["wins"] for p in paired))
(out/'handle-score.json').write_text(json.dumps(dict(summary=summary,by_status_vs_direct=by_status,layouts=report),indent=2)+'\n')
print(json.dumps(summary,indent=2))
coverage=all(set(r['status'])=={oneshot,prepared} and subject in r['medians_s'] and 'mkl' in r['medians_s'] for r in report.values())
(out/'handle-coverage.txt').write_text(('complete' if coverage else 'incomplete: inspect per-layout admissions and witness exclusions')+'\n')
