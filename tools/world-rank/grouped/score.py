"""Summarize the preregistered whole-call observations on this native CPU."""
import csv
import hashlib
import json
import math
import statistics as stats
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
def rows(name):
    with (OUT/name).open() as source:
        return list(csv.DictReader(source))
def require(condition, why):
    if not condition:
        raise ValueError(why)
def gm(values):
    return math.exp(stats.mean(math.log(v) for v in values))
def pairs(competitor, subject):
    return [a/b for a,b in zip(competitor, subject, strict=True)]
def paired(values):
    return dict(ratios=values, median=stats.median(values), wins=sum(v>1 for v in values))

assert hashlib.sha256((OUT/'panel.csv').read_bytes()).hexdigest() == '5e470c40de6e63b6debbb96afd146eda58de83e50ef3da1e2f19186a8990b4c4'
assert hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest() == 'a36cfd1999ebedf902a649f982ff30e498c6b8d8edb69e09ff7862f233f778a2'
shapes, methods, admissions, raw = [rows(name) for name in ('panel.csv','methods.csv','admissions.csv','raw.csv')]
names = [r['method'] for r in methods]
candidate, incumbent = 'grouped-b','stream-incumbent'
independent = [r['method'] for r in methods if r['role']=='independent']
oneshot = [r['method'] for r in methods if r['role']=='independent' and r['policy']=='one-shot']
require((len(names),len(set(names)),len(independent),len(oneshot))==(11,11,9,7),'registered field')
require(len(admissions)==len(shapes)*len(names),'complete admissions')
decisions = {(r['family'],r['method']):r for r in admissions}
result = {}
for s in shapes:
    family, layout = s['family'],s['layout']
    medians, warm, first = {},{},{}
    for method in names:
        admission = decisions[(family,method)]
        require(admission['layout']==layout,'invalid layout')
        if admission['qualified']!='1':
            continue
        assert (admission['accuracy_records'],admission['timing_records'])==('4','6')
        rr = [r for r in raw if (r['family'],r['layout'],r['method'])==(family,layout,method)]
        measurements = sorted((r for r in rr if int(r['round'])>=0),key=lambda r:int(r['round']))
        require(len(rr)==10 and [int(r['round']) for r in measurements]==list(range(6)),'paired coverage')
        require(all(r['calls']=='32' and float(r['per_call_s'])>0 for r in measurements),'timing')
        warm[method] = [float(r['per_call_s']) for r in measurements]
        first[method] = [float(r['first_s']) for r in rr]
        medians[method] = stats.median(warm[method])
    tiers = {}
    for label,field in [('one-shot',oneshot),('including-retained',independent),('whole-field',names)]:
        active = [method for method in field if method in warm]
        if active:
            best = min(active,key=medians.get)
            tiers[label] = dict(method=best,complete=len(active)==len(field),
                                paired=paired(pairs(warm[best],warm[candidate])) if candidate in warm else None)
    result[family] = dict(layout=layout,median_s=medians,first_s=first,tiers=tiers,
            versus={method:paired(pairs(values,warm[candidate])) for method,values in warm.items()} if candidate in warm else {})
fresh = [r['family'] for r in shapes if r['family'].startswith('fresh-')]
assert len(fresh)==8
route_rows = rows('grouped-routes.csv')
route_keys = {(r['implementation'],r['family'],r['layout']) for r in route_rows if r['passed']=='1'}
require(len(route_rows)==2*len(shapes) and all((m,s['family'],s['layout']) in route_keys for m in (incumbent,candidate) for s in shapes), 'normal and instrumented route witness')
control_parity = rows('grouped-control-parity.csv')
require(len(control_parity)==8 and all(r['passed']=='1' and len(r['candidate_sha256'])==64 and r['incumbent_sha256']==r['candidate_sha256'] for r in control_parity), 'four-fixture bypass parity')
complete = all(incumbent in result[r['family']]['versus'] for r in shapes)
primary = dict(complete=complete, passed=False)
if complete:
    matched = [result[f]['versus'][incumbent] for f in fresh]
    medians = [r['median'] for r in matched]
    blocks = [gm([stats.median(r['ratios'][start:start+3]) for r in matched]) for start in (0,3)]
    loo = [gm(medians[:i]+medians[i+1:]) for i in range(len(medians))]
    mean, wins, worst = gm(medians),sum(v>1 for v in medians),min(medians)
    primary.update(geomean=mean,layout_wins=wins,paired_wins=sum(r['wins'] for r in matched),
                   min_layout=worst,block_geomeans=blocks,leave_one_out=[min(loo),max(loo)],
                   passed=mean>=1.03 and wins>=6 and worst>=0.97 and min(blocks)>1)
external = {}
for tier in ('one-shot','including-retained'):
    rows_here = [result[f]['tiers'].get(tier) for f in fresh]
    if not all(r and r['paired'] for r in rows_here):
        external[tier]=dict(complete=False,stable_lead=False)
        continue
    p = [r['paired'] for r in rows_here]
    full = all(r['complete'] for r in rows_here)
    external[tier] = dict(complete=full,geomean=gm([r['median'] for r in p]),layout_wins=sum(r['median']>1 for r in p),
                          paired_wins=sum(r['wins'] for r in p),stable_lead=full and all(r['median']>1.02 and r['wins']>=5 for r in p))
report=dict(primary=primary,external=external,layouts=result)
(OUT/'grouped-score.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(primary=primary,external=external),indent=2))
require(complete,'installed standalone comparison incomplete')
