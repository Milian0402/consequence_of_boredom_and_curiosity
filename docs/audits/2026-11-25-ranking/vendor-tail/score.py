"""Validate and compare fully qualified, matched one-thread study observations."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import geometric_mean, median
import sys
from routes import COUNTS, DIMS, NAMES, expected

PRIMARY = ('fresh-thin-tail','fresh-wide-tail','fresh-deeper','fresh-compact-tile','fresh-near-footprint')
TREATMENTS = tuple('fresh-tail-'+str(i) for i in range(1,5))
CONTROL = ('repeat-skinny-wide','repeat-wide','outside-width')
EXTERNAL = {'openblas','mkl','mkl-jit','aocl','blasfeo','onednn-oneshot','onednn-cached-plan','matched-openblas','matched-mkl'}
STANDALONE = {'stream-native','stream-incumbent'}
REQUIRED = EXTERNAL | STANDALONE | set(NAMES)
PAIRS = [(b,r) for b in (1,2) for r in (1,2,3)]


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def validate(ok, msg):
    if not ok:raise ValueError(msg)


def score(root, panel):
    issues=[]
    panel_rows=read(panel)
    cases={r['family']:tuple(int(r[d]) for d in DIMS) for r in panel_rows}
    validate(len(cases)==len(panel_rows)==12 and set(cases)==set(PRIMARY+TREATMENTS+CONTROL),'panel families changed')
    panel_hash=hashlib.sha256(panel.read_bytes()).hexdigest()
    validate(panel_hash=='c05f30a91720c54086c0b6d5e858267501af8488e4af9614555e2f5e12c55bd6','frozen panel sha256 differs')
    ranking=json.loads((root/'ranking.json').read_text())
    validate(ranking['panel_sha256']==panel_hash and ranking['rounds_per_seed']==3 and float(ranking['full_output_tolerance'])==.002,'study metadata')
    seeds=ranking['seeds'];validate(ranking['phase']=='stream-b' and seeds==[348271759,710438957],'invalid frozen phase or seeds')
    phase=ranking['phase'];methods=list(ranking['implementations'])
    manifest=read(root/'methods.csv')
    validate(len({r['implementation'] for r in manifest})==len(manifest),'duplicate inventory name')
    manifest_names={r['implementation'] for r in manifest}
    if missing:=REQUIRED-set(methods):issues.append('missing study methods: '+','.join(sorted(missing)))
    if set(methods)!=manifest_names:issues.append('inventory differs from measured methods')
    for r in manifest:
        if not r['role'] or not r['abi'] or not re.fullmatch('[a-f0-9]{64}',r['artifact_sha256']):
            issues.append('incomplete build metadata '+r['implementation'])
    if ranking.get('errors'):issues.append(f"study recorded {len(ranking['errors'])} errors")
    cpu=set();routes={};checks=defaultdict(dict);times=defaultdict(dict);calls=defaultdict(dict)
    def key(row):
        f,n=row['family'],row['implementation']
        validate(f in cases and tuple(int(row[d]) for d in DIMS)==cases[f],'shape mismatch '+str((f,n)))
        return f,n
    for row in read(root/'routes.csv'):
        try:
            f,n=k=key(row);validate(n in NAMES and k not in routes,'unexpected/duplicate dispatch row')
            routes[k]=False
            validate(tuple(int(row[k]) for k in COUNTS)==tuple(expected(n,*cases[f][:3])) and row['passed']=='1','dispatch counts')
            validate(int(row['timed_checked'])==int(row['trace_checked'])==cases[f][0]*cases[f][1] and
                     row['guards']==row['inputs_unchanged']==row['threads']=='1','public fixture')
            validate(row['cpu'].isdigit(),'route CPU');cpu.add(row['cpu']);routes[k]=True
        except (KeyError,ValueError) as err:issues.append(f"route {row.get('family')}/{row.get('implementation')}: {err}")
    fixtures={('random',1,seeds[0]),('random',2,seeds[1]),('scaled',0,seeds[0]),('cancel',0,seeds[0])}
    for row in read(root/'accuracy.csv'):
        try:
            f,n=k=key(row);token=row['mode'],int(row['block']),int(row['seed'])
            validate(n in methods and token in fixtures and token not in checks[k] and row['phase']==phase,'fixture identity')
            checks[k][token]=False
            detail=dict(i.split('=',1) for i in row['detail'].split(',') if '=' in i)
            err=float(row['max_abs_error']);first=float(detail['first_sgemm_seconds'])
            validate(math.isfinite(err) and 0<=err<=.002 and row['verified']=='1' and row['nonfinite']==row['guard_changes']==detail['input_changes']=='0','FP64/input/guard check')
            validate(int(row['checked'])==cases[f][0]*cases[f][1] and row['stress']==str(int(token[1]==0)) and first>0 and math.isfinite(first),'output coverage/stress/first call')
            validate(row['threads']==detail['threads']=='1' and row['cpu'].isdigit() and detail['cpu']==row['cpu'],'worker/thread/CPU')
            if n.startswith('onednn-'):validate(detail.get('onednn_impl'),'oneDNN kernel identity')
            checks[k][token]=True;cpu.add(row['cpu'])
        except (KeyError,ValueError) as err:issues.append(f"accuracy {row.get('family')}/{row.get('implementation')}: {err}")
    family_index={f:i for i,f in enumerate(cases)}
    for row in read(root/'timings.csv'):
        try:
            f,n=k=key(row);b,r=pair=int(row['block']),int(row['round'])
            validate(n in methods and pair in PAIRS and pair not in times[k] and row['phase']==phase,'timing identity')
            t=float(row['seconds']);batch=int(row['calls'])
            validate(math.isfinite(t) and t>0 and batch>0 and int(row['seed'])==seeds[b-1] and row['stress']=='0' and row['mode']=='random','time/seed')
            validate(row['threads']=='1' and row['cpu'].isdigit(),'timing CPU or thread count')
            shift=(family_index[f]+r-1)%len(methods)
            order=methods[shift:]+methods[:shift]
            if (r+b-2)%2:order.reverse()
            validate(int(row['position'])==order.index(n)+1,'counterbalanced timing position')
            times[k][pair]=t;calls[k][pair]=batch;cpu.add(row['cpu'])
        except (KeyError,ValueError) as err:issues.append(f"timing {row.get('family')}/{row.get('implementation')}: {err}")
    if len(cpu)!=1:issues.append('study and fixture processes do not share one pinned CPU')
    qualified=set()
    for f in cases:
        for n in sorted(REQUIRED|set(methods)):
            k=f,n
            if (set(checks[k])==fixtures and all(checks[k].values()) and set(times[k])==set(PAIRS)
                and (n not in NAMES or routes.get(k,False))):qualified.add(k)
            else:issues.append('incomplete or failed qualification '+f+'/'+n)
    result=[];by_candidate=defaultdict(dict)
    for f in cases:
        active={n for n in methods if (f,n) in qualified}
        elapsed={n:median(times[f,n].values()) for n in active}
        external=EXTERNAL & active
        if not external:continue
        fastest=lambda members:min(members,key=lambda n:(elapsed[n],n)) if members else None
        best=fastest(external);best_oneshot=fastest(external-{'onednn-cached-plan'});best_standalone=fastest(STANDALONE&active)
        for c in NAMES:
            if c not in active:continue
            matched='matched-'+c.removeprefix('vendor-wide-').rsplit('-',1)[0]
            for n in sorted(active-{c}):
                if calls[f,n]!=calls[f,c]:
                    issues.append('unmatched calls '+f+'/'+c+'/'+n);continue
                z=[times[f,n][p]/times[f,c][p] for p in PAIRS]
                row=dict(family=f,**dict(zip(DIMS,cases[f])),candidate=c,peer=n,
                         candidate_ms=elapsed[c]*1000,peer_ms=elapsed[n]*1000,paired_speedup=median(z),
                         paired_wins=sum(v>1 for v in z),seed1=median(z[:3]),seed2=median(z[3:]),
                         matched_direct=int(n==matched),fastest_independent_any=int(n==best),
                         fastest_independent_oneshot=int(n==best_oneshot),fastest_standalone=int(n==best_standalone),
                         ratios=json.dumps(z,separators=(',',':')))
                result.append(row)
                for comparison,is_peer in [('matched_direct',n==matched),('fastest_independent_any',n==best),
                                            ('fastest_independent_oneshot',n==best_oneshot),('fastest_standalone',n==best_standalone),
                                            ('stream-incumbent',n=='stream-incumbent'),('stream-native',n=='stream-native'),
                                            ('same_provider_full',c.endswith('-split') and n==c[:-5]+'full')]:
                    if is_peer:by_candidate[c][f,comparison]=row
    fields=['family',*DIMS,'candidate','peer','candidate_ms','peer_ms','paired_speedup','paired_wins','seed1','seed2','matched_direct',
            'fastest_independent_any','fastest_independent_oneshot','fastest_standalone','ratios']
    with (root/'vendor-wide-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result)
    metrics={}
    for candidate,rows in by_candidate.items():
        groups={}
        for group,shapes in [('original_five',PRIMARY),('four_tail_treatments',TREATMENTS)]:
            tallies={}
            for comp in sorted({tag for _,tag in rows}):
                cohort=[rows[f,comp] for f in shapes if (f,comp) in rows]
                stats={'shapes':len(cohort),'expected':len(shapes),'complete':len(shapes)==len(cohort)}
                if stats['complete']:
                    stats.update(geometric_mean=geometric_mean(x['paired_speedup'] for x in cohort),
                                 seed_geometric_means=[geometric_mean(x[s] for x in cohort) for s in ('seed1','seed2')],
                                 shape_wins=sum(x['paired_speedup']>1 for x in cohort),minimum=min(x['paired_speedup'] for x in cohort))
                tallies[comp]=stats
            groups[group]=tallies
        if candidate.endswith('-split'):
            eligible_tail_wins=[]
            for f in cases:
                if (f,'same_provider_full') not in rows or (f,'matched_direct') not in rows:continue
                if expected(candidate,*cases[f][:3])[2] and rows[f,'same_provider_full']['paired_speedup']>1 and rows[f,'matched_direct']['paired_speedup']>1:
                    eligible_tail_wins.append(f)
            groups['shapes_beating_both_matched_full_and_direct_with_tail_selected']=eligible_tail_wins
        metrics[candidate]=groups
    return dict(complete=not issues,phase=phase,panel_sha256=panel_hash,seeds=seeds,cpus=sorted(cpu),
                issues=sorted(set(issues)),candidates=metrics)


if __name__=='__main__':
    root,panel=map(Path,sys.argv[1:])
    try:
        report=score(root,panel)
    except (OSError,ValueError,KeyError,IndexError,ZeroDivisionError) as exc:
        report=dict(complete=False,issues=[str(exc)])
    (root/'vendor-wide-score.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['complete']:raise SystemExit('incomplete vendor-wide study; see vendor-wide-score.json')
