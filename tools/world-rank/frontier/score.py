"""Score the registered standalone and external comparisons on this native worker."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median

PAIRS = [(b,r) for b in (1,2) for r in (1,2,3)]
SEEDS = (856103219,1743857903)
FIXTURES = {("random",1,SEEDS[0]),("random",2,SEEDS[1]),("cancel",0,SEEDS[0]),("scaled",0,SEEDS[0])}
CANDIDATE = "stream-packed-a"
BASELINE = "frontier-incumbent"

def need(test,why):
    if not test:
        raise ValueError(why)

def read(root,name):
    with (root/name).open(newline="") as f:
        return list(csv.DictReader(f))

def score(root):
    run=json.loads((root/"run.json").read_bytes())
    for n,d in run["source_files"].items():
        need(hashlib.sha256((root/n).read_bytes()).hexdigest()==d,"Modified "+n)
    ranking=json.loads((root/"ranking.json").read_bytes())
    methods={x["implementation"]:x for x in read(root,"methods.csv")}
    panel={x["family"]:(int(x["m"]),int(x["n"]),int(x["k"]),int(x["pad"])) for x in read(root,"panel.csv")}
    fresh=[f for f in panel if f.startswith("fresh-")]
    need(len(panel)==12 and len(fresh)==8 and list(ranking["implementations"])==list(methods) and
         ranking["seeds"]==list(SEEDS) and ranking["phase"]=="frontier-x86" and ranking["rounds_per_seed"]==3,"Cohort or panel")
    xsmm={(r["family"],r["implementation"]):r for r in read(root,"libxsmm-routes.csv")}
    rows=read(root,"accuracy.csv")
    proofs=defaultdict(dict);times=defaultdict(dict);admitted=defaultdict(set);excluded={};maxerr=defaultdict(float)
    for r in rows:
        f,n=r["family"],r["implementation"];m,w,k,p=panel[f]
        q=r["mode"],int(r["block"]),int(r["seed"])
        d=dict(x.split("=",1) for x in r["detail"].split(",") if "=" in x)
        er=float(r["max_abs_error"])
        need(n in methods and q in FIXTURES and q not in proofs[f,n] and int(r["checked"])==m*w and
             r["threads"]==d["threads"]=="1" and int(r["cpu"])==run["cpu"] and r["nonfinite"]==r["guard_changes"]==d["input_changes"]=="0" and
             er>=0 and math.isfinite(er) and (r["verified"]=="1")==(er<=0.002),"Numerical proof: "+str((f,n,q)))
        if n.startswith("libxsmm-"):
            need(int(d["xsmm_jit_bytes"])==int(xsmm[f,n]["code_bytes"])>0,"LIBXSMM identity")
        if n.startswith("onednn-"):
            need(bool(d["onednn_impl"]),"oneDNN runtime")
        proofs[f,n][q]=er
        maxerr[n]=max(maxerr[n],er)
    for r in read(root,"timings.csv"):
        f,n=r["family"],r["implementation"];key=int(r["block"]),int(r["round"])
        t=float(r["seconds"])
        need(n in methods and key in PAIRS and key not in times[f,n] and r["threads"]=="1" and int(r["cpu"])==run["cpu"] and
             int(r["seed"])==SEEDS[key[0]-1] and int(r["calls"])==32 and math.isfinite(t) and t>0,"Warm timing")
        times[f,n][key]=t
    cases=read(root,"case-eligibility.csv")
    need(len(cases)==len(panel)*len(methods) and len({(x["family"],x["implementation"]) for x in cases})==len(cases),"Case coverage")
    for r in cases:
        key=f,n=r["family"],r["implementation"]
        unavailable=key in xsmm and xsmm[key]["admitted"]=="0"
        if unavailable:
            need(not times[key] and not proofs[key] and r["qualified"]=="0","Timed rejected JIT")
        if r["qualified"]=="1":
            need(set(proofs[key])==FIXTURES and max(proofs[key].values())<=0.002 and set(times[key])==set(PAIRS),"Admission")
            admitted[f].add(n)
        else:
            excluded[f+"/"+n]=r["reason"]
    decisions=[];leaders={};all_peers={};seed=[[],[]]
    def pair(f,c,n):
        rr=[times[f,n][i]/times[f,c][i] for i in PAIRS]
        return dict(peer=n,ratios=rr,speedup=median(rr),wins=sum(x>1 for x in rr),
                    seed_medians=[median(rr[:3]),median(rr[3:])],peer_us=1e6*median(times[f,n].values()),
                    candidate_us=1e6*median(times[f,c].values()))
    for f in panel:
        names=admitted[f]
        need({BASELINE,CANDIDATE}<=names,"Missing standalone public entry "+f)
        external={n for n in names if methods[n]["role"]=="independent"}
        ordinary={n for n in external if methods[n]["policy"]=="one-shot"}
        need(ordinary,"No independent one-shot peer "+f)
        best=lambda group:min(group,key=lambda n:(median(times[f,n].values()),n))
        leaders[f]={"field":best(names),"independent":best(ordinary),"including_retained":best(external),
                   "project":best({n for n in names if methods[n]["role"]!="independent"})}
        all_peers[f]={n:pair(f,CANDIDATE,n) for n in sorted(names-{CANDIDATE})}
        if f in fresh:
            pr=all_peers[f][BASELINE];decisions.append(pr["speedup"])
            for b in range(2):seed[b].append(pr["seed_medians"][b])
    gm=geometric_mean(decisions);by_seed=[geometric_mean(x) for x in seed];slow=min(decisions);wins=sum(x>1 for x in decisions)
    result=dict(run=run,method_order=list(methods),leaders=leaders,comparisons=all_peers,excluded=excluded,numerical_maxima=maxerr,
                standalone_gate=dict(geometric_mean=gm,seed_gms=by_seed,shape_wins=wins,minimum=slow,passed=gm>=1.03 and min(by_seed)>1 and wins>=6 and slow>=0.97))
    (root/"frontier-score.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["standalone_gate"]))
if __name__=="__main__":
    score(Path(sys.argv[1]))
