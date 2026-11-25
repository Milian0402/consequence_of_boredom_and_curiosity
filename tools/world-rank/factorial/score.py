"""Verify and score the six-factor parent-matrix layout experiment. Ratios are high/low elapsed time."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from statistics import fmean,geometric_mean

PANEL_SHA="5742b14e8e2fd71993ce98070fabf85b648f1ad69efb7f16ff07f2a8eb246631"
BASE="a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6"
FACTORS=("A_stride","B_stride","C_stride","A_origin","B_origin","C_origin")
FAMILIES=("repeat-win-deeper","repeat-win-ragged","fresh-short")
METHODS=("stream","mkl")
SEEDS=(73194513,38641729)
FIXTURES=(("random",SEEDS[0]),("random",SEEDS[1]),("cancel",SEEDS[0]),("scaled",SEEDS[1]))
ROUNDS=range(6)
MASKS=range(64)

def need(ok,text):
    if not ok: raise ValueError(text)

def read(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def pos(x):
    v=float(x);need(math.isfinite(v) and v>0,"Invalid timing "+str(v));return v

def idx(records,key):
    d={}
    for r in records:
        k=key(r);need(k not in d,f"Duplicate record {k}");d[k]=r
    return d

def layouts(root):
    raw=(root/"panel.csv").read_bytes()
    need(sha(raw)==PANEL_SHA,"Frozen factorial panel changed")
    p=idx(read(root/"panel.csv"),lambda r:(r["family"],int(r["mask"])))
    need(set(p)==set(itertools.product(FAMILIES,MASKS)),"Incomplete factorial design")
    for (f,mask),r in p.items():
        lo,hi=p[f,0],p[f,63]
        for i,k in enumerate(("lda","ldb","ldc","ao","bo","co")):
            need(r[k]==(hi if mask&(1<<i) else lo)[k] and hi[k]!=lo[k],f"Unexpected axis value {f} {mask} {k}")
        need(all(r[k]==lo[k] for k in ("m","n","k")) and int(r["lda"])>=int(r["k"]) and
             min(int(r["ldb"]),int(r["ldc"]))>=int(r["n"]),f"Invalid logical rectangle {f} {mask}")
    return p

def schedule():
    order=[]
    for r in ROUNDS:
        t=r%3
        cells=[(s,i,(37*s+11*i+17*t)%64) for s in MASKS for h in range(3) for i in [(h+t)%3]]
        if r>=3:cells.reverse()
        for seq,(s,i,mask) in enumerate(cells):
            first=(i+s+t+int(r>=3))%2
            for place,n in enumerate((METHODS[first],METHODS[1-first])):
                order.append((FAMILIES[i],mask,r,seq,place,n))
    return order

def verify(root,panel):
    page=int((root/"page-size.txt").read_text())
    need(page>=64 and page&(page-1)==0,"Invalid native memory page size")
    rows=read(root/"raw.csv")
    need(len(rows)==3*64*2*10,"Missing or additional observations")
    expected=set(itertools.product(FAMILIES,MASKS,METHODS,FIXTURES))
    data={}; cold={};pre=set();cpus=set(); hashes=defaultdict(set);chashes=defaultdict(set);plain={};warm=[];worst=defaultdict(float); count=0;guards=0
    for r in rows:
        f=r["family"];mask=int(r["mask"]);n=r["method"];seed=int(r["seed"]);mode=r["mode"];rd=int(r["round"])
        case=panel[f,mask];m,w,k,lc,co=map(int,(case["m"],case["n"],case["k"],case["ldc"],case["co"]))
        scope=f,mask,n
        need(all(int(r[k+"_observed"])==int(case[k]) for k in ("ao","bo","co","lda","ldb","ldc")),
             f"Measured worker leading dimension or pointer offset differs {scope}")
        need(n in METHODS and r["threads"]=="1" and int(r["cpu"])>=0,"Unexpected method or worker task/CPU")
        cpus.add(int(r["cpu"]))
        observed=tuple(int(r[a+"_mod64"]) for a in "abc")
        predicted=tuple(4*int(case[a+"o"])%64 for a in "abc")
        allocated=((4*(co//lc+m+2)*lc+page-1)//page)*page//4
        # For a flat parent, a crop may straddle a nominal parent row at mixed corners.
        # All logical rows remain disjoint; guards include the unused prefix and suffix.
        need(corner_fits(co,lc,m,w,allocated) and observed==predicted and int(r["checked"])==m*w and int(r["guards"])==allocated-m*w,
             f"Actual storage or whole-parent guard coverage {scope}")
        error=float(r["error"]);need(math.isfinite(error) and 0<=error<=.002,f"FP64 tolerance {scope}")
        fields={a:r[a+"_hash"].lower() for a in "abc"};need(all(re.fullmatch(r"[0-9a-f]{16}",v) for v in fields.values()),"Incomplete data checksum")
        hashes[f,mode,seed].add((fields["a"],fields["b"]))
        chashes[f,mask,n,mode,seed].add(fields["c"])
        first=pos(r["first_s"]);worst[n]=max(worst[n],error);count+=m*w;guards+=allocated-m*w
        if rd<0:
            token=(f,mask,n,(mode,seed))
            need(token in expected and token not in pre and rd==int(r["sequence"])==int(r["place"])==-1 and int(r["calls"])==0 and float(r["per_call_s"])==0,
                 f"Preflight fixtures {token}")
            pre.add(token)
            if n=="stream" and mode=="random" and seed==SEEDS[0]:plain[f,mask]=r
            if mode=="random":cold[f,mask,n,seed]=first
        else:
            need(rd in ROUNDS and seed==SEEDS[rd//3] and mode=="random" and int(r["calls"])==16 and (*scope,rd) not in data,
                 f"Warm timing {scope} round {rd}")
            data[(*scope,rd)]=pos(r["per_call_s"])
            warm.append((f,mask,rd,int(r["sequence"]),int(r["place"]),n))
    need(pre==expected and warm==schedule() and len(cpus)==1,"Incomplete fixture matrix, changed warm order or inconsistent affinity")
    need(all(len(v)==1 for v in hashes.values()),"Logical A or B differed between methods, masks, or matched observations")
    # A and B have different normal/scaled fixtures at both seeds; cancellation is seed independent by construction.
    digest_input={f:{f"{mo}/{s}":list(next(iter(hashes[f,mo,s]))) for mo,s in FIXTURES} for f in FAMILIES}
    c_variation=[dict(family=f,mask=m,method=n,mode=mo,seed=s,digests=sorted(v)) for (f,m,n,mo,s),v in chashes.items() if len(v)>1]
    native=native_evidence(root,panel,plain,next(iter(cpus)))
    return data,cold,dict(coverage=dict(warm_observations=len(warm),warm_complete_calls=16*len(warm),preflight_fixtures=len(pre),
                          independent_fp64_checked_positions=count,checked_parent_guard_positions=guards,logical_cpu=cpus.pop(),
                          page_bytes=page,max_abs=worst),inputs=digest_input,native=native,c_digest_variation_within_same_method_mask_and_input=c_variation)

def native_evidence(root,panel,ordinary,cpu):
    src=json.loads((root/"sources.json").read_text())
    valid=lambda s:bool(re.fullmatch(r"[0-9a-f]{64}",s))
    need(src["release"]==BASE and src["tree"]=="18f22622de1a4b1b7aebbdb66985741aa38c6e9e" and
         src["historical_benchmark"]=="bce9e5367a2b1b414207460f635ac498663b213f" and src["panel_sha256"]==PANEL_SHA and
         src["reference_panel_sha256"]==sha((root/"previous-panel.csv").read_bytes())=="ab9115f17a698938d14e3fb0010f05e450b214eca8876d7fd6d59e2323b17c80" and
         re.fullmatch(r"[0-9a-f]{40}",src["benchmark"]) and src["cpu"]==cpu,"Production source, actual CPU or historical controls differ")
    receipt=src["source_sha256"]
    for k in ("panel.csv","protocol.md","factors.json"):
        need(sha((root/k).read_bytes())==receipt[k],"Measured file differs from source "+k)
    need(valid(src["trace_worker_sha256"]) and valid(receipt["score.py"]) and all(valid(receipt[k]) for k in ("run.sh","worker.c","study.py","estimands.md","historical-endpoints.csv")),
         "Incomplete native benchmark source inventory")
    bins=idx(read(root/"binaries.csv"),lambda x:x["method"])
    need(set(bins)==set(METHODS) and valid(bins["stream"]["worker_sha256"]) and bins["stream"]["worker_sha256"]==bins["mkl"]["worker_sha256"]!=src["trace_worker_sha256"] and
         bins["stream"]["adapter_sha256"]=="none" and bins["mkl"]["archive_sha256"]=="none" and valid(bins["mkl"]["adapter_sha256"]) and valid(bins["stream"]["archive_sha256"]),
         "Standalone installed worker, normal sequential MKL adapter or trace identity differs")
    factors=json.loads((root/"factors.json").read_text())
    need(factors["bits"]==["lda","ldb","ldc","ao","bo","co"] and factors["methods"]==list(METHODS) and factors["seeds"]==list(SEEDS) and factors["rounds"]==6 and factors["calls"]==16,
         "Factor definitions differ")
    old=idx(read(root/"previous-panel.csv"),lambda r:(r["family"],r["layout"]))
    for f in FAMILIES:
        for mask,level,prior in [(0,"low","aligned"),(63,"high","cropped")]:
            p=panel[f,mask];o=old[f,prior];d=factors["families"][f][level]
            need(all(int(p[k])==d[k] for k in d) and all(o[k]==p[k] for k in ("m","n","k","lda","ldb","ldc")) and
                 all(int(p[a+"o"])==int(o[a+"r"])*int(o["ld"+a])+int(o[a+"c"]) for a in "abc"),"Historical corner ABI mismatch "+f)
    features=(root/"isa.txt").read_text().split()
    need(features in (["1","0"],["1","1"]),"Unqualified x86 CPU feature set")
    avx512=int(features[1]);routes=idx(read(root/"routes.csv"),lambda r:(r["family"],int(r["mask"])))
    need(set(routes)==set(panel)==set(ordinary),"Missing actual native route")
    for key,r in routes.items():
        o=ordinary[key]
        need(all(r[k]==o[k] for k in ("cpu","checked","a_mod64","b_mod64","c_mod64","a_hash","b_hash","c_hash")) and
             all(r[k]=="1" for k in ("public_calls","args_ok","alignment_ok")) and all(r[k]=="0" for k in ("packed_avx2","packed_avx512","refused")) and
             int(r["avx512"])==avx512 and int(r["avx2"])==1-avx512,"Incorrect actual native stream route or trace result "+str(key))
    return dict(release=src["release"],benchmark=src["benchmark"],run=src["run"],attempt=src["attempt"],
                binaries=bins,trace_worker_sha256=src["trace_worker_sha256"],actual_isa="avx512" if avx512 else "avx2",matched_trace_cells=len(routes))

def corner_fits(offset,ld,rows,cols,cap):
    return offset>=0 and ld>=cols and offset+(rows-1)*ld+cols<=cap

def descriptors():
    return [("main",i,None) for i in range(6)]+[("interaction",i,j) for i,j in itertools.combinations(range(6),2)]

def stats(logs):
    gs=[math.exp(x) for x in logs]
    return dict(ratio=math.exp(fmean(logs)),round_ratios=gs,seed_ratios=[math.exp(fmean(logs[:3])),math.exp(fmean(logs[3:]))],
                rounds_above_one=sum(x>0 for x in logs),rounds_below_one=sum(x<0 for x in logs),min_ratio=min(gs),max_ratio=max(gs))

def contrasts(logtime, i,j):
    a=1<<i
    if j is None:
        return fmean(logtime[m|a]-logtime[m] for m in MASKS if not m&a)
    b=1<<j
    return fmean((logtime[m|a|b]-logtime[m|b])-(logtime[m|a]-logtime[m]) for m in MASKS if not m&(a|b))

def calculate(data):
    effects=[];ends=[];edge_raw={};conditional=[]
    for f in FAMILIES:
        for n in (*METHODS,"stream_over_mkl"):
            lg=[{m:(math.log(data[f,m,"stream",r])-math.log(data[f,m,"mkl",r]) if n=="stream_over_mkl" else math.log(data[f,m,n,r]))
                 for m in MASKS} for r in ROUNDS]
            ends.append(dict(family=f,method=n,**stats([x[63]-x[0] for x in lg])))
            if n in METHODS:
                edge_raw[f,n]={mask:[data[f,mask,n,r] for r in ROUNDS] for mask in (0,63)}
            for kind,i,j in descriptors():
                v=[contrasts(x,i,j) for x in lg]
                effects.append(dict(family=f,method=n,kind=kind,factor=FACTORS[i],other=FACTORS[j] if j is not None else "",**stats(v)))
                if j==i+3:
                    a,b=1<<i,1<<j
                    for bit in (0,1):
                        cond=[fmean(x[m|a]-x[m] for m in MASKS if not m&a and bool(m&b)==bool(bit)) for x in lg]
                        conditional.append(dict(family=f,method=n,stride=FACTORS[i],origin=FACTORS[j],origin_level=bit,**stats(cond)))
    groups=defaultdict(list)
    for row in effects:groups[row["method"],row["kind"],row["factor"],row["other"]].append(row)
    pooled=[]
    for (n,kind,i,j),g in groups.items():
        vals={r["family"]:r["ratio"] for r in g}
        pooled.append(dict(method=n,kind=kind,factor=i,other=j,geometric_mean=geometric_mean(vals.values()),geometry_ratios=vals,
                           geometries_above_one=sum(x>1 for x in vals.values()),geometries_below_one=sum(x<1 for x in vals.values())))
    return effects,ends,pooled,conditional,edge_raw

def flatten(outname,values):
    rows=[]
    for raw in values:
        row=dict(raw)
        for key in ("round_ratios","seed_ratios"):
            for i,val in enumerate(row.pop(key,[]),1):row[f"{key[:-1]}_{i}"]=val
        rows.append(row)
    with outname.open("w",newline="") as h:
        wr=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");wr.writeheader();wr.writerows(rows)

def historical(path,time,model):
    need(path.is_file() and sha(path.read_bytes())=="97804ff015e676a2ff5ad8fe752a79a8d12ab6a0852a345bcbc8c7f903abe4df",
         "Cannot match prior source-bound endpoint observations")
    rows=read(path);h=idx(rows,lambda x:(x["family"],x["method"],int(x["mask"]),int(x["round"])))
    need(set(h)==set(itertools.product(FAMILIES,METHODS,(0,63),ROUNDS)),"Incomplete historical endpoint evidence")
    need(all(int(x["seed"])==SEEDS[r//3] and x["historical_run"]=="45699142561" for (f,n,m,r),x in h.items()),"Historic endpoint pairing changed")
    output={}
    for f in FAMILIES:
        for n in METHODS:
            old=[math.log(pos(h[f,n,63,r]["per_call_s"])/pos(h[f,n,0,r]["per_call_s"])) for r in ROUNDS]
            now=[math.log(time[f,n][63][r]/time[f,n][0][r]) for r in ROUNDS]
            a,b=stats(old),stats(now)
            oldcpu=h[f,n,0,0]["model"]
            output.setdefault(f,{})[n]=dict(historical_model=oldcpu,same_model=model==oldcpu,prior=a,current=b,
                                           current_over_prior_effect=b["ratio"]/a["ratio"],direction_repeated=(a["ratio"]-1)*(b["ratio"]-1)>0,
                                           corresponding_round_direction_matches=sum(x*y>0 for x,y in zip(old,now)))
    return output

def score(root,out):
    p=layouts(root);data,cold,proof=verify(root,p)
    effects,ends,pooled,conditional,edge_raw=calculate(data)
    host=(root/"host.txt").read_text()
    hit=re.search(r"^Model name:\s*(.+)$",host,re.M);need(hit,"Actual CPU model unavailable")
    hfile=Path(__file__).with_name("historical-endpoints.csv")
    if (root/"historical-endpoints.csv").exists():hfile=root/"historical-endpoints.csv"
    result=dict(model=hit[1].strip(),seeds=SEEDS,families=FAMILIES,methods=METHODS,panel_sha256=PANEL_SHA,**proof,
                effects=effects,endpoints=ends,descriptive_equal_geometry_means=pooled,conditional_stride_effects=conditional,
                historical_endpoints=historical(hfile,edge_raw,hit[1].strip()))
    out.mkdir(parents=True,exist_ok=True)
    (out/"score.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    flatten(out/"effects.csv",[r for r in effects if r["kind"]=="main"])
    flatten(out/"interactions.csv",[r for r in effects if r["kind"]=="interaction"])
    return result

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("raw",type=Path);cli.add_argument("--output",type=Path)
    a=cli.parse_args();report=score(a.raw,a.output or a.raw);print(json.dumps(report["coverage"]))
