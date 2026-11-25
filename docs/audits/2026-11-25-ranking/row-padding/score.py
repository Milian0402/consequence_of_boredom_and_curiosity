"""Replay the prespecified paired comparison for one CPU's padded BLAS study."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import geometric_mean, median

PANEL = "b6d451644a0a979e1b09796ff22c6b4ff24b537a7a701e68afad72ada14d5190"
SEEDS = (856103219, 1743857903)
KEYS = [(seed, r) for seed in SEEDS for r in (1, 2, 3)]
FIXTURES = {("random", seed) for seed in SEEDS} | {("cancel", SEEDS[0]), ("scaled", SEEDS[0])}


def require(test, message):
    if not test:
        raise ValueError(message)


def records(root, name):
    with (root / name).open(newline="") as source:
        return list(csv.DictReader(source))


def verify(root):
    require(hashlib.sha256((root / "panel.csv").read_bytes()).hexdigest() == PANEL, "changed panel")
    panel = records(root, "panel.csv")
    labels = [r["family"] for r in panel]
    geometries = {r["family"]: tuple(int(r[d]) for d in ("m", "n", "k", "pad")) for r in panel}
    witness = records(root, "host-pad-witness.csv")
    providers = {r["provider"] for r in witness}
    require(len(providers) == 1 and providers <= {"mkl", "aocl"}, "ambiguous provider")
    provider = providers.pop()
    peers = ("host-direct", provider)
    methods = {*peers, "host-pad8"}
    modes = {"host-pad8", "host-direct"}
    admitted = {tuple(r[k] for k in ("implementation", "family")) for r in records(root, "case-eligibility.csv") if r["qualified"]=="1"}
    require({(name, f) for name in methods for f in labels} <= admitted, "missing numerically admitted observation")
    times = defaultdict(dict)
    cpus = set()
    for r in records(root, "timings.csv"):
        name, f=r["implementation"], r["family"]
        if name not in methods: continue
        key=int(r["seed"]), int(r["round"])
        t=float(r["seconds"])
        require(key in KEYS and key not in times[name,f] and r["threads"]=="1" and int(r["calls"])==32 and math.isfinite(t) and t>0, "invalid timing")
        cpus.add(r["cpu"])
        times[name, f][key]=t
    accuracy=defaultdict(set)
    for r in records(root, "accuracy.csv"):
        name,f=r["implementation"],r["family"]
        if name not in methods: continue
        m,n,_,_=geometries[f]
        fixture=(r["mode"], int(r["seed"]))
        err=float(r["max_abs_error"])
        require(fixture in FIXTURES and fixture not in accuracy[name,f] and r["verified"]=="1" and r["nonfinite"]==r["guard_changes"]=="0" and int(r["checked"])==m*n and 0 <= err <= .002 and r["threads"]=="1", "failed accuracy qualification")
        accuracy[name,f].add(fixture);cpus.add(r["cpu"])
    required={(method,f,trace) for method in modes for f in labels for trace in ("0","1")}
    for proof in witness:
        method,f,trace=proof["implementation"],proof["family"],proof["traced"]
        m,n,k,pad=geometries[f]
        rounded=(m+7)//8*8
        padded=method=="host-pad8" and 24<=m<=96 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and 0<8*(rounded-m)<=m
        observed=tuple(int(proof[col]) for col in ("padded","provider_m","provider_lda","provider_ldb","provider_ldc","checked","passed"))
        expected=(int(padded),rounded if padded else m, (k+15)//16*16 if padded else k+pad,n+pad,(n+15)//16*16 if padded else n+pad,m*n,1)
        require(observed==expected and (method,f,trace) in required,"provider witness mismatch")
        required.remove((method,f,trace));cpus.add(proof["cpu"])
    require(not required and len(cpus)==1,"incomplete routing or CPU identity")
    rows=[]
    active={r["family"] for r in witness if r["padded"]=="1"}
    for f in labels:
        for n in methods:
            require(set(times[n,f])==set(KEYS) and accuracy[n,f]==FIXTURES,"incomplete measurement")
        row={"family":f,"padded":f in active,"candidate_us":1e6*median(times["host-pad8",f].values()),"comparisons":{}}
        for n in peers:
            ratios=[times[n,f][k]/times["host-pad8",f][k] for k in KEYS]
            row["comparisons"][n]={"median":median(ratios),"wins":sum(r>1 for r in ratios),"ties":sum(r==1 for r in ratios),"round_ratios":ratios}
        rows.append(row)
    eligible=[r for r in rows if r["padded"] and r["family"].startswith("fresh-")]
    successes=sum(all(c["wins"]>=5 and c["median"]>1.02 for c in r["comparisons"].values()) for r in eligible)
    aggregates={n:geometric_mean(r["comparisons"][n]["median"] for r in eligible) for n in peers}
    minima={n:min(r["comparisons"][n]["median"] for r in eligible) for n in peers}
    return {"provider":provider,"logical_cpu":next(iter(cpus)),"fresh_padded":len(eligible),"fresh_passing_both":successes,
            "geometric_means":aggregates,"minima":minima,"advance":successes>=2 and min(aggregates.values())>1.02 and min(minima.values())>=.97,"rows":rows}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("raw",type=Path)
    print(json.dumps(verify(parser.parse_args().raw),indent=2,allow_nan=False))
