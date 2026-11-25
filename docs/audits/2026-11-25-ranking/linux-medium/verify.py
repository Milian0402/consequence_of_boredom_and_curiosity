"""Rebuild the medium FP32 public tables from their individual measurements, offline."""
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean,median

ARCHES=("x86_64","aarch64")
OLD,NEW,CACHED="medium-incumbent","medium-transposed","onednn-cached-plan"
SEEDS=(68539041,93461257)
ROUNDS=[(b,r) for b in (1,2) for r in (1,2,3)]


def demand(ok,message):
    if not ok: raise ValueError(str(message))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_data(rows):
    out=io.StringIO(newline="")
    writer=csv.DictWriter(out,list(rows[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return out.getvalue()


def number(value):
    return "" if value is None else format(value,".12g")


def calculate(root):
    panel=read(root/"panel.csv")
    batches={r["family"]:max(32,min(512,math.ceil(8e7/(2*int(r["m"])*int(r["n"])*int(r["k"]))))) for r in panel}
    methods=read(root/"methods.csv")
    coverage=read(root/"coverage.csv")
    cpus=json.loads((root/"provenance.json").read_text())["hosts"]
    times=defaultdict(dict); checks=defaultdict(dict); routes={}
    for row in read(root/"timings.csv"):
        key=row["arch"],row["family"],row["implementation"]
        order=int(row["block"]),int(row["round"])
        value=float(row["seconds"]); batch=int(row["calls"])
        demand(order in ROUNDS and order not in times[key] and int(row["seed"])==SEEDS[order[0]-1], key)
        demand(value>0 and math.isfinite(value) and row["threads"]=="1" and batch==batches[row["family"]], key)
        demand(str(cpus[key[0]]["logical_cpu"])==row["cpu"],"timing pin")
        times[key][order]=value
    for row in read(root/"accuracy.csv"):
        key=row["arch"],row["family"],row["implementation"]
        token=row["mode"],int(row["block"]),int(row["seed"])
        demand(token not in checks[key],key)
        demand(row["threads"]=="1" and str(cpus[key[0]]["logical_cpu"])==row["cpu"],"fixture pin")
        checks[key][token]=row
    for row in read(root/"standalone-routes.csv"):
        key=row["arch"],row["family"],row["implementation"]
        demand(key not in routes,"duplicate route")
        routes[key]=row
    eligibility={(r["arch"],r["family"],r["implementation"]):r["qualified"]=="1" for r in coverage}
    demand(len(eligibility)==len(coverage),"duplicate coverage record")
    expected={("random",1,SEEDS[0]),("random",2,SEEDS[1]),("cancel",0,SEEDS[0]),("scaled",0,SEEDS[0])}
    case_rows=[]; method_rows=[]; gate_rows=[]
    for arch in ARCHES:
        names={r["implementation"]:r["role"] for r in methods if r["arch"]==arch and r["status"]=="included"}
        pairs_for_gate=[]; authorized=True
        for c in panel:
            family=c["family"]
            dims={d:int(c[d]) for d in ("m","n","k","pad")}
            batch=batches[family]
            warmed={}; paired={}
            for n in names:
                key=(arch,family,n); eligible=eligibility[key]
                t=times.get(key,{})
                fixtures=checks.get(key,{})
                if eligible:
                    demand(set(t)==set(ROUNDS) and set(fixtures)==expected,key)
                    for row in fixtures.values():
                        demand(row["verified"]=="1" and 0<=float(row["max_abs_error"])<=.002 and
                               row["input_changes"]==row["nonfinite"]==row["guard_changes"]=="0" and
                               int(row["checked"])==dims["m"]*dims["n"],key)
                    if n in (OLD,NEW):
                        route=routes[key]
                        demand(route["passed"]=="1" and int(route["output_elements"])==dims["m"]*dims["n"] and
                               all(route[k]=="1" for k in ("threads","timed_binary_exact","trace_binary_exact",
                                                         "timed_inputs_unchanged","trace_inputs_unchanged")),key)
                        if n==NEW and family.startswith("control"):
                            demand(route["seeded_control_equal"]=="1",key)
                    warmed[n]=1000*median(t.values())
            pools={"independent_oneshot":{n for n in warmed if names[n]=="direct" and n!=CACHED},
                   "independent_all":{n for n in warmed if names[n]=="direct"},
                   "whole_field_oneshot":set(warmed)-{CACHED},"whole_field":set(warmed)}
            winners={k:min(v,key=lambda n:(warmed[n],n)) if v else "" for k,v in pools.items()}
            for n in sorted(names):
                key=arch,family,n; cold=checks.get(key,{})
                def first(b):
                    r=cold.get(("random",b,SEEDS[b-1]))
                    return number(1000*float(r["first_sgemm_seconds"])) if r else ""
                ratios=[]
                if n in warmed and NEW in warmed:
                    ratios=[times[key][r]/times[arch,family,NEW][r] for r in ROUNDS]
                    paired[n]=ratios
                method_rows.append(dict(arch=arch,family=family,implementation=n,qualified=int(eligibility[key]),
                        calls_per_batch=batch,warm_median_ms=number(warmed.get(n)),first_seed1_ms=first(1),
                        first_seed2_ms=first(2),peer_over_candidate=number(median(ratios)) if ratios else "",
                        candidate_paired_wins=sum(x>1 for x in ratios) if ratios else "",pairs=len(ratios)))
            row=dict(arch=arch,family=family,**dims,candidate_ms=number(warmed.get(NEW)),
                     incumbent_ms=number(warmed.get(OLD)),qualified_methods=len(warmed),measured_methods=len(names))
            for group,who in {"incumbent":OLD,**winners}.items():
                row[group]=who
                if group!="incumbent":row[group+"_ms"]=number(warmed.get(who))
                q=paired.get(who)
                row[group+"_over_candidate"]=number(median(q)) if q else ""
                row[group+"_candidate_wins"]=sum(x>1 for x in q) if q else ""
            case_rows.append(row)
            authorized &= OLD in paired and NEW in paired
            if family.startswith("fresh-") and OLD in paired: pairs_for_gate.append(paired[OLD])
        enough=len(pairs_for_gate)==5
        g=geometric_mean(median(x) for x in pairs_for_gate) if enough else None
        blocks=[geometric_mean(median(x[3*b:3*b+3]) for x in pairs_for_gate) for b in (0,1)] if enough else [None,None]
        wins=sum(median(x)>1 for x in pairs_for_gate)
        regression=max(median(1/v for v in x) for x in pairs_for_gate) if enough else None
        passed=authorized and enough and g>=1.05 and wins>=4 and min(blocks)>1 and regression<=1.10
        gate_rows.append(dict(arch=arch,qualified_all_nine=int(authorized),fresh_layouts_scored=len(pairs_for_gate),
                              fresh_geometric_mean=number(g),fresh_paired_winning_layouts=wins,seed1_geomean=number(blocks[0]),
                              seed2_geomean=number(blocks[1]),max_candidate_over_incumbent=number(regression),promote=int(passed)))
    return {"summary.csv":case_rows,"method-summary.csv":method_rows,"decision.csv":gate_rows}


def verify(root):
    file_hashes={name:d for d,name in (line.split("  ",1) for line in (root/"SHA256SUMS").read_text().splitlines())}
    for name,d in file_hashes.items():
        demand(Path(name).name==name and sha(root/name)==d,name)
    demand(set(file_hashes)=={f.name for f in root.iterdir() if f.is_file() and f.name!="SHA256SUMS"},"file inventory")
    for name,rows in calculate(root).items(): demand((root/name).read_text()==csv_data(rows),name)
    print("Verified file hashes, qualification evidence, raw medians, six paired ratios and both CPU decisions.")


if __name__=="__main__":
    verify(Path(__file__).resolve().parent)
