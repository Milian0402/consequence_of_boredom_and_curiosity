import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

def load(root,file):
    with (root/file).open(newline="") as stream:return list(csv.DictReader(stream))

def score(root):
    accuracy=load(root,"accuracy.csv")
    times=load(root,"time.csv")
    controls={(r["shape"],r["seed"],r["fixture"]):r for r in accuracy if r["method"]=="cob"}
    local=defaultdict(list);cost=defaultdict(dict)
    for r in accuracy:
        k=(r["shape"],r["method"]);d=controls[r["shape"],r["seed"],r["fixture"]]
        if r["route"]!="executed":
            local[k].append((False,r["route"]));continue
        maxlim=2.*float(d["max_abs"])+2.**-23*float(d["ref_max"])
        rmslim=2.*float(d["rms"])+2.**-23*float(d["ref_rms"])
        ok=int(r["over_abs"])==0 and int(r["nonfinite"])==0 and float(r["max_abs"])<=maxlim and float(r["rms"])<=rmslim
        local[k].append((ok,{"seed":int(r["seed"]),"fixture":r["fixture"],"max":float(r["max_abs"]),
                        "rms":float(r["rms"]),"over_abs":int(r["over_abs"]),"max_limit":maxlim,"rms_limit":rmslim}))
    for r in times:
        k=(r["shape"],r["method"]);p=(int(r["seed"]),int(r["round"]))
        assert p not in cost[k] and float(r["seconds"])>0
        cost[k][p]=float(r["seconds"])
    result={}
    for h in sorted({r["shape"] for r in accuracy},key=int):
        methods={}
        for name in [m for h1,m in local if h1==h]:
            k=(h,name);checks=local[k]
            assert len(checks)==4
            values=list(cost[k].values())
            methods[name]={"accuracy_pass":all(c[0] for c in checks),"failed":[c[1] for c in checks if not c[0]],
                           "timed_calls":len(values),"median_seconds":statistics.median(values) if values else None}
        strong=[name for name,v in methods.items() if not name.startswith("fp16-") and v["accuracy_pass"] and v["timed_calls"]==12]
        fastest=min(strong,key=lambda x:methods[x]["median_seconds"]) if strong else None
        raw_controls=[name for name,v in methods.items() if not name.startswith("fp16-") and v["timed_calls"]==12]
        fastest_time=min(raw_controls,key=lambda x:methods[x]["median_seconds"])
        for method in (name for name in methods if name.startswith("fp16-")):
            if methods[method]["timed_calls"]!=12:continue
            for key,peer in [("vs_qualified",fastest),("vs_fastest_fp32_raw_time",fastest_time)]:
                if peer:
                    r=[cost[h,peer][p]/z for p,z in cost[h,method].items()]
                    methods[method][key]={"peer":peer,"paired_median_speed":statistics.median(r),
                                         "paired_wins":sum(x>1. for x in r)}
        result[h]={"best_qualified_fp32":fastest,"fastest_native_fp32_raw_time":fastest_time,"methods":methods}
    return result

if __name__=="__main__":print(json.dumps(score(Path(sys.argv[1])),indent=2,allow_nan=False))
