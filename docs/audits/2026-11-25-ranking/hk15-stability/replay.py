"""Recalculate each local stability table and verify matched methods, fixtures and output counts."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
fixtures = {
    "fresh": ["centered","positive","log-range","soft-complement","sparse-complement","reciprocal","tilted"],
    "biased": ["centered","positive","row-col-signs","partial-bias","biased-signed","one-sided","depth-signs"],
}
fixtures["projection-validation"] = [*fixtures["biased"][:-1],"two-depth-profiles","weak-bias","depth-signs"]
fixtures["coherence-validation"] = [*fixtures["biased"][:-1],"two-depth-profiles","three-depth-profiles","rare-group","weak-bias","depth-signs"]
layouts = {
    "fresh": [(1987,3113,3017,170039301),(2251,3203,3181,58699377)],
    "biased": [(2083,3037,3121,800571253),(2233,3307,3187,179693167)],
    "projection-validation": [(1877,2957,2903,218140063),(2317,3457,3379,441085729)],
    "coherence-validation": [(2011,3137,3001,318577921),(2297,3509,3449,277899713)],
}
methods = {"fresh":["old","guarded","support","center","coherence","openblas"],
           "biased":["old","support","center","coherence","openblas","accelerate","classical"],
           "projection-validation":["support","center","coherence","openblas"],
           "coherence-validation":["support","coherence","openblas"]}

def records(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def continuous():
    suites={}
    for panel,shapes in layouts.items():
        table={}
        for method in methods[panel]:
            over=total=nonfinite=0
            peak=0.
            for m,n,k,seed in shapes:
                rows=records(root/panel/f"{method}-{m}.csv")
                assert [r["fixture"] for r in rows]==fixtures[panel]
                for r in rows:
                    assert tuple(map(int,(r["m"],r["n"],r["k"],r["seed"])))==(m,n,k,seed)
                    assert int(r["cells"])==m*n and 0<=int(r["over_abs_0_002"])<=m*n
                    assert (float(r["max_abs"])>.002)==bool(int(r["over_abs_0_002"]))
                    assert 0<=int(r["row"])<m and 0<=int(r["col"])<n
                    over+=int(r["over_abs_0_002"]); nonfinite+=int(r["nonfinite"]); total+=m*n
                    peak=max(peak,float(r["max_abs"]))
            table[method]={"cells":total,"over_0_002":over,"nonfinite":nonfinite,"max_abs":peak}
        suites[panel]=table
    return suites

def exact():
    cases=[("outer",s) for s in (0,4,5,6,7,8,12,16,24)]
    cases += [("inside",s) for s in (8,16,24)]
    cases += [("scatter",8),("scatter",16),("spike",12),("zero-axes",12),("cancel",10),
              ("global",127),("subnormal",0),("complement",8),("complement",16),("complement-tile",16)]
    table={}
    for method in ("issued","openblas","guarded","support","center","coherence"):
        cells=wrong=nonfinite=over=0
        for m,n,k in ((1931,2903,2881),(2117,3071,3013)):
            rows=records(root/"study"/f"{method}-{m}.csv")
            assert [(r["pattern"],int(r["shift"])) for r in rows]==cases
            for r in rows:
                assert (int(r["m"]),int(r["n"]),int(r["k"]),int(r["cells"]))==(m,n,k,m*n)
                assert 0<=int(r["over_0_002"])<=int(r["wrong"])<=m*n
                cells+=m*n; wrong+=int(r["wrong"]); over+=int(r["over_0_002"]); nonfinite+=int(r["nonfinite"])
        table[method]={"cells":cells,"incorrect":wrong,"nonfinite":nonfinite,"over_0_002":over}
    return table

def whole_call():
    result={}
    for experiment,labels in (("timing",{"center","support","classical","openblas"}),
                              ("dispatch",{"coherence","old","classical","accelerate"})):
        for m,n,k,seed in layouts["biased"]:
            rows=records(root/"biased"/f"{experiment}-{m}.csv")
            assert len(rows)==len(fixtures["biased"])*8*4
            for case in fixtures["biased"]:
                cells=defaultdict(dict)
                for row in (r for r in rows if r["fixture"]==case):
                    assert (int(row["m"]),int(row["n"]),int(row["k"]),int(row["seed"]))==(m,n,k,seed)
                    rep=int(row["block"]); method=row["method"]; sec=float(row["seconds"])
                    assert rep in range(8) and method not in cells[rep] and sec>0
                    cells[rep][method]=(sec,int(row["position"]))
                assert set(cells)==set(range(8))
                assert all(set(rep)==labels and {v[1] for v in rep.values()}==set(range(4)) for rep in cells.values())
                result[f"{experiment}-{m}-{case}"]={method:statistics.median(x[method][0] for x in cells.values()) for method in sorted(labels)}
    return result

def guard_overhead():
    output={}
    for method in ("guard","support","coherence"):
        rows=records(root/f"{method}-timings.csv")
        assert len(rows)==144
        groups=defaultdict(dict)
        for r in rows:
            key=(int(r["id"]),int(r["seed"]),int(r["block"]))
            assert int(r["verified_cells"])==int(r["m"])*int(r["n"]) and int(r["reference_spots"])==96
            assert r["method"] not in groups[key]
            groups[key][r["method"]]=float(r["seconds"])
        ratios=defaultdict(list); noise=defaultdict(list)
        assert len(groups)==48
        for (shape,seed,block),times in groups.items():
            assert set(times)==set("ABG") and min(times.values())>0
            ratios[shape].append(math.sqrt(times["A"]*times["B"])/times["G"])
            noise[shape].append(times["A"]/times["B"])
        assert all(len(values)==12 for values in ratios.values())
        med={key:statistics.median(values) for key,values in ratios.items()}
        output[method]={"speed_ratios":med,"geomean":statistics.geometric_mean(med.values()),
                        "identical_function_ratios":{key:statistics.median(values) for key,values in noise.items()}}
    return output

if __name__=="__main__":
    print(json.dumps({"exact":exact(),"continuous":continuous(),"whole_call_seconds":whole_call(),
                      "normal_data_overhead":guard_overhead()},indent=2,sort_keys=True,allow_nan=False))
