"""Reconstruct the full-output accuracy, interleaved call schedule and paired whole-call ratios."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import geometric_mean, median

AXES = ("m", "n", "k", "pad")
SLOTS = [(b, r) for b in (1, 2) for r in (1, 2, 3)]


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def rows(file):
    with Path(file).open(newline="") as handle:
        return list(csv.DictReader(handle))


def shape(row):
    return tuple(int(row[a]) for a in AXES)


def single(items, key):
    result = {}
    for row in items:
        identifier = key(row)
        demand(identifier not in result, f"Duplicate observation {identifier}")
        result[identifier] = row
    return result


def details(row):
    items = [p.split("=", 1) for p in row["detail"].split(",")]
    value = dict(items)
    demand(len(items) == len(value), "Duplicate worker diagnostics")
    return value


def positive(value):
    value = float(value)
    demand(math.isfinite(value) and value > 0, f"Invalid complete-call time {value}")
    return value


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=2e-10, abs_tol=1e-14)


def compare(new, old):
    values = [old[p] / new[p] for p in SLOTS]
    leave_one = [median(values[:i] + values[i+1:]) for i in range(len(values))]
    return {"median": median(values), "wins": sum(x > 1 for x in values), "ties":sum(x==1 for x in values),
            "ratios": values, "seed_medians": [median(values[:3]), median(values[3:])],
            "min": min(values), "max": max(values), "geomean": geometric_mean(values),
            "leave_one_out_median_range": [min(leave_one), max(leave_one)]}


def measure(root, phase, panel, roles, seeds, route_gate, method_order):
    names = list(method_order)
    ranking = json.loads((root / "ranking.json").read_text())
    methods = set(names)
    demand(len(methods)==len(names) and set(roles)==methods==set(ranking["implementations"]), "Method order/membership")
    demand(ranking["seeds"] == list(seeds) and ranking["completed_cases"] == ranking["expected_cases"] == len(panel)
           and ranking["phase"] == phase and ranking["rounds_per_seed"] == 3 and ranking["full_output_tolerance"] == 0.002,
           "Recorded experiment does not match frozen inputs")
    fixtures = {("random",b,s) for b,s in enumerate(seeds,1)} | {("cancel",0,seeds[0]),("scaled",0,seeds[0])}
    cohort = [(f,n) for f in panel for n in names]
    ident = lambda r: (r["family"],r["implementation"],r["mode"],int(r["block"]),int(r["seed"]))
    checks = single(rows(root/"accuracy.csv"),ident)
    first = single(rows(root/"first-calls.csv"),ident)
    skipped={(f,n) for f,n in cohort if n.startswith("libxsmm-") and route_gate.get((f,n)) is False}
    demand(set(checks) == set(first) == {(f,n,*x) for f,n in cohort if (f,n) not in skipped for x in fixtures},
           "Four distinct fixtures and unbatched first call required for every enabled method and case")
    cpus=set(); acc=defaultdict(dict); t0=defaultdict(dict); times=defaultdict(dict); residents=defaultdict(set)
    fail_random=set(); checked=0; zero_checks=Counter(); batch_counts=Counter(); cpu_pins=0
    for (f,n,mode,b,seed), r in checks.items():
        key=(f,n,mode,b,seed); fr=first[key]; a=details(r); e=float(r["max_abs_error"])
        lo,hi=float(a["signed_min_error"]),float(a["signed_max_error"])
        demand(shape(r)==shape(fr)==panel[f] and r["phase"]==phase and r["stress"]==str(int(mode!="random"))
               and r["threads"]==a["threads"]=="1" and r["cpu"].isdigit() and r["cpu"]==a["cpu"],f"FP64 input/affinity {key}")
        demand(int(r["checked"])==panel[f][0]*panel[f][1],f"Partial FP64 validation {key}")
        counts={k:int(v) for k,v in (("guard_changes",r["guard_changes"]),("nonfinite",r["nonfinite"]),("input_changes",a["input_changes"]))}
        demand(all(v>=0 for v in counts.values()),f"Invalid safety counters {key}")
        demand(not counts["nonfinite"] and math.isfinite(e) and e >= 0 and math.isfinite(lo) and math.isfinite(hi)
               and lo<=hi and close(max(abs(lo),abs(hi)),e) and math.isfinite(float(a["checksum"])),
               f"Nonfinite or inconsistent FP64 evidence {key}")
        demand(counts["guard_changes"]==counts["input_changes"]==0, f"Memory/input integrity failure {key}")
        performance_ok = e<=0.002
        demand(r["verified"]==fr["verified"]==str(int(performance_ok)),f"Misreported full output tolerance {key}")
        onednn=a.get("onednn_impl","")
        demand(onednn==fr["onednn_impl"] and (not n.startswith("onednn-") or bool(onednn)), f"oneDNN implementation identity {key}")
        seconds=positive(a["first_sgemm_seconds"])
        demand(seconds==positive(fr["first_sgemm_seconds"]),f"First-call duration {key}")
        if not performance_ok and mode=="random":fail_random.add((f,n,b))
        acc[f,n][mode,b,seed]=dict(passed=performance_ok,max_abs=e,signed_min=lo,signed_max=hi)
        if mode=="random":t0[f,n][b]=seconds
        cpus.add(int(r["cpu"]));checked+=int(r["checked"]);zero_checks.update(counts)
    actual_schedule=[]
    for r in rows(root/"timings.csv"):
        f,n=r["family"],r["implementation"]; b,i=int(r["block"]),int(r["round"]); p=int(r["position"])
        m,w,k,_=panel[f]
        count=int(r["calls"])
        demand((b,i) in SLOTS and n in methods and (b,i) not in times[f,n] and shape(r)==panel[f]
               and r["mode"]=="random" and r["stress"]=="0" and r["phase"]==phase and int(r["seed"])==seeds[b-1]
               and r["threads"]=="1" and r["cpu"].isdigit() and r["resident"] in ("0","1")
               and count==max(32,min(512,math.ceil(8e7/(2*m*w*k)))),f"Invalid measured full-call batch {f} {n} {b} {i}")
        cpus.add(int(r["cpu"]));residents[f].add(r["resident"]);times[f,n][b,i]=positive(r["seconds"])
        actual_schedule.append((f,n,b,i,p));batch_counts[count]+=1
    expected_schedule=[]
    for b in (1,2):
        for f in (list(panel) if b==1 else list(reversed(panel))):
            for i in (1,2,3):
                present=[n for n in names if (f,n) not in skipped]
                offset=(list(panel).index(f)+i-1)%len(present)
                roster=present[offset:]+present[:offset]
                if (b+i)%2:roster.reverse()
                expected_schedule.extend((f,n,b,i,p) for p,n in enumerate(roster,1) if (f,n,b) not in fail_random)
    demand(expected_schedule==actual_schedule and all(len(v)==1 for v in residents.values()),"Counterbalance, matrix residency or observation order differs")
    demand(len(cpus)==1,"Workers moved across logical CPUs")
    recorded=single(rows(root/"case-eligibility.csv"),lambda r:(r["family"],r["implementation"]))
    demand(set(recorded)==set(cohort),"Incomplete case/method qualification inventory")
    qualified=defaultdict(set); excludes=defaultdict(dict)
    for f,n in cohort:
        data=acc[f,n]; t=times[f,n]; count=sum(r["passed"] for r in data.values()); route=route_gate.get((f,n),True)
        ok=count==len(fixtures) and len(t)==len(SLOTS) and route
        q=recorded[f,n]
        demand(shape(q)==panel[f] and int(q["verified_cases"])==count and int(q["warm_pairs"])==len(t)
               and q["qualified"]==str(int(ok)) and (q["reason"]=="all_checks_passed")==ok,f"Contradictory admission {f} {n}")
        if ok:qualified[f].add(n)
        else:
            excludes[f][n]=dict(reason=q["reason"],route=route,failures=[dict(mode=x[0],block=x[1],seed=x[2],**rec) for x,rec in data.items() if not rec["passed"]])
    actual=single(rows(root/"comparisons.csv"),lambda r:(r["family"],r["implementation"],r["peer"]))
    expected={(f,n,p) for f,ns in qualified.items() for n in ns for p in ns if n!=p}
    demand(set(actual)==expected,"Incomplete or unqualified published pair")
    for (f,n,p),r in actual.items():
        metric=compare(times[f,n],times[f,p])
        demand(shape(r)==panel[f] and r["phase"]==phase and r["mode"]=="random" and int(r["pairs"])==6 and
               int(r["paired_wins"])==metric["wins"] and all(close(r[key],metric[k]) for key,k in
                (("median_speedup","median"),("minimum_speedup","min"),("maximum_speedup","max"))),f"Published paired ratio {f} {n} {p}")
    posted=single(ranking["shape_rankings"],lambda r:r["family"])
    demand(set(posted)==set(panel),"Missing or extra hosted per-geometry ranking")
    for f,g in posted.items():
        measured={n:median(times[f,n].values()) for n in qualified[f]}
        demand(shape(g)==panel[f] and set(measured)==set(g["median_seconds"]) and all(close(g["median_seconds"][n],v) for n,v in measured.items())
               and measured[g["fastest_median"]]==min(measured.values()),f"Hosted ranking diverges on {f}")
    return (times,t0,qualified,excludes,acc,dict(warm_observations=len(actual_schedule),warm_calls=sum(k*v for k,v in batch_counts.items()),
            calls_per_observation=dict(sorted(batch_counts.items())),accuracy_fixtures=len(checks),checked_outputs=checked,logical_cpu=cpus.pop(),
            qualified_cells=sum(map(len,qualified.values())),failed_cells=sum(map(len,excludes.values())),**zero_checks))
