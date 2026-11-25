"""Recompute qualifications and paired rankings from the individual worker records."""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def main(root, status):
    panel = read(root / "panel.csv")
    cases = {row["family"]: row for row in panel}
    provider = (root / "leaf.txt").read_text().strip()
    require(provider in ("mkl", "openblas"), "unknown provider")
    peers = ["openblas", "mkl", "mkl-jit", "aocl"] if provider == "mkl" else ["openblas", "armpl"]
    methods = [*peers, "form48", "form49"]
    accuracy = defaultdict(dict)
    times = defaultdict(dict)
    failures = []
    cpus = set()
    expected_fixtures = {("random",35485721),("random",93392083),("scaled",35485721),("cancel",35485721)}
    shapes = ("m","n","k","pad")
    route_rows = read(root / "routes.csv")
    keys = {(int(row["rank"]), *(row[f] for f in shapes)) for row in route_rows}
    require(len(route_rows) == len(panel)*2 and keys == {(rank, *(r[f] for f in shapes)) for r in panel for rank in (48,49)}, "missing routes")
    for r in route_rows:
        require(int(r["core_calls"]) == int(r["rank"]) and int(r["fringe_calls"]) == (int(r["m"])%4 != 0)+(int(r["n"])%4 != 0) and
                int(r["checked"]) == int(r["m"])*int(r["n"]) and r["errors"] == "0" and r["inputs_unchanged"] == r["route_verified"] == "1", "invalid route")
    for r in read(root / "accuracy.csv"):
        case, method = r["family"], r["implementation"]
        require(method in methods and case in cases and all(r[f] == cases[case][f] for f in shapes), "shape or method mismatch")
        key = r["mode"],int(r["seed"])
        require(key in expected_fixtures and key not in accuracy[case,method], "unexpected fixture")
        details = dict(t.split("=", 1) for t in r["detail"].split(",") if "=" in t)
        require(r["threads"]=="1" and r["cpu"] == details["cpu"] and details["threads"] == "1" and
                r["guard_changes"] == r["nonfinite"] == details["input_changes"] == "0" and
                int(r["checked"]) == int(r["m"])*int(r["n"]), "validation infrastructure or integrity failure")
        err = float(r["max_abs_error"]); require(math.isfinite(err), "nonfinite error")
        require((r["verified"]=="1") == (err <= 0.002), "inconsistent validation")
        cpus.add(r["cpu"])
        accuracy[case,method][key] = err
        if err > 0.002: failures.append((case,method,key,err))
    for r in read(root / "timings.csv"):
        key = int(r["block"]), int(r["round"])
        require(r["cpu"] in cpus and r["threads"] == "1" and r["calls"] == "1", "invalid timed worker")
        obs = times[r["family"],r["implementation"]]
        require(key not in obs, "repeated timing")
        t=float(r["seconds"]); require(t>0 and math.isfinite(t),"invalid elapsed time"); obs[key]=t
    expected_pairs = {(b,j) for b in (1,2) for j in (1,2,3)}
    require(len(cpus)==1 and bool(status)==bool(failures),"unexplained process status")
    output=[]
    for case in cases:
        qualified={}
        for method in methods:
            require(set(accuracy[case,method]) == expected_fixtures, f"missing validation for {case} {method}")
            if max(accuracy[case,method].values()) <= 0.002:
                require(set(times[case,method])==expected_pairs, f"missing timed pairs for {case} {method}")
                qualified[method] = times[case,method]
        valid_peers=[p for p in peers if p in qualified]
        require(valid_peers,"no admissible direct comparator")
        best=min(valid_peers,key=lambda p:statistics.median(qualified[p].values()))
        row={"family":case, "best_direct":best, "best_direct_ms":1000*statistics.median(qualified[best].values())}
        for form in ("form48","form49"):
            row[form+"_max_abs"]=max(accuracy[case,form].values())
            row[form+"_ms"]=1000*statistics.median(qualified[form].values()) if form in qualified else ""
        for target,peer in (("rank49","form49"),("direct",best)):
            ratios=[qualified[peer][key]/qualified["form48"][key] for key in sorted(expected_pairs)] if "form48" in qualified and peer in qualified else []
            row["speedup_vs_"+target] = statistics.median(ratios) if ratios else ""
            row["wins_vs_"+target] = sum(x>1 for x in ratios) if ratios else ""
        output.append(row)
    with (root/"scores.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=output[0]);writer.writeheader();writer.writerows(output)
    print(f"{sum(len(d) for d in accuracy.values())} full-output fixtures, {len(failures)} numerical failures, {sum(map(len,times.values()))} timed observations")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
