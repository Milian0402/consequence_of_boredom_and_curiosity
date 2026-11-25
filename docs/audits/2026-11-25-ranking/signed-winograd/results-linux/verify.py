"""Replay the signed Winograd public rankings from full per-layout measurement records."""
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median

ARCHES = ("x86_64", "aarch64")
SEEDS = (31048579, 196173023)
ROUNDS = tuple((b,r) for b in (1,2) for r in (1,2,3))
FIXTURES = {("random",1,SEEDS[0]),("random",2,SEEDS[1]),("cancel",0,SEEDS[0]),("scaled",0,SEEDS[0])}
ODD = {"deep-odd-followup", "wide-odd-followup"}
CACHED = "onednn-cached-plan"


def check(ok, message):
    if not ok:
        raise ValueError(str(message))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_text(rows, fields=None):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def fmt(value):
    return format(value, ".12g")


def verify_linked_builds(root, hosts, panel):
    builds = read(root/"builds.csv")
    parity = read(root/"parity.csv")
    ids = {(r["arch"],r["implementation"]):r for r in builds}
    proofs = {(r["arch"],r["family"],r["provider"]):r for r in parity}
    check(len(ids) == len(builds) == 8 and len(proofs) == len(parity) == 20, "linked build and parity inventory")
    for arch in ARCHES:
        providers = ("openblas", "mkl" if arch=="x86_64" else "armpl")
        for provider in providers:
            old,new = (ids[arch,role+"-"+provider] for role in ("original","signed"))
            sdk = hosts[arch]["matched_sequential_sdks"][provider]
            check(all(old[k] == new[k] for k in ("base_commit","patch_sha256","gemm_sha256","winograd_sha256","vendor_sha256","pristine_library_sha256"))
                  and old["signed"] == "0" and new["signed"] == "1"
                  and old["library_sha256"] != new["library_sha256"] and bool(sdk["files"])
                  and old["vendor_sha256"] == sdk["manifest_sha256"], "unmatched paired build")
            for r in (old,new):
                check(r["public_symbol"] == r["trace_absent"] == r["tests_passed"] == "1"
                      and r["worker_sha256"] == hosts[arch]["worker_binaries"][r["implementation"]]["sha256"]
                      and r["worker_sha256"] != r["trace_worker_sha256"], "production worker identity")
            for c in panel:
                row = proofs[arch,c["family"],provider]
                check(row["passed"] == "1" and int(row["elements"]) == int(c["m"])*int(c["n"])
                      and re.fullmatch(r"[a-f0-9]{64}",row["pristine_sha256"])
                      and row["pristine_sha256"] == row["disabled_sha256"], "pristine/disabled output mismatch")


def replay(root):
    panel = read(root/"panel.csv")
    hosts = json.loads((root/"provenance.json").read_text())["hosts"]
    verify_linked_builds(root,hosts,panel)
    methods = read(root/"methods.csv")
    coverage = read(root/"coverage.csv")
    eligible = {(r["arch"],r["family"],r["implementation"]): r for r in coverage}
    check(len(eligible) == len(coverage), "duplicate coverage")
    routes = {}
    for file in ("routes.csv", "fastmmw-routes.csv"):
        for r in read(root/file):
            key = r["arch"],r["family"],r["implementation"]
            check(key not in routes, (key,"duplicate route"))
            routes[key] = r
    times, accuracy = defaultdict(dict), defaultdict(dict)
    for row in read(root/"timings.csv"):
        key = row["arch"], row["family"], row["implementation"]
        pair = int(row["block"]), int(row["round"])
        t = float(row["seconds"])
        check(key in eligible and pair in ROUNDS and pair not in times[key] and
              int(row["seed"]) == SEEDS[pair[0]-1] and row["threads"] == row["calls"] == "1"
              and row["cpu"] == str(hosts[key[0]]["cpu"]) and math.isfinite(t) and t > 0, ("timing",key,pair))
        times[key][pair] = t
    for row in read(root/"accuracy.csv"):
        key = row["arch"], row["family"], row["implementation"]
        fixture = row["mode"],int(row["block"]),int(row["seed"])
        check(key in eligible and fixture in FIXTURES and fixture not in accuracy[key] and row["threads"] == "1"
              and row["cpu"] == str(hosts[key[0]]["cpu"]), ("accuracy",key,fixture))
        error, low, high = (float(row[k]) for k in ("max_abs_error","signed_min_error","signed_max_error"))
        check(math.isfinite(error) and low <= high and math.isclose(max(abs(low),abs(high)),error,rel_tol=3e-10)
              and math.isfinite(float(row["checksum"])) and 0 < float(row["first_sgemm_seconds"]) < math.inf,
              ("invalid signed residuals or first call",key,fixture))
        accuracy[key][fixture] = row
    case_summary, method_summary, pairs, decisions = [], [], [], []
    for arch in ARCHES:
        host = hosts[arch]
        roles = {r["implementation"]: r["role"] for r in methods if r["arch"] == arch and r["status"] == "included"}
        proofs = {n for n,r in roles.items() if r in ("original","signed","external_winograd")}
        providers = ("openblas", "mkl" if arch == "x86_64" else "armpl")
        gate = defaultdict(list)
        check({k for k in eligible if k[0] == arch} == {(arch,c["family"],n) for c in panel for n in roles},
              "missing case inventory")
        check({k for k in routes if k[0] == arch} == {(arch,c["family"],n) for c in panel for n in proofs},
              "missing route inventory")
        for c in panel:
            f = c["family"]
            m,n,k,pad = (int(c[d]) for d in ("m","n","k","pad"))
            warm, cold = {}, {}
            for method,role in roles.items():
                key = (arch,f,method)
                t, a, q = times[key], accuracy[key], eligible[key]
                check(set(a) == FIXTURES, (key,"missing fixture"))
                passing = set()
                for token, row in a.items():
                    good = row["nonfinite"] == row["guard_changes"] == row["input_changes"] == "0"
                    good &= 0 <= float(row["max_abs_error"]) <= .002 and int(row["checked"]) == m*n
                    if method.startswith("onednn-"):
                        good &= bool(row["onednn_impl"])
                    check((row["verified"] == "1") == good, (key,"invalid admission bit"))
                    if good:
                        passing.add(token)
                route_ok = method not in proofs or routes[key]["passed"] == "1"
                if role in ("signed","original") and route_ok:
                    r = routes[key]
                    sign = role == "signed"
                    check(r["cpu"] == str(host["cpu"]) and r["threads"] == r["baseline_bitwise_equal"] == "1" and
                          int(r["linked_checked"]) == int(r["trace_checked"]) == m*n and
                          (int(r["core_m"]),int(r["core_n"]),int(r["core_k"])) == (m-m%64,n-n%64,k-k%32) and
                          int(r["core_calls"]) == 49 and int(r["core_overwrites"]) == (35 if sign else 49) and
                          int(r["positive_updates"]) == int(r["negative_updates"]) == (7 if sign else 0) and
                          all(r[d]=="1" for d in ("quadrants_ok","edges_ok","provider_ok","inputs_unchanged","guards_ok")) and
                          int(r["calls"]) == 49 + int(m%64!=0) + int(n%64!=0) + int(k%32!=0),
                          (key,"wrong linked route"))
                if role == "external_winograd" and route_ok:
                    r = routes[key]
                    splits = 1
                    cutoff = int(method.rsplit("-",1)[1])
                    while (min(m,n,k)+splits-1)//splits > cutoff:
                        splits *= 2
                    check(int(r["leaf_calls"]) == 7**(splits.bit_length()-1) and r["leaf"] == "openblas" and
                          int(r["padded"]) == int(any(d%splits for d in (m,n,k))) and int(r["output_elements"]) == m*n and
                          all(r[d]=="1" for d in ("timed_binary_exact","trace_binary_exact","timed_inputs_unchanged","trace_inputs_unchanged")),
                          (key,"wrong external Winograd route"))
                ready = route_ok and passing == FIXTURES and set(t) == set(ROUNDS)
                check(ready == (q["qualified"] == "1") and int(q["warm_pairs"]) == len(t) and
                      int(q["verified_cases"]) == len(passing), (key,"qualification differs"))
                first = [1000*float(a["random",i,seed]["first_sgemm_seconds"]) for i,seed in enumerate(SEEDS,1)]
                if ready:
                    warm[method] = 1000*median(t.values())
                    cold[method] = median(first)
                method_summary.append(dict(arch=arch,family=f,implementation=method,role=role,qualified=int(ready),
                         warm_samples=len(t),warm_median_ms=fmt(warm[method]) if ready else "",
                         first_seed1_ms=fmt(first[0]),first_seed2_ms=fmt(first[1])))
            direct = {n for n in warm if roles[n] == "direct"}
            independent = direct | {n for n in warm if roles[n] == "external_winograd"}
            groups = dict(direct_oneshot=direct-{CACHED}, direct_all=direct, independent_oneshot=independent-{CACHED},
                          independent_all=independent, whole_field=set(warm), whole_field_oneshot=set(warm)-{CACHED})
            row = dict(arch=arch,family=f,m=m,n=n,k=k,pad=pad,qualified_methods=len(warm),measured_methods=len(roles))
            for label, pool in groups.items():
                winner = min(pool, key=lambda n:(warm[n],n))
                first_winner = min(pool, key=lambda n:(cold[n],n))
                row.update({label:winner, label+"_ms":fmt(warm[winner]), label+"_first":first_winner,
                           label+"_first_ms":fmt(cold[first_winner])})
            case_summary.append(row)
            for provider in providers:
                candidate = "signed-"+provider
                check(candidate in warm and "original-"+provider in warm and provider in direct,
                      (arch,f,"missing required linked or native method"))
                for peer in sorted(set(warm)-{candidate}):
                    ratios = [times[arch,f,peer][z]/times[arch,f,candidate][z] for z in ROUNDS]
                    ratio, wins = median(ratios), sum(r>1 for r in ratios)
                    pairs.append(dict(arch=arch,family=f,candidate=candidate,peer=peer,candidate_ms=fmt(warm[candidate]),
                         peer_ms=fmt(warm[peer]),peer_over_candidate=fmt(ratio),candidate_paired_wins=wins,pairs=6,
                         seed1_median=fmt(median(ratios[:3])),seed2_median=fmt(median(ratios[3:]))))
                    if f in ODD and peer == "original-"+provider:
                        gate[provider].append((ratio,wins))
        for provider in providers:
            items = gate[provider]
            check(len(items) == 2, (arch,provider,"missing fresh odd layout"))
            passing = sum(w >= 5 and r > 1.02 for r,w in items)
            target = provider == "mkl" and bool(re.search(r"\bIntel(?:\(R\))?\s+Xeon(?:\(R\))?\s+6975P-C\b",host["model"],re.I))
            decisions.append(dict(arch=arch,provider=provider,intended_target=int(target),odd_followups=len(items),
                 odd_passing=passing,odd_geomean=fmt(geometric_mean(r for r,w in items)),
                 reaches_numeric_threshold=int(passing==2),preregistered_confirmation=int(target and passing==2)))
    return {"summary.csv":case_summary,"method-summary.csv":method_summary,"pairs.csv":pairs,"decision.csv":decisions}


def verify(root):
    listed = {name:d for d,name in (r.split("  ",1) for r in (root/"SHA256SUMS").read_text().splitlines())}
    check(set(listed) == {p.name for p in root.iterdir() if p.is_file() and p.name!="SHA256SUMS"}, "file inventory")
    for name,d in listed.items():
        check(Path(name).name == name and sha(root/name) == d, name)
    for name,rows in replay(root).items():
        check((root/name).read_text() == csv_text(rows), name)
    print("Verified published file hashes, per-layout eligibility, raw and paired timings, winners and target decisions.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
