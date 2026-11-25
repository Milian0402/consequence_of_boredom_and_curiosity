"""Score qualified signed workers against their matched originals, direct rivals and external Winograd."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

SEEDS = (31048579, 196173023)
ROUNDS = [(seed, r) for seed in SEEDS for r in (1, 2, 3)]
FIXTURES = {("random", s) for s in SEEDS} | {("cancel", SEEDS[0]), ("scaled", SEEDS[0])}
FRESH_ODD = {"deep-odd-followup", "wide-odd-followup"}
CACHED = "onednn-cached-plan"


def read(root, name):
    with (root / name).open(newline="") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(candidate, reference):
    ratios = [reference[key] / candidate[key] for key in ROUNDS]
    return dict(pairs=6, paired_wins=sum(v > 1 for v in ratios), median_speedup=median(ratios),
                seed1_median=median(ratios[:3]), seed2_median=median(ratios[3:]))


def score(root, native, study_status, pinned_cpu):
    inventory = read(root, "eligibility.csv")
    roles = {r["implementation"]: r["role"] for r in inventory if r["status"] == "included"}
    providers = ("openblas", native)
    required = set(providers) | {f"{prefix}-{provider}" for prefix in ("original", "signed") for provider in providers}
    require(required.issubset(roles), "missing required direct or matched release worker")
    panel = read(root, "panel.csv")
    families = [r["family"] for r in panel]
    require(len(set(families)) == len(panel) == 5 and FRESH_ODD.issubset(families), "incomplete panel")
    ranking = json.loads((root / "ranking.json").read_text())
    require(ranking["phase"] == "signed-leaves" and ranking["seeds"] == list(SEEDS)
            and ranking["rounds_per_seed"] == 3 and ranking["completed_cases"] == len(panel)
            and ranking["expected_cases"] == len(panel) and set(ranking["implementations"]) == set(roles),
            "incomplete measurement run")
    coverage = read(root, "case-eligibility.csv")
    eligible = {(r["family"],r["implementation"]):r["qualified"] == "1" for r in coverage}
    expected = {(f,n) for f in families for n in roles}
    require(set(eligible) == expected and len(eligible) == len(coverage), "incomplete eligibility inventory")
    geometry = {tuple(r[d] for d in ("m","n","k","pad")): r["family"] for r in panel}
    route_methods = {n for n in roles if roles[n] in ("original","signed","external_winograd")}
    routes = {}
    for filename in ("routes.csv", "fastmmw-routes.csv"):
        for r in read(root, filename):
            key = geometry[tuple(r[d] for d in ("m","n","k","pad"))], r["implementation"]
            require(key not in routes and r["passed"] in ("0","1"), f"malformed route record {key}")
            routes[key] = r["passed"] == "1"
    require(set(routes) == {(f,n) for f in families for n in route_methods}, "missing or extra applicable route records")
    for family in families:
        for method in required:
            require(eligible[family,method], f"required method failed qualification: {family} {method}")

    timings = {key:{} for key in expected}
    for row in read(root, "timings.csv"):
        key = row["family"], row["implementation"]
        pairing = int(row["seed"]), int(row["round"])
        value = float(row["seconds"])
        require(key in expected and pairing in ROUNDS and pairing not in timings[key]
                and row["cpu"] == pinned_cpu and row["threads"] == "1"
                and int(row["calls"]) == 1 and math.isfinite(value) and value > 0,
                f"invalid warm measurement {key} {pairing}")
        timings[key][pairing] = value
    checks = {key:{} for key in expected}
    optional_failures = set()
    for row in read(root, "accuracy.csv"):
        key = row["family"], row["implementation"]
        token = row["mode"], int(row["seed"])
        require(key in expected and token in FIXTURES and token not in checks[key], f"duplicate or unknown accuracy fixture {key} {token}")
        detail = dict(x.split("=", 1) for x in row["detail"].split(",") if "=" in x)
        error = float(row["max_abs_error"])
        require(row["cpu"] == pinned_cpu and row["threads"] == "1" and
                int(row["checked"]) == int(row["m"]) * int(row["n"]) and
                row["guard_changes"] == row["nonfinite"] == detail.get("input_changes") == "0"
                and math.isfinite(error) and error >= 0 and
                math.isfinite(float(detail["first_sgemm_seconds"])) and float(detail["first_sgemm_seconds"]) > 0,
                f"malformed or unsafe completed fixture {key} {token}")
        checks[key][token] = row
        if row["verified"] == "0" and error > 0.002 and key[1] not in required and not eligible[key]:
            optional_failures.add((*key, *token))
        else:
            require(row["verified"] == "1" and error <= .002, f"unexpected validation failure {key} {token}")

    # The coordinator deliberately retains all failures, even numerical losses by optional methods.
    # Accept exactly the raw, finite limit breaches above. A lost process or file remains an error.
    reported = set()
    for issue in ranking["errors"]:
        token = (issue["family"], issue.get("implementation"), issue["mode"], int(issue.get("seed", -1)))
        require(token in optional_failures and token not in reported and
                "failed full-output validation" in issue.get("error",""), f"unexplained coordinator failure {issue}")
        reported.add(token)
    require(reported == optional_failures and bool(study_status) == bool(reported),
            "coordinator exit status does not match the recorded optional numerical results")
    for key in expected:
        require(set(checks[key]) == FIXTURES, f"missing completed startup metrics {key}")
        expected_rounds = {r for r in ROUNDS if (*key, "random", r[0]) not in optional_failures}
        require(set(timings[key]) == expected_rounds, f"unexplained missing warm measurements {key}")
        should_qualify = routes.get(key,True) and all(row["verified"]=="1" for row in checks[key].values())
        require(eligible[key] == should_qualify, f"qualifications disagree with raw fixture and route evidence {key}")
    output = []
    decisions = []
    for provider in providers:
        marks = []
        candidate, original = "signed-" + provider, "original-" + provider
        for family in families:
            available = {n:timings[family,n] for n in roles if eligible[family,n]}
            elapsed = {n:median(t.values()) for n,t in available.items()}
            def fastest(names):
                return min(names, key=lambda n:(elapsed[n],n))
            direct = [n for n in available if roles[n] == "direct"]
            external = [n for n in available if roles[n] in ("direct","external_winograd")]
            targets = {"original": original, "matched_direct": provider,
                       "fastest_direct_oneshot": fastest(n for n in direct if n != CACHED),
                       "fastest_direct_any": fastest(direct),
                       "fastest_external_oneshot": fastest(n for n in external if n != CACHED),
                       "fastest_external_any": fastest(external)}
            if native == "mkl" and "mkl-jit" in available:
                targets["direct_mkl_jit"] = "mkl-jit"
            for group, peer in targets.items():
                comparison = paired(available[candidate], available[peer])
                output.append(dict(family=family, implementation=candidate, group=group, peer=peer,
                                   candidate_seconds=elapsed[candidate], peer_seconds=elapsed[peer], **comparison))
                if family in FRESH_ODD and group == "original":
                    marks.append(comparison["paired_wins"] >= 5 and comparison["median_speedup"] > 1.02)
        require(len(marks) == len(FRESH_ODD), "incorrect gate population")
        decisions.append(dict(implementation=candidate, matched_original=original, odd_followups=len(marks),
                              odd_followups_passing=sum(marks), passes_numerical_speed_criterion=all(marks)))
    with (root / "signed-comparisons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    model = next(line.split(":",1)[1].strip() for line in (root/"host.txt").read_text().splitlines()
                 if line.startswith("Model name:"))
    report = dict(seeds=SEEDS, cpu=int(pinned_cpu), model=model, native_provider=native,
                  matches_preregistered_intel_cpu="6975P-C" in model.upper(), decisions=decisions,
                  optional_accuracy_exclusions=[dict(family=f, implementation=n, mode=m, seed=s)
                                                for f,n,m,s in sorted(optional_failures)])
    (root / "signed-score.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("native", choices=("mkl", "armpl"))
    parser.add_argument("study_status", type=int)
    parser.add_argument("pinned_cpu")
    args = parser.parse_args()
    score(args.directory, args.native, args.study_status, args.pinned_cpu)
