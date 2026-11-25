"""Recalculate the one-core stream-B comparison from raw CSV and source/build receipts."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median

PHASE = "stream-b"
REVISION = "b218beff1eaf8fd606d0934405cdc1e28ecdfe15"
PANEL = "466c2b51ee6df5fd737f3f4b1977c5b4ad75cdb7ae61bbaec7bc5919530f18fd"
PROTOCOL = "694e4ab5b18602ff460209d13e0ef827e7442278d33100f661ced75dec299eb8"
PATCH = "3fd0a13fc41d9f04441fb2f58981f49354e51c2856e2512972f3883778420d43"
CHECK = "c053e534b4759042eb0dd334df0abb85219a8e0a510d4bfd5c71b2054c71b9c3"
INVENTORIES = {
    "original": ("ce8377fd1574187615e356cf59425127deacd2d532bce602c41ac932e4dff440",
                 "2ded86410ab416d51508bbb28fe6c494f3b1138c910992d01badf3ac7916e5c9"),
    "candidate": ("9acc944c3b022ef22611a9e38c52e97c8feea49614597683505e2ab60d9d5646",
                  "3652f18c193dcd3ea6099dd51e011ffc4ccd9a0d4832934bbb889c17e9e99138"),
}
CONFIG = {
    "native": ("stream-incumbent", "stream-native"),
    "avx2": ("stream-avx2-incumbent", "stream-avx2"),
    "avx512": ("stream-avx512-incumbent", "stream-avx512"),
}
PEERS = {"openblas", "mkl", "mkl-jit", "aocl", "blasfeo", "onednn-oneshot", "onednn-cached-plan"}
FIELDS = ("m", "n", "k", "pad")
SEEDS = (348271759, 710438957)
CHECKS = {("random", 1, SEEDS[0]), ("random", 2, SEEDS[1]),
          ("cancel", 0, SEEDS[0]), ("scaled", 0, SEEDS[0])}
PAIRS = [(b, r) for b in (1, 2) for r in (1, 2, 3)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rows(root, name):
    with (root / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distinct(records, key):
    output = {}
    for record in records:
        token = key(record)
        require(token not in output, f"Duplicate record {token}")
        output[token] = record
    return output


def positive(raw):
    x = float(raw)
    require(math.isfinite(x) and x > 0, f"Nonpositive or nonfinite duration {raw}")
    return x


def check_sources(root):
    spec = json.loads((root / "preregistration.json").read_bytes())
    require(spec["phase"] == PHASE and spec["seeds"] == list(SEEDS) and spec["production_revision"] == REVISION, "registration")
    expected = {"panel.csv": PANEL, "protocol.md": PROTOCOL, "check.c": CHECK, "stream-b.patch": PATCH}
    require(spec["files"] == expected, "registration changed")
    for name, sha in expected.items():
        path = root / name
        if name in ("panel.csv", "protocol.md") or path.exists():
            require(digest(path.read_bytes()) == sha, f"Frozen file changed: {name}")
    sources = json.loads((root / "sources.json").read_bytes())
    require((sources["base"], sources["patch"]) == (REVISION, PATCH), "source/patch identity")
    for variant, (sha, _) in INVENTORIES.items():
        data = json.dumps(sources[variant], separators=(",", ":"), sort_keys=True).encode()
        require(digest(data) == sha, f"Complete {variant} inventory differs from public source")
    return sources


def check_builds(root, ranking, sources, inventory):
    entries = ranking["implementations"]
    included = {n for n, r in inventory.items() if r["status"] == "included"}
    standalone = {n for pair in CONFIG.values() for n in pair}
    require(set(inventory) == PEERS | standalone and set(entries) == included, "method inventory")
    required = (PEERS - {"aocl"}) | set(CONFIG["native"]) | set(CONFIG["avx2"])
    require(required <= included, "required native peer or ISA missing")
    hashes = {}
    for line in (root / "stream-binaries.sha256").read_text().splitlines():
        sha, entry = line.split(maxsplit=1)
        name = Path(entry.lstrip("*")).name
        require(re.fullmatch("[0-9a-f]{64}", sha) and name not in hashes, "binary receipt")
        hashes[name] = sha
    require(set(hashes) == {Path(p).name for _, p in entries.values()}, "binary receipt coverage")
    for label, (abi, path) in entries.items():
        direct = label in PEERS
        require(abi == ("direct" if direct else "linked-public"), f"ABI {label}")
        require(inventory[label]["role"] == ("direct" if direct else "standalone"), f"role {label}")
        basename = label.replace("onednn-", "onednn-f32-") + ".so" if direct else label + ".worker"
        require(Path(path).name == basename, f"binary label {label}")
    builds = distinct(rows(root, "builds.csv"), lambda r: r["implementation"])
    require(set(builds) == standalone, "build inventory")
    flags = (root / "native-features.txt").read_text().strip().split()
    require(flags in (["1", "1"], ["1", "0"]), "runtime CPU feature probe")
    protected = {}
    for target, pair in CONFIG.items():
        enabled = target != "avx512" or flags[1] == "1"
        isa = builds[pair[1]]["isa"] if target == "native" else target
        if enabled:
            require(isa == "avx2" or (isa == "avx512" and flags[1] == "1"), f"CPU/OS feature mismatch {target}")
        for n, variant in zip(pair, ("original", "candidate")):
            b = builds[n]
            if not enabled:
                require(b["status"] == "unsupported" and n not in included, f"ungated ISA {n}")
                continue
            require(n in included and b["isa"] == isa and b["status"] == "qualified", f"ISA {n}")
            candidate = variant == "candidate"
            require(b["variant"] == ("candidate" if candidate else "incumbent"), f"variant {n}")
            require(b["source_sha256"] == sources[variant]["src/gemm.c"] and
                    b["trace_source_sha256"] == INVENTORIES[variant][1], f"compiled source {n}")
            require(b["worker_sha256"] == hashes[n + ".worker"] and b["worker_sha256"] != b["trace_worker_sha256"] and
                    all(re.fullmatch("[0-9a-f]{64}", b[k]) for k in ("trace_worker_sha256", "object_sha256")), f"object/worker receipt {n}")
            common = "-O3 -std=c11 -fPIC -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_X86_MEDIUM_ROUTE=1 -DCOB_ENABLE_BLOCKED_LEAF=1 -DCOB_ENABLE_DIRECT_SMALL_GEMM=0 -DCOB_ENABLE_SHORT_AXES=1 -DCOB_ENABLE_FUSED_STRASSEN=1 -DCOB_ENABLE_SMALL_X86=1 -DCOB_ENABLE_FEW_ROW_PANEL=1 -march=native".split()
            extra = {"native": [], "avx2": ["-mno-avx512f", "-mprefer-vector-width=256", "-DCOB_DISABLE_X86_AVX512=1"], "avx512": ["-mavx512f", "-mprefer-vector-width=512"]}[target]
            require(b["flags"].split() == common + extra + (["-DCOB_ENABLE_X86_STREAM_B=1"] if candidate else []), f"build flags {n}")
            if candidate:
                text = (root / f"{n}-fixtures.txt").read_text().strip()
                proof = re.fullmatch(r"PASS tile=(\d+)x(\d+) checked=(\d+) max_abs_error=(\S+)", text)
                require(proof, f"guard-page and fallback fixture receipt {n}")
                mr, nr, cells = map(int, proof.groups()[:3])
                worst = float(proof[4])
                require((mr, nr) == ((6, 64) if isa == "avx512" else (3, 32)) and cells == 435666 and
                        0 <= worst <= 0.002, f"guard-page fixture result {n}")
                protected[n] = {"cells": cells, "max_error": worst}
    return builds, hashes, protected


def keyed(row):
    return row["family"], row["implementation"]


def dimensions(row):
    return tuple(int(row[p]) for p in FIELDS)


def check_routes(root, builds, names, panel):
    records = distinct(rows(root, "routes.csv"), keyed)
    pairs = {(f, n) for f in panel for n in names if n not in PEERS}
    require(set(records) == pairs, "ordinary and instrumented route matrix")
    cpus = set()
    total = 0
    for (family, name), r in records.items():
        m, n, k, pad = panel[family]
        b = builds[name]
        hit = b["variant"] == "candidate" and 24 <= m <= 160 and 1024 <= n <= 4096 and 96 <= k <= 384 and n >= 8*m and m*k <= 49152
        mr, nr = ((6, 64) if b["isa"] == "avx512" else (3, 32)) if hit else (0, 0)
        packs = math.ceil(n / nr) if hit else 0
        tiles = packs * math.ceil(m / mr) if hit else 0
        old = not hit and m >= 64 and k >= 128
        fallback = not hit and not old
        require(dimensions(r) == (m, n, k, pad) and r["isa"] == b["isa"] and r["passed"] == "1" and not r["error"], f"route {family} {name}")
        counts = tuple(int(r[c]) for c in ("selected", "mr", "nr", "packs", "tiles", "old_medium", "blocked", "expected_old_medium", "expected_blocked"))
        require(counts == (hit, mr, nr, packs, tiles, old, fallback, old, fallback), f"dispatch counters {family} {name}")
        require(int(r["linked_checked"]) == int(r["trace_checked"]) == m*n and r["inputs_unchanged"] == r["guards"] == r["threads"] == "1", f"exact public entry {family} {name}")
        cpus.add(int(r["cpu"]))
        total += 2*m*n
    return cpus, total, len(records)


def audit(root):
    sources = check_sources(root)
    ranking = json.loads((root / "ranking.json").read_bytes())
    require((ranking["phase"], ranking["panel_sha256"], ranking["seeds"], ranking["rounds_per_seed"], ranking["full_output_tolerance"]) ==
            (PHASE, PANEL, list(SEEDS), 3, 0.002), "phase protocol")
    host = dict(line.split(":", 1) for line in (root / "host.txt").read_text().splitlines() if ":" in line)
    model = host["Model name"].strip()
    require(host["Architecture"].strip() == "x86_64" and model, "host identity")
    panel = {r["family"]: dimensions(r) for r in rows(root, "panel.csv")}
    names = list(ranking["implementations"])
    inventory = distinct(rows(root, "eligibility.csv"), lambda r: r["implementation"])
    builds, binaries, protected = check_builds(root, ranking, sources, inventory)
    cpus, exact_cells, route_count = check_routes(root, builds, names, panel)
    expected = {(f, n) for f in panel for n in names}
    admissions = distinct(rows(root, "case-eligibility.csv"), keyed)
    require(set(admissions) == expected, "case eligibility matrix")
    acc = rows(root, "accuracy.csv")
    warm = rows(root, "timings.csv")
    first = distinct(rows(root, "first-calls.csv"), lambda r: (keyed(r), r["mode"], int(r["block"]), int(r["seed"])))
    checks, times, maxima = defaultdict(dict), defaultdict(dict), defaultdict(float)
    engines, rejected = defaultdict(set), []
    first_by_case = defaultdict(list)
    cells = 0
    for r in acc:
        key = f, n = keyed(r)
        proof = r["mode"], int(r["block"]), int(r["seed"])
        m, w, k, pad = panel[f]
        require(key in expected and r["phase"] == PHASE and dimensions(r) == (m, w, k, pad) and proof in CHECKS and proof not in checks[key], f"FP64 fixture identity {key}")
        d = dict(x.split("=", 1) for x in r["detail"].split(",") if "=" in x)
        require(r["cpu"].isdigit() and r["cpu"] == d["cpu"] and r["threads"] == d["threads"] == "1" and
                int(r["checked"]) == m*w and r["nonfinite"] == r["guard_changes"] == d["input_changes"] == "0", f"FP64 fixture integrity {key}")
        error = float(r["max_abs_error"])
        signed = [float(d["signed_min_error"]), float(d["signed_max_error"])]
        require(math.isfinite(error) and error >= 0 and signed[0] <= signed[1] and math.isclose(max(map(abs, signed)), error, rel_tol=1e-10, abs_tol=1e-14), f"signed error range {key}")
        verified = error <= 0.002
        require(r["verified"] == str(int(verified)) and int(r["stress"]) == (proof[0] != "random"), f"numerical acceptance {key}")
        first_row = first.pop((key, *proof))
        engine = d.get("onednn_impl", "")
        t0 = positive(d["first_sgemm_seconds"])
        require(dimensions(first_row) == panel[f] and first_row["verified"] == r["verified"] and first_row["onednn_impl"] == engine and positive(first_row["first_sgemm_seconds"]) == t0 and bool(engine) == n.startswith("onednn-"), f"first call/oneDNN record {key}")
        if engine:
            engines[f, n].add(engine)
        if proof[0] == "random":
            first_by_case[key].append(t0)
        if not verified:
            rejected.append({"family": f, "method": n, "mode": proof[0], "seed": proof[2], "max_error": error})
        maxima[n] = max(maxima[n], error)
        checks[key][proof] = verified
        cpus.add(int(r["cpu"]))
        cells += m*w
    require(not first, "orphan first-call records")
    counts = defaultdict(set)
    for r in warm:
        key = f, n = keyed(r)
        m, w, k, pad = panel[f]
        pair = b, j = int(r["block"]), int(r["round"])
        require(key in expected and r["phase"] == PHASE and r["mode"] == "random" and r["stress"] == "0" and pair in PAIRS and
                dimensions(r) == (m, w, k, pad) and int(r["seed"]) == SEEDS[b-1] and pair not in times[key], f"warm-pair identity {key}")
        i = (list(panel).index(f) + j - 1) % len(names)
        order = names[i:] + names[:i]
        if (j + b) % 2:
            order.reverse()
        require(r["threads"] == "1" and r["cpu"].isdigit() and r["resident"] in ("1", "0") and int(r["position"]) == order.index(n)+1, f"counterbalance or thread record {key} {pair}")
        budget = min(512, max(32, math.ceil(8e7/(2*m*w*k))))
        require(int(r["calls"]) == budget, f"mismatched batch {key} {pair}")
        counts[f].add(budget)
        cpus.add(int(r["cpu"]))
        times[key][pair] = positive(r["seconds"])
    require(len(cpus) == 1 and min(cpus) >= 0, "workers spanned logical CPUs")
    qualified = set()
    for key, row in admissions.items():
        require(dimensions(row) == panel[key[0]] and set(checks[key]) == CHECKS, f"four full FP64 fixtures {key}")
        for (b, j) in times[key]:
            require(checks[key]["random", b, SEEDS[b-1]], f"timed failed random fixture {key}")
        ok = all(checks[key].values()) and set(times[key]) == set(PAIRS)
        reason = "all_checks_passed" if ok else ("incomplete_or_failed_full_fp64" if not all(checks[key].values()) else "incomplete_paired_timings")
        require((int(row["verified_cases"]), int(row["warm_pairs"]), int(row["qualified"]), row["reason"]) ==
                (sum(checks[key].values()), len(times[key]), int(ok), reason), f"case admission {key}")
        if ok:
            qualified.add(key)
    def comparison(f, c, p):
        ratios = [times[f, p][j] / times[f, c][j] for j in PAIRS]
        return {"method": p, "time_ms": 1000*median(times[f, p].values()), "speedup": median(ratios), "wins": sum(x > 1 for x in ratios),
                "seed_medians": [median(ratios[:3]), median(ratios[3:])], "ratios": ratios}
    results, gates = {}, {}
    for target, (old, new) in CONFIG.items():
        if new not in names:
            continue
        holdouts = []
        results[target] = {}
        for f in panel:
            require({(f, old), (f, new)} <= qualified, f"candidate or incumbent excluded on {f}")
            direct = {n for n in PEERS if (f, n) in qualified}
            require(direct - {"onednn-cached-plan"}, f"no qualified single-call independent on {f}")
            by_time = lambda n: (median(times[f, n].values()), n)
            fastest = min(direct - {"onednn-cached-plan"}, key=by_time)
            any_plan = min(direct, key=by_time)
            prior = comparison(f, new, old)
            results[target][f] = dict(candidate_ms=1000*median(times[f, new].values()), incumbent=prior,
                best_oneshot=comparison(f, new, fastest), best_any=comparison(f, new, any_plan),
                external={n:comparison(f, new, n) for n in sorted(direct)},
                random_first_median_ms={n:1000*median(first_by_case[f, n]) for n in sorted(direct|{old, new})},
                onednn_implementations={n:sorted(engines[f, n]) for n in sorted(direct) if n.startswith("onednn-")})
            if f.startswith("fresh-"):
                holdouts.append(prior)
        vals = [row["speedup"] for row in holdouts]
        by_seed = [geometric_mean(row["seed_medians"][i] for row in holdouts) for i in (0, 1)]
        gm = geometric_mean(vals)
        wins = sum(x > 1 for x in vals)
        require(len(vals) == 5, "five registered holdouts")
        gates[target] = {"geometric_mean": gm, "seed_geometric_means": by_seed, "shape_wins": wins, "minimum_paired_speedup": min(vals),
                         "passed": gm >= 1.03 and min(by_seed) > 1 and wins >= 4 and min(vals) >= 0.98}
    return {"phase": PHASE, "host": model, "logical_cpu": next(iter(cpus)), "release": REVISION,
            "source_patch_sha256": PATCH, "cases": results, "gates": gates, "max_errors": dict(maxima), "numerical_failures": rejected,
            "coverage": {"methods": names, "fp64_fixtures": len(acc), "fp64_cells": cells, "warm_batches": len(warm),
                         "total_timed_gemms": sum(int(r["calls"]) for r in warm), "exact_route_cases": route_count,
                         "exact_route_cells": exact_cells, "eligible_cases": len(qualified), "expected_cases": len(expected), "first_calls": len(acc),
                         "batch_size_by_layout": {f:sorted(v) for f,v in counts.items()}, "protected_fixtures": protected},
            "binaries": binaries, "omissions": [r for r in inventory.values() if r["status"] != "included"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("evidence", type=Path)
    print(json.dumps(audit(cli.parse_args().evidence), indent=2, allow_nan=False))
