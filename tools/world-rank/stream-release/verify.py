"""Independently qualify the production acceptance experiment and score paired SGEMMs."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median

PHASE = "stream-release"
RELEASE = "a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6"
RELEASE_TREE = "18f22622de1a4b1b7aebbdb66985741aa38c6e9e"
RELEASE_MANIFEST = "31c88bd81cea91d33400d4c7cbeadb41ebef749bc6db081e28af17bde1e583ef"
BASE = "b218beff1eaf8fd606d0934405cdc1e28ecdfe15"
RESEARCH = "b2cdc6a2985726cb1658dc7cc730b5a1c4b99680"
PANEL_SHA = "466c2b51ee6df5fd737f3f4b1977c5b4ad75cdb7ae61bbaec7bc5919530f18fd"
SEEDS = (348271759, 710438957)
FIXTURES = (("random", 1, SEEDS[0]), ("random", 2, SEEDS[1]),
            ("cancel", 0, SEEDS[0]), ("scaled", 0, SEEDS[0]))
PAIRS = [(b, r) for b in (1, 2) for r in (1, 2, 3)]
DIRECT = ("openblas", "mkl", "mkl-jit", "aocl")
PRODUCTS = ("cmake-default", "cmake-att", "cmake-intel", "make-default", "make-att", "make-intel",
            "cmake-native-default", "cmake-native-on")
METHODS = (*DIRECT, *PRODUCTS, "research-incumbent", "research-candidate")
MATCHUPS = {
    "cmake-att": ("cmake-default", "research-candidate", *DIRECT),
    "cmake-native-on": ("cmake-native-default", "research-candidate", "research-incumbent", *DIRECT),
    "cmake-intel": ("cmake-att",), "make-att": ("make-default", "cmake-att"),
    "make-intel": ("make-att",), "research-candidate": ("research-incumbent",),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(root, name):
    with (root / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows, key):
    result = {}
    for row in rows:
        token = key(row)
        require(token not in result, f"Duplicate record {token}")
        result[token] = row
    return result


def positive(raw):
    value = float(raw)
    require(math.isfinite(value) and value > 0, f"Invalid duration {raw}")
    return value


def dimensions(row):
    return tuple(int(row[k]) for k in ("m", "n", "k", "pad"))


def case(row):
    return row["family"], row["implementation"]


def fixture(row):
    return row["mode"], int(row["block"]), int(row["seed"])


def properties(text, delimiter):
    return dict(part.split("=", 1) for part in text.split(delimiter) if "=" in part)


def check_identities(root, sources, builds, ranking):
    require((root / "release-revision.txt").read_text().strip() == sources["release"] == RELEASE and
            sources["release_tree"] == RELEASE_TREE and sources["source_manifest_sha256"] == RELEASE_MANIFEST,
            "Release checkout or production source manifest differs from the registered Git tree")
    for name, field in (("protocol.md", "protocol_sha256"), ("production-files.sha256", "source_manifest_sha256")):
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == sources[field], "Unbound "+name)
    files = {}
    for line in (root/"production-files.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        require(re.fullmatch("[0-9a-f]{64}", digest) and path not in files, "Duplicate or malformed production file")
        files[path] = digest
    required_files = {"CMakeLists.txt", "Makefile", "include/cob_gemm.h", "src/gemm.c",
                      "src/cob_stream_b_dispatch.inc", "src/cob_stream_b_kernel.c", "tests/test_stream_b.c"}
    require(required_files <= files.keys(), "Missing production source or dispatcher")
    links = index(table(root, "linked-products.csv"), lambda r: r["implementation"])
    require(set(links) == set(builds), "Complete static-link receipts")
    for n, b in builds.items():
        r = links[n]
        require(r["header_sha256"] == files["include/cob_gemm.h"] and
                r["build_archive_sha256"] == r["linked_archive_sha256"] == b["archive_sha256"] and
                r["worker_sha256"] == b["worker_sha256"] and r["public_symbol"] == "cob_sgemm_rowmajor" and
                re.fullmatch("[0-9a-f]{64}", r["trace_sha256"]) and r["trace_sha256"] != b["worker_sha256"] and
                int(r["trace_wrappers"]) == (2 if b["option"] == "ON" else 0), "Archive or trace linkage mismatch for "+n)
    frozen = json.loads((root/"research/sources.json").read_text())
    require(frozen["base"] == BASE and frozen["patch"] == "3fd0a13fc41d9f04441fb2f58981f49354e51c2856e2512972f3883778420d43",
            "Frozen research source identity")
    for variant, expected in (("original", "ce8377fd1574187615e356cf59425127deacd2d532bce602c41ac932e4dff440"),
                              ("candidate", "9acc944c3b022ef22611a9e38c52e97c8feea49614597683505e2ab60d9d5646")):
        serialized = json.dumps(frozen[variant], separators=(",", ":"), sort_keys=True).encode()
        require(hashlib.sha256(serialized).hexdigest() == expected, "Frozen "+variant+" inventory")
    research_builds = index(table(root, "research/builds.csv"), lambda r: r["implementation"])
    timed = index(table(root, "timed-binaries.csv"), lambda r: r["implementation"])
    require(set(timed) == set(METHODS), "Timed binary inventory")
    for method, product in builds.items():
        if product["build_system"] != "cmake":
            continue
        flags = json.loads((root/(method+"-flags.json")).read_text())
        on = product["option"] == "ON"
        require(set(flags) == ({"public", "avx2", "avx512"} if on else {"public"}), "CMake compilation units for "+method)
        for unit, arguments in flags.items():
            targets = [v for v in arguments if v.startswith("-march=")]
            target = "-march=native" if unit == "public" and method.startswith("cmake-native-") else "-march=x86-64"
            require(targets and targets[-1] == target and "-masm="+product["dialect"] in arguments,
                    f"Compiler ISA/dialect mismatch {method}/{unit}")
            if unit == "public":
                require(("-DCOB_ENABLE_X86_STREAM_B=1" in arguments) == on and
                        not any(a.startswith(("-mavx", "-mfma")) for a in arguments), "Unsafe global ISA flag for "+method)
            else:
                require("-mfma" in arguments and ("-mavx2" if unit == "avx2" else "-mavx512f") in arguments,
                        f"Incorrect private compiler target {method}/{unit}")
    for name, row in timed.items():
        digest = row["sha256"]
        require(re.fullmatch("[0-9a-f]{64}", digest), "Missing timed binary digest for "+name)
        if name in builds:
            expected_file = name+".worker"
            require(digest == builds[name]["worker_sha256"], "Different product worker timed for "+name)
        elif name.startswith("research-"):
            original, variant = ("stream-native", "candidate") if name == "research-candidate" else ("stream-incumbent", "original")
            expected_file = original+".worker"
            b = research_builds[original]
            require(b["status"] == "qualified" and digest == b["worker_sha256"] and
                    b["source_sha256"] == frozen[variant]["src/gemm.c"], "Different frozen research worker timed for "+name)
        else:
            expected_file = name+".so"
        require(Path(ranking["implementations"][name][1]).name == expected_file, "Different executable path for "+name)
    return {n: r["sha256"] for n, r in timed.items()}


def check_builds(root, panel):
    features = (root / "native-features.txt").read_text().split()
    require(len(features) == 2 and features[0] == "1" and features[1] in ("0", "1"), "Expected AVX2/FMA capable x86")
    width = 64 if features[1] == "1" else 32
    builds = index(table(root, "product-builds.csv"), lambda r: r["implementation"])
    receipts = index([properties(line, " ") for line in (root / "generic-build-tests.txt").read_text().splitlines()
                      if line.strip()], lambda r: r["method"])
    require(set(builds) == set(receipts) == set(PRODUCTS), "Incomplete CMake/Make build receipts")
    enabled = {}
    for name, build in builds.items():
        enabled[name] = not name.endswith("default")
        on = enabled[name]
        option = "ON" if on else "OFF"
        dialect = "intel" if name.endswith("intel") else "att"
        system = "make-objects" if name.startswith("make-") else "cmake"
        require((build["build_system"], build["option"], build["dialect"], build["public_symbol"]) ==
                (system, option, dialect, "cob_sgemm_rowmajor"), f"Unexpected product type for {name}")
        require(all(re.fullmatch("[a-f0-9]{64}", build[field]) for field in ("worker_sha256", "archive_sha256")),
                f"Missing linked archive or worker identity for {name}")
        require(int(build["has_avx2"]) == int(build["has_avx512"]) == int(on), f"Unexpected kernel symbols for {name}")
        record = receipts[name]
        require((record["test"], record["option"], record["dialect"]) == ("pass", option, dialect),
                f"Failed build-system test for {name}")
        if on:
            error = float(record["max_abs_error"])
            require(int(record["native_width"]) == width and int(record["AVX2_width"]) == 32 and
                    int(record["checked"]) > 0 and 0 <= error <= 0.002, f"Dispatch/fallback fixture {name}")
    for default, on in (("cmake-default", "cmake-att"), ("make-default", "make-att"), ("cmake-native-default", "cmake-native-on")):
        require(all(builds[default][key] != builds[on][key] for key in ("archive_sha256", "worker_sha256")),
                f"Identical opt-in and opt-out products {on}")
    routes = index(table(root, "routes.csv"), case)
    require(set(routes) == {(f, n) for f in panel for n in PRODUCTS}, "Missing per-layout CMake/Make route")
    exact_cells = 0
    for (family, name), r in routes.items():
        m, n, _, _ = panel[family]
        selected = width if enabled[name] and family != "outside-width" else 0
        require(dimensions(r) == panel[family] and int(r["width"]) == selected and
                int(r["avx2"]) == int(selected == 32) and int(r["avx512"]) == int(selected == 64) and
                int(r["checked"]) == m*n and r["threads"] == r["passed"] == r["inputs_unchanged"] == r["guards"] == "1",
                f"Product public route and exact full outputs {name}, {family}")
        exact_cells += 2*m*n  # ordinary executable and companion linked to the same archive
    research = index(table(root, "research-routes.csv"), case)
    expected = {"stream-incumbent", "stream-native", "stream-avx2-incumbent", "stream-avx2"}
    if width == 64:
        expected.update(("stream-avx512-incumbent", "stream-avx512"))
    require(set(research) == {(f, n) for f in panel for n in expected}, "Research route coverage")
    cpus = set()
    for (family, name), r in research.items():
        m, n, k, _ = panel[family]
        on = "incumbent" not in name and family != "outside-width"
        old = not on and m >= 64 and k >= 128
        blocked = not on and not old
        nr = (32 if "avx2" in name else width) if on else 0
        mr = {0: 0, 32: 3, 64: 6}[nr]
        packs = (n+nr-1)//nr if on else 0
        require(dimensions(r) == panel[family] and int(r["selected"]) == on and
                (int(r["nr"]), int(r["mr"]), int(r["packs"]), int(r["tiles"])) ==
                (nr, mr, packs, packs*((m+mr-1)//mr) if on else 0) and
                int(r["old_medium"]) == int(r["expected_old_medium"]) == old and
                int(r["blocked"]) == int(r["expected_blocked"]) == blocked and
                int(r["linked_checked"]) == int(r["trace_checked"]) == m*n and
                r["passed"] == r["inputs_unchanged"] == r["guards"] == r["threads"] == "1" and r["cpu"].isdigit(),
                f"Research public route and full outputs {name}, {family}")
        cpus.add(int(r["cpu"]))
    return builds, width, len(routes), exact_cells, cpus


def verify(root, expected_release=None, expected_benchmark=None):
    sources = json.loads((root / "sources.json").read_text())
    for key in ("release", "benchmark"):
        require(re.fullmatch(r"[a-f0-9]{40}", sources[key]), f"Invalid {key} source")
    require(sources["base"] == BASE and sources["research"] == RESEARCH and sources["panel_sha256"] == PANEL_SHA
            and sources["release"] != BASE and sources["benchmark"] != RESEARCH, "Source identity")
    if expected_release:
        require(sources["release"] == expected_release, "Unexpected production revision")
    if expected_benchmark:
        require(sources["benchmark"] == expected_benchmark, "Unexpected benchmark revision")
    require(int(sources["github_run"]) > 0 and int(sources["github_attempt"]) >= 1, "Missing run identity")
    require(hashlib.sha256((root / "panel.csv").read_bytes()).hexdigest() == PANEL_SHA, "Changed frozen panel")
    panel = {r["family"]: dimensions(r) for r in table(root, "panel.csv")}
    expected = {(f, n) for f in panel for n in METHODS}
    ranking = json.loads((root / "ranking.json").read_text())
    require(ranking["phase"] == PHASE and ranking["seeds"] == list(SEEDS) and ranking["rounds_per_seed"] == 3 and
            ranking["panel_sha256"] == PANEL_SHA and ranking["full_output_tolerance"] == 0.002 and
            tuple(ranking["implementations"]) == METHODS, "Unexpected ranking method order or benchmark protocol")
    for n, spec in ranking["implementations"].items():
        require(spec[0] == ("direct" if n in DIRECT else "linked-public"), f"Timed entry type for {n}")
    builds, width, route_count, route_cells, cpus = check_builds(root, panel)
    binaries = check_identities(root, sources, builds, ranking)
    acc = index(table(root, "accuracy.csv"), lambda r: (*case(r), *fixture(r)))
    first = index(table(root, "first-calls.csv"), lambda r: (*case(r), *fixture(r)))
    require(set(acc) == set(first) == {(*c, *f) for c in expected for f in FIXTURES}, "Four complete FP64/first-call records per method/layout")
    first_times = defaultdict(dict)
    max_errors = defaultdict(float)
    numeric_cells = 0
    for token, r in acc.items():
        family, method, mode, block, seed = token
        m, n, _, _ = panel[family]
        detail = properties(r["detail"], ",")
        error = float(r["max_abs_error"])
        observed = first[token]
        require(dimensions(r) == dimensions(observed) == panel[family] and r["phase"] == PHASE and
                r["stress"] == str(int(mode != "random")) and int(r["checked"]) == m*n, f"FP64 coverage {token}")
        require(r["threads"] == detail["threads"] == "1" and r["cpu"].isdigit() and r["cpu"] == detail["cpu"] and
                r["guard_changes"] == r["nonfinite"] == detail["input_changes"] == "0" and
                r["verified"] == observed["verified"] == "1" and math.isfinite(error) and 0 <= error <= 0.002,
                f"FP64/guard/thread qualification {token}")
        seconds = positive(detail["first_sgemm_seconds"])
        require(seconds == positive(observed["first_sgemm_seconds"]), f"Inconsistent first-call timing {token}")
        first_times[family, method][mode, block, seed] = seconds
        cpus.add(int(r["cpu"]))
        max_errors[method] = max(max_errors[method], error)
        numeric_cells += m*n
    warm_rows = index(table(root, "timings.csv"), lambda r: (*case(r), int(r["block"]), int(r["round"])))
    require(set(warm_rows) == {(*c, *t) for c in expected for t in PAIRS}, "Six complete paired warm observations per case")
    warm = defaultdict(dict)
    total_gemms = 0
    for (family, method, b, j), r in warm_rows.items():
        m, n, k, _ = panel[family]
        shift = (list(panel).index(family) + j - 1) % len(METHODS)
        order = METHODS[shift:] + METHODS[:shift]
        if (b+j) % 2:
            order = tuple(reversed(order))
        require(dimensions(r) == panel[family] and r["phase"] == PHASE and r["mode"] == "random" and r["stress"] == "0" and
                int(r["seed"]) == SEEDS[b-1] and int(r["position"]) == order.index(method)+1, f"Paired timing order {family} {method} {b}/{j}")
        batch = min(512, max(32, math.ceil(8e7 / (2*m*n*k))))
        require(int(r["calls"]) == batch and r["cpu"].isdigit() and r["threads"] == "1", f"Paired batch/thread count {family}/{method}")
        warm[family, method][b, j] = positive(r["seconds"])
        total_gemms += batch
        cpus.add(int(r["cpu"]))
    require(cpus == {int(sources["cpu"])}, "Process workers or research route ran on a different logical CPU")
    admissions = index(table(root, "case-eligibility.csv"), case)
    require(set(admissions) == expected and all(dimensions(r) == panel[f] and
            (r["verified_cases"], r["warm_pairs"], r["qualified"], r["reason"]) == ("4", "6", "1", "all_checks_passed")
            for (f, n), r in admissions.items()), "Recorded per-layout admission disagrees with independent validation")

    cases, groups = {}, {}
    subsets = {"five_prior_holdouts": [f for f in panel if f.startswith("fresh-")],
               "three_controls": [f for f in panel if not f.startswith("fresh-")]}
    for candidate, peers in MATCHUPS.items():
        for peer in peers:
            label = f"{candidate}_vs_{peer}"
            cases[label] = {}
            for f in panel:
                ratios = [warm[f, peer][q] / warm[f, candidate][q] for q in PAIRS]
                first_ratios = [first_times[f, peer][q] / first_times[f, candidate][q] for q in FIXTURES]
                cases[label][f] = dict(candidate_us=1e6*median(warm[f, candidate].values()),
                    peer_us=1e6*median(warm[f, peer].values()), paired_ratios=ratios, median=median(ratios),
                    wins=sum(v>1 for v in ratios), seed_medians=[median(ratios[:3]), median(ratios[3:])],
                    first_ratios=first_ratios, first_random_median=median(first_ratios[:2]))
            groups[label] = {}
            for subset, families in subsets.items():
                samples = [cases[label][f] for f in families]
                medians = [s["median"] for s in samples]
                gm = geometric_mean(medians)
                by_seed = [geometric_mean(s["seed_medians"][i] for s in samples) for i in (0, 1)]
                wins, floor = sum(v>1 for v in medians), min(medians)
                groups[label][subset] = dict(geometric_mean=gm, per_seed_geometric_means=by_seed,
                    faster_layouts=wins, minimum=floor, paired_wins=sum(s["wins"] for s in samples),
                    shapes=len(samples), first_random_geometric_mean=geometric_mean(s["first_random_median"] for s in samples))
    generic = groups["cmake-att_vs_cmake-default"]["five_prior_holdouts"]
    parity = groups["cmake-native-on_vs_research-candidate"]["five_prior_holdouts"]
    gates = {"generic_speed": generic["geometric_mean"] >= 1.03 and min(generic["per_seed_geometric_means"]) > 1 and
                              generic["faster_layouts"] >= 4 and generic["minimum"] >= 0.98,
             "matched_native_fidelity": parity["geometric_mean"] >= 0.98 and parity["minimum"] >= 0.95}
    direct_leaders = {}
    for n in ("cmake-att", "cmake-native-on", "research-candidate"):
        leads = {}
        for f in panel:
            best = min(DIRECT, key=lambda d: (median(warm[f, d].values()), d))
            ratios = [warm[f, best][p] / warm[f, n][p] for p in PAIRS]
            leads[f] = dict(peer=best, peer_us=1e6*median(warm[f, best].values()), speedup=median(ratios),
                            wins=sum(x > 1 for x in ratios), ratios=ratios)
        direct_leaders[n] = leads
    host = dict(line.split(":", 1) for line in (root/"host.txt").read_text().splitlines() if ":" in line)
    model = host["Model name"].strip()
    require(host["Architecture"].strip() == "x86_64" and model, "Native x86 host model")
    return dict(phase=PHASE, host=model, release=sources["release"], benchmark=sources["benchmark"], research=RESEARCH,
                github_run=int(sources["github_run"]), github_attempt=int(sources["github_attempt"]),
                cpu=next(iter(cpus)), tile_width=width, panel_sha256=PANEL_SHA, product_builds=builds, timed_binaries=binaries,
                methods=METHODS, gates=gates, groups=groups, cases=cases, direct_leaders=direct_leaders,
                first_ratio_order=[list(f) for f in FIXTURES], max_abs_errors=dict(max_errors),
                coverage=dict(qualified_cases=len(expected), fp64_fixtures=len(acc), fp64_cells=numeric_cells,
                              warm_batches=len(warm_rows), warm_gemms=total_gemms, first_calls=len(first),
                              product_routes=route_count, exact_product_route_cells=route_cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--release")
    parser.add_argument("--benchmark")
    args = parser.parse_args()
    print(json.dumps(verify(args.raw, args.release, args.benchmark), indent=2, allow_nan=False))
