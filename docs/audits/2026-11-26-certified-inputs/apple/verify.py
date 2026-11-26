"""Replay both M5 experiments from the saved measurements using Python's standard library."""
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics
from pathlib import Path

from boundary import portable


ROOT = Path(__file__).resolve().parent
SHAPES = {
    "optical-counts": (1797, 3823, 64),
    "optical-binary": (1797, 1934, 1024),
    "integer-cube": (1024, 1024, 1024),
    "general-float": (1797, 3823, 64),
}
SOURCES = {"optical-counts": ("pixel_count_features", 1761),
           "optical-binary": ("original_bitmaps", 1731)}
BASE = ["exact-int8", "accelerate", "cob"]


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_files():
    files = set()
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        assert name not in files and not path.is_absolute() and ".." not in path.parts, name
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, name
        files.add(name)
    required = {"README.md", "LICENSE", "exact.c", "bitset.c", "build.sh", "verify.py", "boundary.py",
                "boundary.json", "study.py", "confirm.py", "prepare.py", "protocol.md",
                "confirm-protocol.md", "instructions.txt", "reference.py", "reference.json"}
    for folder, check_file in (("results", "accuracy.csv"), ("confirm-results", "checks.csv")):
        required.update(f"{folder}/{f}" for f in ("timings.csv", "summary.json", "identity.json", check_file))
    assert files == required


def route(case, name):
    if name == "exact-int8":
        return "0" if case == "general-float" else "1"
    return "1" if name == "binary-bitset" else ""


def orders(methods, repeat, confirmed):
    if confirmed:
        permutations = list(itertools.permutations(methods))
        permutations *= repeat // len(permutations)
        random.Random(33114791 + len(methods)).shuffle(permutations)
        return permutations
    sequences = []
    for i in range(repeat):
        seq = methods[::-1] if (i // len(methods)) % 2 else methods
        pos = i % len(methods)
        sequences.append(seq[pos:] + seq[:pos])
    return sequences


def verify_timings(folder, confirmed):
    rows = records(folder / "timings.csv")
    summary = read_json(folder / "summary.json")
    cases = set(SOURCES if confirmed else SHAPES)
    assert set(summary) == {r["case"] for r in rows} == cases
    assert len(rows) == (168 if confirmed else 129)
    for case in sorted(cases):
        methods = BASE + (["binary-bitset"] if case == "optical-binary" else [])
        repeats = 24 if confirmed else 3 * len(methods)
        subset = [r for r in rows if r["case"] == case]
        assert set(summary[case]) == set(methods) and len(subset) == repeats * len(methods)
        times = {name: [] for name in methods}
        for rep, expected in enumerate(orders(methods, repeats, confirmed)):
            block = sorted((r for r in subset if int(r["round"]) == rep),
                           key=lambda r: int(r["position"]))
            assert [r["method"] for r in block] == list(expected)
            assert [int(r["position"]) for r in block] == list(range(len(methods)))
            for r in block:
                name = r["method"]
                assert tuple(int(r[k]) for k in ("m", "n", "k")) == SHAPES[case]
                assert all(0 < float(r[k]) < math.inf for k in ("seconds", "thread_seconds"))
                assert r["route"] == route(case, name)
                if confirmed:
                    assert int(r["slot"]) == rep % 5
                times[name].append(float(r["seconds"]))
        for name, values in times.items():
            rec = summary[case][name]
            assert rec["pairs" if confirmed else "rounds"] == repeats
            assert math.isclose(rec["median_ms"], 1000 * statistics.median(values), rel_tol=1e-13)
            if confirmed:
                comparisons = [(name, "exact-int8", "over_exact_int8", "exact_faster")]
            else:
                comparisons = [(base, name, "vs_" + base, wins)
                               for base, wins in (("accelerate", "wins_accel"), ("cob", "wins_cob"))]
            for baseline, candidate, ratio_field, wins_field in comparisons:
                paired = [u / v for u, v in zip(times[baseline], times[candidate], strict=True)]
                assert math.isclose(rec[ratio_field], statistics.median(paired), rel_tol=1e-13)
                assert rec[wins_field] == sum(p > 1 for p in paired)
        if confirmed:
            r = summary[case]["cob"]
            print(f"{case}: vs COB {r['over_exact_int8']:.3f}x, {r['exact_faster']}/{repeats} paired wins")
    return len(rows)


def verify_numerics():
    input_hashes = {}
    checks = records(ROOT / "results/accuracy.csv")
    assert len(checks) == 13 and {r["case"] for r in checks} == set(SHAPES)
    for case, (m, n, k) in SHAPES.items():
        methods = BASE + (["binary-bitset"] if case == "optical-binary" else [])
        subset = [r for r in checks if r["case"] == case]
        assert len(subset) == len(methods) and {r["method"] for r in subset} == set(methods)
        hashes = {(r["A_sha256"], r["B_sha256"]) for r in subset}
        assert len(hashes) == 1
        input_hashes[case] = hashes.pop()
        assert all(re.fullmatch("[a-f0-9]{64}", h) for h in input_hashes[case])
        for r in subset:
            assert tuple(int(r[i]) for i in ("m", "n", "k", "checked")) == (m, n, k, m*n)
            assert r["route"] == route(case, r["method"])
            if case != "general-float":
                assert all(float(r[i]) == 0 for i in ("fp32_mismatches", "max_abs", "rms"))
        if case == "general-float":
            metrics = {r["method"]: tuple(r[i] for i in ("fp32_mismatches", "max_abs", "rms")) for r in subset}
            assert metrics["exact-int8"] == metrics["accelerate"]
    refs = {r["workload"]: r for r in read_json(ROOT / "reference.json")}
    assert set(refs) == {w for w, _ in SOURCES.values()}
    confirmations = records(ROOT / "confirm-results/checks.csv")
    assert len(confirmations) == 10 and {r["case"] for r in confirmations} == set(SOURCES)
    for case, (workload, recognized) in SOURCES.items():
        m, n, k = SHAPES[case]
        ref = refs[workload]
        assert tuple(ref[i] for i in ("m", "n", "k", "test_records", "recognized")) == (m, n, k, m, recognized)
        subset = [r for r in confirmations if r["case"] == case]
        assert len(subset) == 5 and {int(r["slot"]) for r in subset} == set(range(5))
        for r in subset:
            assert tuple(int(r[i]) for i in ("lda", "ldb", "ldc", "cells")) == (k, n, n+8, m*n)
            assert tuple(int(r[i]) for i in ("distinct_allocations", "a_mod64", "b_mod64", "c_mod64")) == (3, 0, 0, (4*(n+8+3)) % 64)
            assert r["inputs_unchanged"] == r["all_method_call_guards_passed"] == "1"
            assert r["output_sha256"] == ref["f32_row_major_product_sha256"]
            assert (r["A_sha256"], r["B_sha256"]) == input_hashes[case]


def main():
    if not __debug__:
        raise SystemExit("Run Python without -O to enable verification checks.")
    verify_files()
    pilot, confirm = (read_json(ROOT / d / "identity.json") for d in ("results", "confirm-results"))
    assert pilot["reference_threads"] == 1 and pilot["time_unit"] == "seconds"
    for rec in (pilot, confirm):
        assert rec["cpu"] == "Apple M5 Max" and rec["machine"] == "arm64"
        assert rec["svl_words"] == 16 and rec["serial_accelerate_setting"] == 1
        assert rec["affinity"] == "unavailable"
    assert all(pilot[k] == confirm[k] for k in ("macos", "numpy"))
    assert set(pilot["libraries"]) == {"kernel", "openblas", "cob"}
    assert set(confirm["libraries"]) == {"kernel", "cob"}
    assert all(pilot["libraries"][k] == h for k, h in confirm["libraries"].items())
    assert all(re.fullmatch("[0-9a-f]{64}", h) for h in pilot["libraries"].values())
    instructions = (ROOT / "instructions.txt").read_text()
    assert len(re.findall(r"\bsmopa\s+za[0-3]\.s,.*\.b", instructions)) == 20
    assert read_json(ROOT / "boundary.json") == {
        "cases": 54, "fallback_cases": 46, "exact_integer_cells": 5605,
        "alias_stride_zero_checks": 3, "portable_checked_words": portable()}
    verify_numerics()
    p = verify_timings(ROOT / "results", False)
    c = verify_timings(ROOT / "confirm-results", True)
    print(f"Verified {p} pilot calls, {c} confirmation calls, and {portable()} FP32 certificate boundary words.")


if __name__ == "__main__":
    main()
