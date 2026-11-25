"""Verify all timing pairs and full-output checks before summarizing a phase."""
import csv
from collections import defaultdict
from math import isclose, isfinite
from statistics import geometric_mean, median
import sys


TARGETS = {
    "discovery": [(2048,4096,2048,0), (3200,1920,2432,0), (1985,2753,2017,0), (3072,3072,3072,0), (4096,4096,4096,0), (2304,4608,2304,0)],
    "confirm": [(2240,4480,2240,0), (3840,2304,2944,0), (2560,2560,5120,0), (2689,3329,2497,0), (3073,3073,3073,0), (3584,3584,3584,0), (2688,5376,2688,0)],
    "recursive": [(3072,3072,3072,0), (4096,4096,4096,0)],
    "bilinear": [(2304,4608,2304,0), (2048,4096,2048,0)],
    "recursive-confirm": [(3584,3584,3584,0), (5120,5120,5120,0)],
    "bilinear-confirm": [(2688,5376,2688,0), (3072,6144,3072,0)],
    "hybrid-confirm": [(2816,3840,2304,0), (3520,2112,3072,0), (2817,4033,2113,0), (2816,3840,2304,7)],
    "skinny-final": [],
}
SKINNY = {"discovery": [(3,2048,1536,0), (6,3072,2048,0)],
          "skinny-final": [(2,2053,1559,0), (4,2519,1473,0), (5,3001,1283,7), (6,2689,2049,0), (7,3585,1793,0), (8,1793,2433,0)],
          "confirm": [(2,1537,2049,0), (5,4095,1025,0), (8,1024,3072,0)], "recursive": [], "bilinear": []}
CONTROLS = {"discovery": [], "recursive": [], "bilinear": [], "skinny-final": [(8,1024,3072,0)], "confirm": [(1536,1536,1536,0), (320,960,768,0), (2240,4480,2240,7), (16,1536,1024,0)]}
CANCELLATION = [(2048,2048,2048,0), (1793,2049,1537,7)]
SCALED = (1536,2688,1664,0)
EXTRA_SCALED = {"recursive": [(3072,3072,3072,0)], "bilinear": [(2304,4608,2304,0)]}
OPTIONAL_CONFIRM = {"recursive-confirm", "bilinear-confirm", "hybrid-confirm"}
SEEDS = {0: 0, 1: 0x12345678, 2: 0x649f539b}
MODES = {"x86_64": "native-x86-dispatch", "aarch64": "native-arm-neon"}
COMPETITORS = ("baseline", "openblas-sgemm")
HEADER = "phase,mode,m,n,k,pad,block,seed,pair,calls,competitor,first,competitor_seconds,candidate_seconds,candidate_speedup".split(",")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pairs(path, mode, phase, shapes, repeats):
    groups = defaultdict(dict)
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == HEADER, "invalid timing header")
        for row in reader:
            shape = tuple(int(row[name]) for name in ("m", "n", "k", "pad"))
            competitor = row["competitor"]
            block, pair = int(row["block"]), int(row["pair"])
            require(row["phase"] == phase and row["mode"] == mode and shape in shapes,
                    f"unexpected timing phase, mode or input at row {reader.line_num}")
            require(competitor in COMPETITORS and block in (1, 2) and 1 <= pair <= repeats,
                    f"unexpected implementation, block or index at row {reader.line_num}")
            require(int(row["seed"]) == SEEDS[block] and int(row["calls"]) == 1,
                    f"unexpected seed or number of timed calls at row {reader.line_num}")
            first = row["first"]
            before, after, ratio = (float(row[name]) for name in
                                     ("competitor_seconds", "candidate_seconds", "candidate_speedup"))
            require(first in ("candidate", competitor) and all(isfinite(v) and v > 0 for v in (before, after, ratio)),
                    f"invalid measurement at row {reader.line_num}")
            require(isclose(ratio, before / after, rel_tol=1e-7), f"inconsistent ratio at row {reader.line_num}")
            group = groups[shape, competitor]
            require((block, pair) not in group, f"duplicate timing pair at row {reader.line_num}")
            group[block, pair] = (before, after, ratio, first)
    expected_indices = {(b, p) for b in (1, 2) for p in range(1, repeats + 1)}
    for shape in shapes:
        for competitor in COMPETITORS:
            group = groups[shape, competitor]
            require(set(group) == expected_indices, f"incomplete paired timings for {shape}, {competitor}")
            for pair in range(1, repeats + 1):
                require(group[1,pair][3] != group[2,pair][3], f"unbalanced block order for {shape}, {competitor}")
                if pair > 1:
                    require(group[1,pair][3] != group[1,pair-1][3], f"nonalternating pair order for {shape}, {competitor}")
    return groups


def read_accuracy(path, shapes, scaled_shapes, cancel_shapes):
    checks = {}
    implementations = (*COMPETITORS, "candidate")
    expected = {("random", b, impl, s) for b in (1, 2) for impl in implementations for s in shapes}
    expected.update(("cancel", 0, impl, s) for impl in implementations for s in cancel_shapes)
    expected.update(("scaled", 0, impl, s) for impl in implementations for s in scaled_shapes)
    with open(path) as stream:
        for line in stream:
            if not line.startswith("accuracy,"):
                continue
            _, kind, seed, block, impl, m, n, k, pad, count, maximum, checksum, canaries, status = next(csv.reader([line]))
            block = int(block)
            s = (int(m), int(n), int(k), int(pad))
            key = (kind, block, impl, s)
            require(key in expected and key not in checks, f"unexpected or duplicate FP64 scan {key}")
            require(int(seed) == (117 if kind == "scaled" else SEEDS[block]) and int(count) == s[0] * s[1], f"wrong seed or scan extent for {key}")
            error = float(maximum)
            require(status == "pass" and int(canaries) == 0 and isfinite(error) and 0 <= error <= 0.002
                    and isfinite(float(checksum)), f"failed numerical or canary check for {key}")
            checks[key] = error
    require(set(checks) == expected, f"missing {len(expected - set(checks))} full FP64 scans")
    return checks


def main():
    pairs_path, accuracy_path, arch, phase = sys.argv[1:]
    targets = TARGETS[phase]
    skinny = SKINNY.get(phase, [])
    shapes = targets + skinny + CONTROLS.get(phase, [])
    repeats = 3 if phase in ("confirm", "skinny-final") or phase in OPTIONAL_CONFIRM else 1
    groups = read_pairs(pairs_path, MODES[arch], phase, shapes, repeats)
    active_confirm = targets[:1] if phase in OPTIONAL_CONFIRM else []
    scaled_shapes = [] if phase == "skinny-final" else [SCALED, *EXTRA_SCALED.get(phase, []), *active_confirm]
    cancel_shapes = [] if phase == "skinny-final" else [*CANCELLATION, *active_confirm]
    checks = read_accuracy(accuracy_path, shapes, scaled_shapes, cancel_shapes)
    gm = defaultdict(list)
    print(f"## Single-thread GEMM {phase} on {arch}\n")
    if phase == "skinny-final":
        print("Six skinny inputs cover two to eight rows. The seven-float stride padding exercises a noncontiguous layout. "
              "The 8 × 1024 × 3072 guard is reported separately. Ratios use the COB baseline or one-thread OpenBLAS time divided by candidate time.")
    else:
        print("Ratios use competitor time divided by candidate time. Both COB versions enable portable Strassen.")
        print("The baseline is the preceding published COB revision." if phase in ("discovery", "confirm") else "The baseline is the primary candidate.")
    if phase == "hybrid-confirm":
        print("The candidate uses a COB schedule with one-thread OpenBLAS leaves. The openblas-sgemm comparison uses one monolithic OpenBLAS call.")
    print()
    print("| Input | M × N × K | Stride pad | Competitor | Median ms, competitor / candidate | Median ratio | Candidate wins | Block geometric means | Max candidate FP64 error |")
    print("|---|---|---:|---|---:|---:|---:|---|---:|")
    for s in shapes:
        error = max(checks["random", b, "candidate", s] for b in (1, 2))
        for competitor in COMPETITORS:
            group = groups[s, competitor]
            values = list(group.values())
            base_ms = 1000 * median(x[0] for x in values)
            cand_ms = 1000 * median(x[1] for x in values)
            ratio = median(x[2] for x in values)
            block_ratios = [geometric_mean(group[b,p][2] for p in range(1, repeats + 1)) for b in (1, 2)]
            wins = sum(x[2] > 1 for x in values)
            role = "matrix target" if s in targets else "skinny target" if s in skinny else "guard"
            if role != "guard":
                gm[role, competitor].append(ratio)
            print(f"| {role} | {s[0]} × {s[1]} × {s[2]} | {s[3]} | {competitor} | {base_ms:.3f} / {cand_ms:.3f} | "
                  f"{ratio:.4f}× | {wins}/{len(values)} | {block_ratios[0]:.4f} / {block_ratios[1]:.4f} | {error:.7g} |")
    print("\nTarget-only geometric means of per-input median ratios\n")
    for (role, competitor), values in gm.items():
        print(f"- {competitor} {geometric_mean(values):.4f}× over {len(values)} {role}s.")
    peak = max(v for key, v in checks.items() if key[2] == "candidate")
    nout = sum(key[3][0] * key[3][1] for key in checks if key[2] == "candidate")
    tested_inputs = "two random seeds per shape"
    if cancel_shapes or scaled_shapes:
        tested_inputs += f", {len(cancel_shapes)} cancellation cases and {len(scaled_shapes)} scaled inputs"
    print(f"\nChecked all {nout:,} candidate outputs across {tested_inputs}. "
          f"Peak absolute error was {peak:.7g}. All padding canaries passed.\n")
    for shape in cancel_shapes:
        error = checks["cancel", 0, "candidate", shape]
        print(f"- Cancellation {shape[0]} × {shape[1]} × {shape[2]}, stride pad {shape[3]}, candidate max error {error:.7g}.")
    for s in scaled_shapes:
        print(f"- Scaled input {s[0]} × {s[1]} × {s[2]}, candidate max error {checks['scaled',0,'candidate',s]:.7g}.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError) as error:
        print(f"Benchmark verification failed: {error}", file=sys.stderr)
        sys.exit(1)
