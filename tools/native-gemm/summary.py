"""Check completeness and summarize paired native single-thread measurements."""
import csv
from collections import Counter, defaultdict
from statistics import median
import sys

source, arch = sys.argv[1:]
native = {"x86_64": "native-x86-c-autovec", "aarch64": "native-arm-neon"}[arch]
shapes = [(192, 192, 192), (383, 385, 389), (768, 768, 768),
          (256, 1024, 640), (1024, 192, 768), (7, 11, 13), (1, 1536, 1024)]
groups = defaultdict(list)
with open(source, newline="") as f:
    for row in csv.DictReader(f):
        key = (row["mode"], tuple(int(row[x]) for x in ("m", "n", "k")), row["competitor"])
        groups[key].append(row)
expected = {(mode, shape, c) for mode in (native, "plain-c-autovec") for shape in shapes
            for c in ("baseline", "openblas")}
if groups.keys() != expected:
    raise ValueError("Incomplete or unexpected shape/mode/competitor set")
print(f"## Native single-thread FP32 GEMM on {arch}\n")
print("Each speedup is the median of ten within-pair ratios. Above 1.0 favors the candidate.\n")
print("| Path | M × N × K | Relative to baseline | Relative to OpenBLAS |")
print("|---|---|---:|---:|")
for mode in (native, "plain-c-autovec"):
    for shape in shapes:
        ratios = []
        for c in ("baseline","openblas"):
            rows = groups[mode, shape, c]
            indices = Counter((int(r["block"]), int(r["pair"])) for r in rows)
            first = Counter(r["first"] for r in rows)
            if indices != Counter((b, p) for b in (1, 2) for p in range(1, 6)) or first != {c: 5, "candidate": 5}:
                raise ValueError(f"Missing, duplicated or unbalanced pairs: {mode}, {shape}, {c}")
            ratios.append(median(float(r["candidate_speedup"]) for r in rows))
        dim=" × ".join(str(v) for v in shape)
        print(f"| {mode} | {dim} | {ratios[0]:.3f}× | {ratios[1]:.3f}× |")
