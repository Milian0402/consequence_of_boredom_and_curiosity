"""Validate the predeclared groups, order balance and timings before summarizing."""
import csv
from collections import Counter, defaultdict
from math import isfinite
from statistics import median
import sys

source, arch = sys.argv[1:]
native = {"x86_64": "native-x86-dispatch", "aarch64": "native-arm-neon"}[arch]
medium = [(192, 192, 192), (383, 385, 389), (768, 768, 768), (256, 1024, 640),
          (1024, 192, 768), (7, 11, 13), (1, 1536, 1024), (1, 2049, 1025), (256, 256, 256)]
large = [(1024, 1024, 1024), (1536, 1536, 1536), (2048, 2048, 2048),
         (2049, 2112, 1984), (4096, 4096, 4096)]
groups = defaultdict(list)
with open(source, newline="") as f:
    for row in csv.DictReader(f):
        key = (row["mode"], tuple(int(row[x]) for x in ("m", "n", "k")), row["competitor"])
        groups[key].append(row)
expected = {(mode, shape, c) for mode in (native, "plain-c-autovec") for shape in medium
            for c in ("baseline", "openblas")}
expected.update((native, shape, "openblas") for shape in large)
if groups.keys() != expected:
    raise ValueError("Incomplete or unexpected shape/mode/competitor set")

def summarize(mode, shape, competitor, repeats):
    rows = groups[mode,shape,competitor]
    indices = Counter((int(r["block"]),int(r["pair"])) for r in rows)
    first = Counter(r["first"] for r in rows)
    if indices != Counter((b,p) for b in (1,2) for p in range(1,repeats+1)) or first != {competitor:repeats,"candidate":repeats}:
        raise ValueError(f"Missing, duplicated or unbalanced pairs: {mode}, {shape}, {competitor}")
    speedups = [float(r["candidate_speedup"]) for r in rows]
    if not all(isfinite(v) and v>0 for v in speedups):
        raise ValueError("Invalid speedup")
    return median(speedups),sum(v>1 for v in speedups)

print(f"## Native single-thread FP32 GEMM on {arch}\n")
print("Median paired ratios above 1.0 favor the candidate. The baseline is d2a59e8.\n")
print("| Path | M × N × K | vs baseline | vs OpenBLAS | OpenBLAS pairs won | FP64 checks per implementation per block |")
print("|---|---|---:|---:|---:|---:|")
for mode in (native,"plain-c-autovec"):
    for shape in medium + (large if mode == native else []):
        big=shape in large
        baseline="—" if big else f"{summarize(mode,shape,'baseline',5)[0]:.3f}×"
        speed,wins=summarize(mode,shape,"openblas",3 if big else 5)
        cells=shape[0]*shape[1]
        accuracy=f"256 sampled of {cells:,}" if big else f"{cells:,} full"
        dim=" × ".join(str(v) for v in shape)
        print(f"| {mode} | {dim} | {baseline} | {speed:.3f}× | {wins}/{6 if big else 10} | {accuracy} |")
