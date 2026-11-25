"""Report all frozen cases and refuse incomplete or numerically invalid runs."""
import csv
from collections import Counter, defaultdict
from math import isfinite
from statistics import median
import sys

pairs_file, log_file, arch = sys.argv[1:]
mode = {"x86_64":"native-x86-dispatch", "aarch64":"native-arm-neon"}[arch]
shapes = [(1536,1536,1536),(2048,2048,2048),(3072,3072,3072),(4096,4096,4096),
          (3072,3328,2944),(2049,2112,1984),(1024,1024,1024),
          (1,1536,1024),(1,2049,1025)]

def blas_name(s):
    return "openblas-sgemv" if s[0]==1 else "openblas-sgemm"
seeds = {0:0, 1:0x12345678, 2:0x649f539b}
groups = defaultdict(list)
checks = {}
errors = []
with open(pairs_file,newline="") as f:
    for row in csv.DictReader(f):
        shape = tuple(int(row[v]) for v in ("m","n","k"))
        block = int(row["block"])
        if row["mode"]!=mode or shape not in shapes or block not in (1,2) or int(row["seed"])!=seeds[block]:
            errors.append("Unexpected mode, shape, block or seed in pairs.csv")
        groups[shape,row["competitor"]].append(row)
with open(log_file) as f:
    for line in f:
        if not line.startswith("accuracy,"):
            continue
        _,kind,seed,block,impl,m,n,k,count,mx,_,status=next(csv.reader([line]))
        s=(int(m),int(n),int(k)); block=int(block); key=(kind,block,impl,s)
        if key in checks or int(count)!=s[0]*s[1] or int(seed)!=seeds.get(block):
            errors.append("Invalid/duplicate accuracy scan")
        value=float(mx)
        good=status=="pass" and isfinite(value) and value<=0.002
        checks[key]=(value,good)
        if not good: errors.append(f"{kind} block {block} {impl} {s} failed")
expected_checks = {("random",b,impl,s) for s in shapes for b in (1,2)
                   for impl in ("baseline","candidate",blas_name(s))}
expected_checks.update(("cancel",0,impl,(2048,2048,2048)) for impl in ("baseline","candidate","openblas-sgemm"))
if set(checks)!=expected_checks: errors.append("Missing or unexpected full FP64 result")
if set(groups)-{(s,c) for s in shapes for c in ("baseline",blas_name(s))}: errors.append("Unexpected timing group")

def result(s,competitor):
    rows=groups[s,competitor]
    indices=Counter((int(r["block"]),int(r["pair"])) for r in rows)
    first=Counter(r["first"] for r in rows)
    values=[float(r["candidate_speedup"]) for r in rows]
    if indices!=Counter((b,p) for b in (1,2) for p in (1,2,3)) or first!={competitor:3,"candidate":3} or not all(isfinite(v) and v>0 for v in values):
        errors.append(f"Incomplete/invalid pairs for {s}, {competitor}")
        return "unverified"
    return f"{median(values):.3f}× ({sum(v>1 for v in values)}/6)"

print(f"## Optional one-level Strassen on {arch}\n")
print("Ratios above 1.0 favor Strassen. Each ratio is the median of six paired whole calls. Baseline commit 5e40c1c.\n")
print("| M × N × K | vs previous native COB (wins) | OpenBLAS routine | vs OpenBLAS (wins) | Candidate maximum FP64 error, two full scans |")
print("|---|---:|---|---:|---:|")
for s in shapes:
    dim=" × ".join(map(str,s))
    scans=[checks.get(("random",b,"candidate",s)) for b in (1,2)]
    error=f"{max(v[0] for v in scans):.7g}" if all(v and v[1] for v in scans) else "unverified"
    print(f"| {dim} | {result(s,'baseline')} | {blas_name(s).split('-')[1].upper()} | {result(s,blas_name(s))} | {error} |")
print("\nThe 2048-square cancellation case checks all 4,194,304 outputs independently against FP64 for each implementation.\n")
for impl in ("baseline","candidate","openblas-sgemm"):
    reading=checks.get(("cancel",0,impl,(2048,2048,2048)))
    label=(f"{reading[0]:.7g} ({'passed' if reading[1] else 'failed'})") if reading else "missing"
    print(f"- {impl} maximum absolute error {label}")
if errors:
    print("\nIncomplete or failed verification.\n")
    for reason in dict.fromkeys(errors): print(f"- {reason}")
    sys.exit(1)
