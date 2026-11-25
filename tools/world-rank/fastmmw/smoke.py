"""Validate full outputs, unchanged guards and observed native BLAS leaves."""
from array import array
import ctypes as ct
from ctypes.util import find_library
import json
import math
import os
from pathlib import Path
import random
import sys

out, cutoff = Path(sys.argv[1]), int(sys.argv[2])
fp, dp = ct.POINTER(ct.c_float), ct.POINTER(ct.c_double)
ii = ct.c_int
def pointer(values, dtype):
    return ct.cast(values.buffer_info()[0], dtype)

if os.environ.get("FASTMMW_ORACLE"):
    oracle = ct.CDLL(os.environ["FASTMMW_ORACLE"])
    if oracle.rank_vendor_init():
        raise RuntimeError("FP64 ranking oracle rejected single-thread initialization")
    oracle.rank_vendor_dgemm.argtypes = [ii]*3 + [dp,ii,dp,ii,dp,ii]
    oracle.rank_vendor_dgemm.restype = None
    reference = oracle.rank_vendor_dgemm
else:
    oracle = ct.CDLL(find_library("openblas"))
    oracle.cblas_dgemm.argtypes = [ii]*6 + [ct.c_double,dp,ii,dp,ii,ct.c_double,dp,ii]
    oracle.cblas_dgemm.restype = None
    oracle.openblas_set_num_threads(1)
    if oracle.openblas_get_num_threads() != 1 or oracle.openblas_get_parallel() != 0:
        raise RuntimeError("expected sequential OpenBLAS for the FP64 reference")

    def reference(m,n,k,a,lda,b,ldb,c,ldc):
        oracle.cblas_dgemm(101,111,111,m,n,k,1,a,lda,b,ldb,0,c,ldc)

engines = {}
for name in ("stock", "stock-trace", "padded", "padded-trace"):
    lib = ct.CDLL(str((out / f"fastmmw-{name}.so").resolve()))
    if lib.rank_vendor_init() or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("expected a single-thread initialized backend")
    lib.rank_vendor_sgemm.argtypes = [ii] * 3 + [fp, ii, fp, ii, fp, ii]
    lib.rank_vendor_sgemm.restype = None
    if hasattr(lib, "rank_fastmmw_leaves") != name.endswith("-trace"):
        raise RuntimeError("unexpected trace interface")
    engines[name] = lib

def inputs(rows, cols, ld, rng):
    data = array("f")
    for _ in range(rows):
        data.extend((rng.randrange(-1000, 1001) / 997.0 for _ in range(cols)))
        data.extend([math.nan] * (ld-cols))
    data.extend([math.nan] * 16)
    return data

rows = []
def check(m, n, k, pads):
    multiple, expected_leaves = 1, 1
    while (min(m,n,k) + multiple - 1) // multiple > cutoff:
        multiple *= 2
        expected_leaves *= 7
    needs_pad = any(dim % multiple for dim in (m,n,k))
    rng = random.Random(91771 + m)
    lda, ldb, ldc = k+pads[0], n+pads[1], n+pads[2]
    a, b = inputs(m,k,lda,rng), inputs(k,n,ldb,rng)
    orig_a, orig_b = a.tobytes(), b.tobytes()
    ad, bd = array("d", a), array("d", b)
    ref = array("d", [0]) * (m*n)
    reference(m,n,k,pointer(ad,dp),lda,pointer(bd,dp),ldb,pointer(ref,dp),n)
    for name, lib in engines.items():
        if needs_pad and name.startswith("stock"):
            continue
        sentinel = 731.5
        c = (array("f", [math.nan]) * n + array("f", [sentinel]) * (ldc-n)) * m
        c.extend([sentinel] * 16)
        lib.rank_vendor_sgemm(m,n,k,pointer(a,fp),lda,pointer(b,fp),ldb,pointer(c,fp),ldc)
        leaves = lib.rank_fastmmw_leaves() if name.endswith("-trace") else None
        padded = bool(lib.rank_fastmmw_padded()) if leaves is not None else None
        maximum, nonfinite, guard_errors = 0.0, 0, 0
        for i in range(m):
            for j in range(n):
                delta = abs(c[i*ldc+j]-ref[i*n+j])
                if math.isfinite(delta): maximum = max(maximum,delta)
                else: nonfinite += 1
            guard_errors += sum(c[i*ldc+j] != sentinel for j in range(n,ldc))
        guard_errors += sum(x != sentinel for x in c[-16:])
        unchanged = a.tobytes() == orig_a and b.tobytes() == orig_b
        threads = len(list(Path("/proc/self/task").iterdir()))
        valid = (maximum<=0.002 and not nonfinite and not guard_errors and unchanged and threads==1
                 and (leaves is None or (leaves==expected_leaves and padded==needs_pad)))
        record = dict(variant=name,m=m,n=n,k=k,lda=lda,ldb=ldb,ldc=ldc,output_count=m*n,
                      max_abs_error=maximum,nonfinite=nonfinite,guard_errors=guard_errors,
                      inputs_unchanged=unchanged,threads=threads,observed_leaves=leaves,
                      expected_leaves=expected_leaves,observed_staging=padded,valid=valid)
        rows.append(record)
        print(json.dumps(record),flush=True)
        (out/"qualification.json").write_text(json.dumps(rows,indent=2)+"\n")

# Cover a compact leaf, compact and padded recursive matrices with no staging,
# the original unsupported odd fringe, and the actual native follow-up panels.
even = 770
check(17,26,11,(0,0,0))
check(even,even+4,even+8,(0,0,0))
check(even,even+4,even+8,(3,1,5))
check(even+1,even+5,even+9,(3,1,5))
check(1920,2880,2880,(0,0,0))
check(1970,3099,2829,(2,2,2))
check(1856,3063,3291,(9,9,9))
check(2234,3024,2991,(13,13,13))
if not all(row["valid"] for row in rows):
    raise SystemExit("FastMMW failed qualification; see the complete observations.")
