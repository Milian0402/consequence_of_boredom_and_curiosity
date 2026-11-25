"""Compare complete compact and padded SGEMM outputs with scalar FP64."""
import ctypes as ct
import math
from pathlib import Path
from random import Random
import sys

def check(lib, m, n, k, pads, seed, fixture):
    lda, ldb, ldc = k + pads[0], n + pads[1], n + pads[2]
    def alloc(rows, ld):
        return (ct.c_float * (rows * ld + 16))(*([123456.] * (rows * ld + 16)))
    aa, bb, cc = alloc(m, lda), alloc(k, ldb), alloc(m, ldc)
    rng = Random(seed)
    for i in range(m):
        for j in range(n): cc[i * ldc + j] = float("nan")
    for i in range(m):
        for p in range(k):
            v = rng.uniform(-1.0, 1.0)
            if fixture == "cancel" and k // 2 <= p < 2 * (k // 2): v = aa[i * lda + p - k // 2]
            aa[i * lda + p] = v
    for p in range(k):
        for j in range(n):
            v = rng.uniform(-1.0, 1.0)
            if fixture == "cancel" and k // 2 <= p < 2 * (k // 2):
                v = -bb[(p - k // 2) * ldb + j] + v * 2**-10
            if fixture == "cancel" and k % 2 and p == k - 1: v *= 2**-10
            bb[p * ldb + j] = v
    originals = bytes(aa), bytes(bb)
    lib.rank_vendor_sgemm(m, n, k, aa, lda, bb, ldb, cc, ldc)
    expected_route = 15 if m % 2 == n % 3 == k % 3 == 0 else 0
    if lib.rank_hk233_last_products() != expected_route:
        raise ValueError("wrong route")
    worst = 0.
    for i in range(m):
        for j in range(n):
            ref = sum(aa[i * lda + p] * bb[p * ldb + j] for p in range(k))
            error = abs(ref - cc[i * ldc + j])
            if not math.isfinite(error) or error > .002: raise ValueError((i, j, ref, cc[i * ldc + j]))
            worst = max(worst, error)
        if list(cc[i * ldc + n:(i + 1) * ldc]) != [123456.] * pads[2]:
            raise ValueError("padding overwritten")
    if list(cc[-16:]) != [123456.] * 16 or originals != (bytes(aa), bytes(bb)):
        raise ValueError("end guard or inputs changed")
    print(f"{m}x{n}x{k} pads={pads} fixture={fixture} products={expected_route} max_abs={worst:.9g}")

if __name__ == "__main__":
    lib = ct.CDLL(str(Path(sys.argv[1]).resolve()))
    if lib.rank_vendor_init(): raise ValueError("initialization failed")
    p = ct.POINTER(ct.c_float)
    lib.rank_vendor_sgemm.argtypes = [ct.c_int] * 3 + [p,ct.c_int,p,ct.c_int,p,ct.c_int]
    lib.rank_vendor_sgemm.restype = None
    for fixture in ("random", "cancel"):
        for m, n, k, pads in [(2,3,3,(0,0,0)), (42,60,51,(1,3,7)),
                              (80,96,105,(0,0,0)), (43,60,51,(2,4,7))]:
            check(lib, m, n, k, pads, 802705, fixture)
