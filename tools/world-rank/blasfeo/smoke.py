"""Verify stock BLASFEO's unpacked beta-zero ABI and single-thread linkage."""
import ctypes as c
import math
from pathlib import Path
import re
import subprocess
import sys


def check_binary(plugin):
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", subprocess.check_output(
        ["readelf", "-d", plugin], text=True))
    if set(needed) - {"libc.so.6", "libm.so.6", "ld-linux-aarch64.so.1",
                      "ld-linux-x86-64.so.2", "libgcc_s.so.1"}:
        raise RuntimeError(f"unexpected dependency of standalone SGEMM: {needed}")
    exported = subprocess.check_output(["nm", "-D", "--defined-only", plugin], text=True)
    if set(line.split()[-1] for line in exported.splitlines()) != {"rank_vendor_init", "rank_vendor_sgemm"}:
        raise RuntimeError(f"unexpected exported symbols: {exported}")
    imports = subprocess.check_output(["nm", "-D", "--undefined-only", plugin], text=True)
    if re.search(r"\b(pthread_create|thrd_create|clone3?|fork|vfork|syscall|dlopen|GOMP_\w+|omp_\w+)\b", imports):
        raise RuntimeError(f"thread creation, library loading or external parallel runtime in plugin: {imports}")
    print(f"ELF dependencies={needed}, private BLASFEO symbols, no thread-launch imports")


def check_gemm(plugin):
    lib = c.CDLL(str(plugin))
    if lib.rank_vendor_init():
        raise RuntimeError("BLASFEO initialization failed")
    scalar = c.c_float
    ptr = c.POINTER(scalar)
    gemm = lib.rank_vendor_sgemm
    gemm.argtypes = [c.c_int] * 3 + [ptr, c.c_int, ptr, c.c_int, ptr, c.c_int]
    gemm.restype = None
    cases = [(1, 17, 3, 0, 1, 3), (21, 1, 31, 3, 2, 4),
             (17, 29, 11, 3, 5, 7), (48, 48, 48, 0, 0, 0),
             (31, 65, 337, 2, 7, 3)]
    sentinel = 123456.0
    retained_nan = False
    for m, n, k, pa, pb, pc in cases:
        lda, ldb, ldc = k + pa, n + pb, n + pc
        a = (scalar * (m * lda))(*([sentinel] * (m * lda)))
        b = (scalar * (k * ldb))(*([sentinel] * (k * ldb)))
        out = (scalar * (m * ldc + 16))(*([sentinel] * (m * ldc + 16)))
        for i in range(m):
            for p in range(k):
                a[i * lda + p] = (3 * i - p) % 7 - 3
            for j in range(n):
                out[i * ldc + j] = float("nan")
        for p in range(k):
            for j in range(n):
                b[p * ldb + j] = (5 * p + 3 * j) % 11 - 5
        original = bytes(a), bytes(b)
        gemm(m, n, k, a, lda, b, ldb, out, ldc)
        for i in range(m):
            expected = [sum(a[i * lda + p] * b[p * ldb + j] for p in range(k)) for j in range(n)]
            actual = list(out[i * ldc:(i + 1) * ldc])
            bad = sum(math.isnan(value) for value in actual[:n])
            if bad:
                if not retained_nan:
                    print(f"NaN old C survived beta=0 on {(m, n, k)}, row {i}: {bad} nonfinite cells")
                retained_nan = True
            for j, value in enumerate(actual[:n]):
                if not math.isnan(value) and value != expected[j]:
                    raise RuntimeError(f"incorrect finite result at {(m, n, k, i, j)}")
            if actual[n:] != [sentinel] * pc:
                raise RuntimeError(f"incorrect row padding at {(m, n, k, i)}")
        if (bytes(a), bytes(b)) != original or list(out[m * ldc:]) != [sentinel] * 16:
            raise RuntimeError(f"input mutation or overwritten guard on {(m, n, k)}")
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("BLASFEO smoke process retained auxiliary threads")
    if retained_nan:
        raise SystemExit(3)
    print(f"{len(cases)} complete rectangular / square FP32 checks, NaN old C, padded strides, one OS thread")


if __name__ == "__main__":
    path = Path(sys.argv[1]).resolve()
    check_binary(path)
    check_gemm(path)
