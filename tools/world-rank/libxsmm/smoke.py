"""Verify the JIT row-major adapter, including padding and new inputs on cache hits."""
import ctypes as c
from pathlib import Path
import sys


def check(path):
    lib = c.CDLL(str(Path(path).resolve()))
    if lib.rank_vendor_init() != 0:
        raise RuntimeError("LIBXSMM initialization failed")
    scalar = c.c_float
    ptr = c.POINTER(scalar)
    i = c.c_int
    gemm = lib.rank_vendor_sgemm
    gemm.argtypes = (i, i, i, ptr, i, ptr, i, ptr, i)
    gemm.restype = None
    shapes = [(7, 13, 5, 3), (1, 19, 2, 3), (17, 1, 5, 2),
              (31, 53, 47, 3), (48, 48, 48, 0), (64, 64, 64, 0)]
    layouts = [(m, n, k, k + pad, n + 2 * pad, n + 3 * pad) for m, n, k, pad in shapes]
    # Consecutive calls change just one leading dimension of the same shape.
    layouts += [(7, 13, 5, 5, 13, 13), (7, 13, 5, 8, 13, 13),
                (7, 13, 5, 8, 15, 13), (7, 13, 5, 8, 15, 19)]
    for m, n, k, lda, ldb, ldc in layouts:
        marker = 123456.0
        for offset in (0, 1):
            a = (scalar * (m * lda))(*([marker] * (m * lda)))
            b = (scalar * (k * ldb))(*([marker] * (k * ldb)))
            out = (scalar * (m * ldc))(*([marker] * (m * ldc)))
            for row in range(m):
                for p in range(k):
                    a[row * lda + p] = (row + 3 * p + offset) % 11 - 5
            for p in range(k):
                for col in range(n):
                    b[p * ldb + col] = (2 * p - col + offset) % 13 - 6
            old = bytes(a), bytes(b)
            gemm(m, n, k, a, lda, b, ldb, out, ldc)
            assert old == (bytes(a), bytes(b))
            for row in range(m):
                expected = [sum(a[row * lda + p] * b[p * ldb + col] for p in range(k))
                            for col in range(n)]
                actual = list(out[row * ldc:(row + 1) * ldc])
                assert actual == expected + [marker] * (ldc - n), (m, n, k, row)


if __name__ == "__main__":
    check(sys.argv[1])
    print("Ten LIBXSMM FP32 layouts passed, including same-shape stride changes and fresh inputs")
