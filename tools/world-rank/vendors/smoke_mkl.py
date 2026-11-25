"""Check classic and direct-call Intel SGEMM with the same padded inputs."""
import ctypes
import sys

engine = ctypes.CDLL(sys.argv[1])
if engine.rank_vendor_init() != 0:
    raise RuntimeError("expected oneMKL's sequential LP64 library")
scalar = ctypes.c_float
ptr = ctypes.POINTER(scalar)
integer = ctypes.c_int
multiply = engine.rank_vendor_sgemm
multiply.argtypes = [integer, integer, integer, ptr, integer, ptr, integer, ptr, integer]
multiply.restype = None

# The first shape meets the Intel direct-call JIT size predicate. The second
# exceeds its 262144-product limit. Exercise a cache hit with new operands too.
for m, n, k, seed in ((7, 9, 5, 1), (7, 9, 5, 3), (64, 64, 65, 1)):
    lda, ldb, ldc = k + 1, n + 2, n + 3
    canary = 211
    a = (scalar * (m * lda))(*([canary] * (m * lda)))
    b = (scalar * (k * ldb))(*([canary] * (k * ldb)))
    c = (scalar * (m * ldc))(*([canary] * (m * ldc)))
    for i in range(m):
        for q in range(k):
            a[i * lda + q] = (i + q + seed) % 13 - 6
    for q in range(k):
        for j in range(n):
            b[q * ldb + j] = (q + 3 * j + seed) % 11 - 5
    multiply(m, n, k, a, lda, b, ldb, c, ldc)
    for i in range(m):
        for j in range(ldc):
            expected = sum(a[i * lda + q] * b[q * ldb + j] for q in range(k)) if j < n else canary
            if c[i * ldc + j] != expected:
                raise RuntimeError(f"invalid {m} x {n} x {k} output at ({i}, {j})")
print("Padded FP32 SGEMM and same-shape changing operands passed")
