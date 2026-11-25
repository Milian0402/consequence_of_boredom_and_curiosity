"""Check isolated symbol binding, single-thread settings and padded CBLAS semantics."""
import ctypes
import subprocess
import sys

plugin = sys.argv[1]
symbols = subprocess.check_output(["nm", "-D", plugin], text=True).splitlines()
if any(line.split()[-1] in {"cblas_sgemm", "sgemm_"} for line in symbols):
    raise RuntimeError("BLAS symbols must be local to the AOCL adapter")

engine = ctypes.CDLL(plugin)
if engine.rank_vendor_init() != 0:
    raise RuntimeError("expected native, compiled single-thread, LP64 AOCL-BLAS")
f = ctypes.c_float
p = ctypes.POINTER(f)
engine.rank_vendor_sgemm.argtypes = [ctypes.c_int] * 3 + [p, ctypes.c_int, p, ctypes.c_int, p, ctypes.c_int]
engine.rank_vendor_sgemm.restype = None
a = (f * 8)(1, 2, 3, 111, 4, 5, 6, 111)
b = (f * 9)(1, 2, 222, 3, 4, 222, 5, 6, 222)
c = (f * 6)(9, 9, 333, 9, 9, 333)
engine.rank_vendor_sgemm(2, 2, 3, a, 4, b, 3, c, 3)
if list(c) != [22, 28, 333, 49, 64, 333]:
    raise RuntimeError(f"incorrect CBLAS output or padding: {list(c)}")
print("AOCL-BLAS padded LP64 row-major SGEMM smoke passed")
