"""Ask the actual pinned LIBXSMM dispatcher about each registered row-major shape."""
import csv
import ctypes as ct
from pathlib import Path
import sys

panel, vendor, output = map(Path, sys.argv[1:])
with panel.open(newline="") as source, output.open("w", newline="") as dest:
    rows = list(csv.DictReader(source))
    writer = csv.DictWriter(dest, fieldnames=("family","m","n","k","pad","implementation","code_bytes","admitted","reason"))
    writer.writeheader()
    for name in ("libxsmm-dispatch","libxsmm-retained"):
        lib = ct.CDLL(str((vendor / (name + ".so")).resolve()))
        if lib.rank_vendor_init() != 0:
            raise ValueError("Unavailable LIBXSMM: " + name)
        inspect = lib.rank_xsmm_code_size
        inspect.restype = ct.c_size_t
        inspect.argtypes = [ct.c_int] * 6
        for row in rows:
            m,n,k,p = (int(row[key]) for key in ("m","n","k","pad"))
            extent = inspect(m,n,k,k+p,n+p,n+p)
            writer.writerow(dict(row,implementation=name,code_bytes=extent,admitted=int(extent>0),
                                 reason="nonreference_jit_exact_shape" if extent else "no_verified_nonreference_kernel"))
