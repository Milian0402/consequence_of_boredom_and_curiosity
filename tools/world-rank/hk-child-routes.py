"""Record the native HK15 child route on a separate, full public multiplication."""
import argparse
from array import array
import csv
import ctypes as ct
from pathlib import Path

FP = ct.POINTER(ct.c_float)
ARGS = [ct.c_int, ct.c_int, ct.c_int, FP, ct.c_int, FP, ct.c_int, FP, ct.c_int]


class Matrix:
    def __init__(self, rows, cols, pad):
        self.rows = rows
        self.cols = cols
        self.ld = cols + pad
        self.stride = self.ld * ct.sizeof(ct.c_float)
        self.storage = ct.create_string_buffer(rows * self.stride + 127)
        self.address = (ct.addressof(self.storage) + 127) & ~127
        self.pointer = ct.cast(self.address, FP)

    def row_images(self, magnitude, padding):
        suffix = array("f", [padding]) * (self.ld - self.cols)
        rows = []
        for sign in (1, -1):
            pair = array("f", [sign * magnitude, -sign * magnitude])
            logical = (pair * ((self.cols + 1) // 2))[:self.cols]
            rows.append((logical + suffix).tobytes())
        return rows

    def fill(self, magnitude, padding):
        rows = self.row_images(magnitude, padding)
        for i in range(self.rows):
            ct.memmove(self.address + i * self.stride, rows[i % 2], self.stride)

    def matches(self, magnitude, padding):
        rows = self.row_images(magnitude, padding)
        return all(ct.string_at(self.address + i * self.stride, self.stride) == rows[i % 2]
                   for i in range(self.rows))


def load_method(path):
    handle = ct.CDLL(str(path))
    if handle.rank_vendor_init() != 0:
        raise RuntimeError(f"initialization failed for {path}")
    handle.rank_vendor_sgemm.argtypes = ARGS
    handle.rank_vendor_sgemm.restype = None
    return handle


def check_case(m, n, k, pad, methods, writer, output):
    a, b = Matrix(m, k, pad), Matrix(k, n, pad)
    c = Matrix(m, n, pad)
    a.fill(1, float("nan"))
    b.fill(1, float("nan"))
    # A[i,p]=(-1)^(i+p), B[p,j]=(-1)^(p+j), so C[i,j]=k*(-1)^(i+j).
    # The bounded integer intermediates are exactly representable in FP32.
    args = (m, n, k, a.pointer, a.ld, b.pointer, b.ld, c.pointer, c.ld)
    h, w, d = m // 2, n // 3, k // 3
    failures = 0
    for name, threshold, plain, trace in methods:
        exact = []
        for engine in (plain, trace):
            c.fill(float("nan"), 731.5)
            engine.rank_vendor_sgemm(*args)
            exact.append(c.matches(k, 731.5))
        counts = tuple(trace.rank_hk15_route_count(i) for i in range(4))
        expect_fused = 15 if 3 * h * w * d >= threshold * (h*w + h*d + w*d) else 0
        passed = all(exact) and counts == (15, expect_fused, 15 - expect_fused, 0)
        writer.writerow([name, m, n, k, pad, threshold, *counts, m*n,
                         int(exact[0]), int(exact[1]), int(passed)])
        output.flush()
        failures += not passed
    return failures


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("routes", type=Path)
    cli.add_argument("plugins", type=Path)
    cli.add_argument("--method", nargs=2, action="append", required=True, metavar=("NAME", "CUTOFF"))
    args = cli.parse_args()
    methods = []
    for name, threshold in args.method:
        plain = load_method(args.plugins / f"{name}.so")
        trace = load_method(args.plugins / f"{name}.trace.so")
        if hasattr(plain, "rank_hk15_route_count"):
            raise RuntimeError(f"timed implementation {name} unexpectedly contains counters")
        trace.rank_hk15_route_count.argtypes = [ct.c_int]
        trace.rank_hk15_route_count.restype = ct.c_int
        methods.append((name, int(threshold), plain, trace))
    failures = 0
    with args.panel.open(newline="") as panel, args.routes.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["implementation", "m", "n", "k", "pad", "threshold",
                         "hk_children", "fused_children", "other_children", "non_hk_fused",
                         "output_elements", "timed_binary_exact", "trace_binary_exact", "passed"])
        for case in csv.DictReader(panel):
            failures += check_case(*(int(case[key]) for key in ("m", "n", "k", "pad")),
                                   methods, writer, output)
    if failures:
        raise SystemExit(f"{failures} route or exact-output checks failed, see {args.routes}")


if __name__ == "__main__":
    main()
