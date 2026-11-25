"""Qualify complete, untimed public products and their instrumented companions."""
import argparse
import csv
import ctypes as ct
import importlib.util
from pathlib import Path
import subprocess
import sys

source = Path(__file__).with_name("hk-child-routes.py")
spec = importlib.util.spec_from_file_location("hk_child_routes", source)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

HK = {"cob-hk15-public": 1024, "cob-hk15-parent-960": 960}
FASTMMW = {"fastmmw-768": 49, "fastmmw-1536": 7}
PANEL = [(1970, 3099, 2829, 2), (1856, 3063, 3291, 9), (2234, 3024, 2991, 13)]
DIMENSIONS = ["m", "n", "k", "pad"]
COUNTS = ["hk_children", "fused_children", "other_children", "non_hk_fused", "scoped_children"]
EXACT = ["output_elements", "timed_binary_exact", "trace_binary_exact",
         "timed_inputs_unchanged", "trace_inputs_unchanged"]
TRACE_NAMES = ["rank_hk15_route_count", "rank_fastmmw_leaves", "rank_fastmmw_padded",
               "rank_fastmmw_leaf_config"]


def leaf_identity(trace):
    # The route companion wraps the actual Fortran SGEMM symbol. Dynamic
    # OpenBLAS exposes its own build string. For a hidden static build, both
    # local definitions must be in the companion binary's regular symbol table.
    config = getattr(trace, "rank_fastmmw_leaf_config", None) or getattr(trace, "openblas_get_config", None)
    if config is not None:
        config.argtypes, config.restype = [], ct.c_char_p
        value = (config() or b"").decode(errors="replace")
        if not value.lower().startswith("openblas"):
            raise RuntimeError(f"unexpected BLAS configuration {value!r}")
        return value
    result = subprocess.run(["nm", "--defined-only", trace._name], text=True,
                            capture_output=True, check=True)
    symbols = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    required = {"sgemm_", "__wrap_sgemm_", "openblas_get_config", "openblas_get_parallel"}
    if not required.issubset(symbols):
        raise RuntimeError(f"unverified static OpenBLAS leaf, missing {sorted(required - symbols)}")
    return "linked static OpenBLAS"


def load_pair(directory, name):
    plain = fixture.load_method((directory / f"{name}.so").resolve())
    if any(hasattr(plain, symbol) for symbol in TRACE_NAMES):
        raise RuntimeError(f"{name} timed binary exports a diagnostic interface")
    trace = fixture.load_method((directory / f"{name}.trace.so").resolve())
    if name in HK:
        trace.rank_hk15_route_count.argtypes = [ct.c_int]
        trace.rank_hk15_route_count.restype = ct.c_int
        return plain, trace, ""
    for counter in (trace.rank_fastmmw_leaves, trace.rank_fastmmw_padded):
        counter.argtypes, counter.restype = [], ct.c_int
    return plain, trace, leaf_identity(trace)


def outputs(a, b, c, m, n, k, plain, trace):
    checks = []
    for engine in (plain, trace):
        a.fill(1, float("nan"))
        b.fill(1, float("nan"))
        c.fill(float("nan"), 731.5)
        engine.rank_vendor_sgemm(m, n, k, a.pointer, a.ld, b.pointer, b.ld, c.pointer, c.ld)
        # Each term of C[i,j] equals (-1)^(i+j). Check logical cells and row
        # padding bitwise, and compare both complete inputs after each call.
        checks.append((c.matches(k, 731.5), a.matches(1, float("nan")) and b.matches(1, float("nan"))))
    return dict(zip(EXACT, [m*n, int(checks[0][0]), int(checks[1][0]),
                            int(checks[0][1]), int(checks[1][1])]))


def check_one(name, pair, shape, matrices):
    m, n, k, pad = shape
    row = dict(implementation=name, **dict(zip(DIMENSIONS, shape)))
    if name in HK:
        row["threshold"] = HK[name]
    try:
        if isinstance(pair, Exception):
            raise pair
        plain, trace, config = pair
        row.update(outputs(*matrices, m, n, k, plain, trace))
        if name in HK:
            mh, nh, kh = m // 2, n // 3, k // 3
            area = mh*nh + mh*kh + nh*kh
            volume = 3*mh*nh*kh
            fused = 15 * (volume >= HK[name] * area)
            scoped = 15 * (name == "cob-hk15-parent-960" and 960*area <= volume < 1024*area)
            actual = tuple(trace.rank_hk15_route_count(i) for i in range(5))
            row.update(zip(COUNTS, actual))
            correct_route = actual == (15, fused, 15-fused, 0, scoped)
        else:
            calls = trace.rank_fastmmw_leaves()
            padded = trace.rank_fastmmw_padded()
            row.update(leaf="openblas", leaf_evidence=config, leaf_calls=calls, padded=padded)
            correct_route = calls == FASTMMW[name] and padded == 1
        row["passed"] = int(correct_route and all(row[key] for key in EXACT[1:]))
        row["error"] = "" if row["passed"] else "route or matrix verification failed"
    except (AttributeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        row.update(passed=0, error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "outdir", "plugin_dir", "vendor_dir"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--fastmmw", nargs="*", choices=FASTMMW, default=list(FASTMMW))
    args = parser.parse_args()
    with args.panel.open(newline="") as src:
        cases = [tuple(int(row[key]) for key in DIMENSIONS) for row in csv.DictReader(src)]
    if cases != PANEL:
        parser.error(f"unexpected confirmation panel {cases!r}")
    methods = {}
    for name in [*HK, *dict.fromkeys(args.fastmmw)]:
        try:
            methods[name] = load_pair(args.plugin_dir if name in HK else args.vendor_dir, name)
        except (AttributeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            methods[name] = exc
    failures = 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    common = ["implementation", *DIMENSIONS]
    with (args.outdir / "routes.csv").open("w", newline="") as hk_out, \
         (args.outdir / "fastmmw-routes.csv").open("w", newline="") as fm_out:
        hk = csv.DictWriter(hk_out, common + ["threshold", *COUNTS, *EXACT, "passed", "error"])
        fm = csv.DictWriter(fm_out, common + ["leaf", "leaf_calls", "padded", "leaf_evidence", *EXACT, "passed", "error"])
        hk.writeheader()
        fm.writeheader()
        for case in cases:
            m, n, k, pad = case
            matrices = fixture.Matrix(m, k, pad), fixture.Matrix(k, n, pad), fixture.Matrix(m, n, pad)
            for name, pair in methods.items():
                row = check_one(name, pair, case, matrices)
                writer, handle = (hk, hk_out) if name in HK else (fm, fm_out)
                writer.writerow(row)
                handle.flush()
                if not row["passed"]:
                    failures += 1
                    print(f"{name} {case}: {row['error']}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f"{failures} case/method checks failed; see {args.outdir}")


if __name__ == "__main__":
    main()
