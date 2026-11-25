"""Check exact full outputs and actual native routing for the frozen panel."""
import argparse
import csv
import ctypes as ct
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("parent_fixture", Path(__file__).with_name("hk-parent-routes.py"))
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
HK = "cob-hk15-parent-960"
PUBLIC = "cob-openblas-winograd"
FAST = {"fastmmw-768": 768, "fastmmw-1536": 1536}
OWN = {"winograd-cob-1": 1, "winograd-cob-2": 2}
X86 = ("hk-register", "hk-k288")
PANEL = [(1920,2880,2880,0), (2048,2880,3264,3), (1926,3375,2823,5),
         (2138,2829,3333,7), (3375,1926,2823,5)]
DIMS = parent.DIMENSIONS
EXACT = parent.EXACT
SYMBOLS = [*parent.TRACE_NAMES, "rank_openblas_winograd_calls", "rank_winograd_leaf_count", "rank_hk_x86_leaf_count"]


def getter(handle, name, indexed=False):
    fn = getattr(handle, name)
    fn.argtypes, fn.restype = ([ct.c_int] if indexed else []), ct.c_int
    return fn


def check(name, shape, directory, matrices):
    m, n, k, _ = shape
    plain = parent.fixture.load_method((directory/f"{name}.so").resolve())
    traced = parent.fixture.load_method((directory/f"{name}.trace.so").resolve())
    if any(hasattr(plain, symbol) for symbol in SYMBOLS):
        raise ValueError(f"diagnostics present in timed DSO for {name}")
    public = getter(traced, "rank_openblas_winograd_calls") if name == PUBLIC else None
    before = public() if public else 0
    row = parent.outputs(*matrices, m, n, k, plain, traced)
    valid = all(row[k] == 1 for k in EXACT[1:])
    if name in FAST:
        splits, calls = 1, 1
        while (min(m,n,k)+splits-1)//splits > FAST[name]:
            splits *= 2
            calls *= 7
        measured = getter(traced, "rank_fastmmw_leaves")()
        padded = getter(traced, "rank_fastmmw_padded")()
        row.update(leaf="openblas", leaf_calls=measured, padded=padded, leaf_evidence=parent.leaf_identity(traced))
        valid &= measured == calls and padded == int(any(d%splits for d in (m,n,k)))
    else:
        h, w, d = m//2, n//3, k//3
        hk = m%2 == n%3 == k%3 == 0 and min(h,w,d) >= 768 and max(h,w,d) <= 1280 and 4*max(h,w,d) <= 5*min(h,w,d)
        if name == HK:
            kind, symbol = "hk-parent", "rank_hk15_route_count"
            scoped = hk and 960*(h*w+h*d+w*d) <= 3*h*w*d < 1024*(h*w+h*d+w*d)
            expected = [15,15,0,0,15*int(scoped)] if hk else [0,0,0,1,0]
        elif name in X86:
            kind, symbol = "hk-x86", "rank_hk_x86_leaf_count"
            active = hk and (name == "hk-register" or 513 <= k//6 <= 576)
            expected = [15,105] if active else [0,0]
        else:
            levels = OWN.get(name, 2)
            quantum = 2**levels
            core = 7**levels
            fringe = int(bool(m%quantum))+int(bool(n%quantum))
            if name == PUBLIC:
                kind = "public-openblas"
                expected = [core+fringe]
            else:
                kind, symbol = "winograd-cob", "rank_winograd_leaf_count"
                expected = [core+fringe,core,fringe,levels,0]
        if name == PUBLIC:
            observed = [public()-before]
        else:
            counts = getter(traced, symbol, indexed=True)
            observed = [counts(i) for i in range(len(expected))]
            if name in X86:
                width = counts(2)
                valid &= width in ((16,32) if active else (0,))
                row["native_tile_nr"] = width
            if name in OWN:
                row["nested_hk_entries"], row["nested_fused"], row["nested_hk_fused"], row["nested_hk_children"] = [counts(i) for i in range(5,9)]
        row.update(route=kind, observed="|".join(map(str,observed)), expected="|".join(map(str,expected)))
        valid &= expected == observed
    row["passed"] = int(valid)
    row["error"] = "" if valid else "unexpected routing, input mutation, or output mismatch"
    return row


def main():
    if sys.argv[1:2] == ["--qualify"]:
        if len(sys.argv) != 4 or sys.argv[3] not in FAST:
            raise SystemExit("expected --qualify RESULT_DIR FASTMMW_METHOD")
        destination, name = Path(sys.argv[2]), sys.argv[3]
        with (destination/"fastmmw-routes.csv").open(newline="") as stream:
            records = [r for r in csv.DictReader(stream) if r["implementation"] == name]
        if ([tuple(int(r[d]) for d in DIMS) for r in records] != PANEL
                or not any(r["passed"] == "1" and r["leaf"] == "openblas" for r in records)):
            raise SystemExit(f"no complete route inventory with a qualifying layout for {name}")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "outdir", "plugins", "vendor"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--x86", action="store_true")
    args = parser.parse_args()
    with args.panel.open(newline="") as stream:
        shapes = [tuple(int(r[key]) for key in DIMS) for r in csv.DictReader(stream)]
    if shapes != PANEL:
        parser.error("unexpected panel")
    names = [HK, *OWN, PUBLIC, *(X86 if args.x86 else ()), *FAST]
    failed = 0
    with (args.outdir/"routes.csv").open("w",newline="") as project, (args.outdir/"fastmmw-routes.csv").open("w",newline="") as external:
        hf = csv.DictWriter(project, ["implementation", *DIMS, "route","observed","expected","native_tile_nr",
                            "nested_hk_entries","nested_fused","nested_hk_fused","nested_hk_children", *EXACT,"passed","error"])
        ff = csv.DictWriter(external, ["implementation", *DIMS, "leaf","leaf_calls","padded","leaf_evidence",*EXACT,"passed","error"])
        hf.writeheader()
        ff.writeheader()
        for shape in shapes:
            m,n,k,pad = shape
            matrices = [parent.fixture.Matrix(r,c,pad) for r,c in ((m,k),(k,n),(m,n))]
            for name in names:
                row = dict(implementation=name, **dict(zip(DIMS,shape)))
                try:
                    row.update(check(name,shape,args.vendor if name in FAST else args.plugins,matrices))
                except (OSError,RuntimeError,ValueError,AttributeError) as exc:
                    row.update(passed=0,error=str(exc))
                table, file = (ff,external) if name in FAST else (hf,project)
                table.writerow(row)
                file.flush()
                if row["passed"] != 1:
                    print(f"{name} {shape}: {row['error']}",file=sys.stderr)
                    failed += 1
    if failed:
        raise SystemExit(f"{failed} routes failed checks")


if __name__ == "__main__":
    main()
