"""Qualify the ordinary linked public executable and its CBLAS trace companion."""
import argparse
import csv
import ctypes as ct
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance

study = Path(__file__).resolve().parent.parent
routes = module("core_routes", study / "vendor-public" / "check-routes.py")
helper = module("integer_fixture", study / "hk-parent-routes.py")
LINKED = ["linked_worker_exact", "linked_worker_inputs_unchanged", "linked_worker_outputs_checked"]
FIELDS = [*routes.FIELDS, *LINKED]


def probes(plugins, name):
    provider = "openblas" if name == "release-previous" else name.split("-")[1]
    plain, trace = [helper.fixture.load_method((plugins/f"{name}{tag}.so").resolve()) for tag in ("", ".trace")]
    if any(hasattr(plain, sym) for sym in ("rank_vendor_leaf_count","rank_vendor_leaf_field")):
        raise ValueError("diagnostics exported by ordinary adapter")
    ids = []
    for dll in (plain,trace):
        dll.rank_vendor_provider.argtypes, dll.rank_vendor_provider.restype = [],ct.c_char_p
        ids.append(dll.rank_vendor_provider().decode())
    if ids[0] != ids[1]:
        raise ValueError(f"companion provider mismatch: {ids!r}")
    if provider == "openblas":
        valid = ids[0].startswith("OpenBLAS")
    else:
        base = {"mkl":"libmkl_intel_lp64.so","armpl":"libarmpl_lp64.so"}[provider]
        label = Path(ids[0]).name
        valid = label == base or label.startswith(base+".")
    if not valid:
        raise ValueError(f"unexpected {provider} link: {ids[0]}")
    trace.rank_vendor_leaf_count.argtypes, trace.rank_vendor_leaf_count.restype = [ct.c_int],ct.c_uint
    trace.rank_vendor_leaf_field.argtypes, trace.rank_vendor_leaf_field.restype = [ct.c_uint,ct.c_uint],ct.c_uint64
    return plain,trace,provider,ids[0]


def linked(plugins, name, shape):
    m,n,_,_ = shape
    run = subprocess.run([str((plugins/f"{name}.worker").resolve()),"fixture",*map(str,shape)],
                         text=True, capture_output=True, check=True)
    rows = [line for line in run.stdout.splitlines() if line.startswith("FIXTURE,")]
    if len(rows) != 1: raise ValueError(f"missing unique linked fixture row for {name}: {rows!r}")
    data = {key:int(value) for key,value in (item.split("=") for item in rows[0].split(",")[1:])}
    expected = {"checked":m*n,"output_exact":1,"inputs_unchanged":1,"guards":1,"threads":1}
    if data != expected: raise ValueError(f"linked fixture failed for {name}: {data!r}")
    return dict(zip(LINKED, (data["output_exact"],data["inputs_unchanged"],data["checked"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("panel",type=Path)
    parser.add_argument("plugins",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--method")
    args = parser.parse_args()
    native = {"aarch64":"armpl","x86_64":"mkl"}[platform.machine()]
    methods = ["release-previous",*[f"release-{p}-{v}" for p in ("openblas",native) for v in ("two","aligned-two")]]
    with args.panel.open(newline="") as f:
        shapes = [tuple(int(r[k]) for k in routes.DIMS) for r in csv.DictReader(f)]
    failed = False
    with args.output.open("w",newline="") as out:
        write=csv.DictWriter(out,fieldnames=FIELDS)
        write.writeheader()
        if args.method:
            if args.method not in methods or len(os.sched_getaffinity(0)) != 1: raise ValueError("unexpected method or CPU affinity")
            try:
                pair = probes(args.plugins,args.method)
            except (OSError,AttributeError,RuntimeError,ValueError) as exc:
                pair = exc
            for shape in shapes:
                try:
                    if isinstance(pair,Exception): raise pair
                    row = routes.check(helper,pair,args.method,shape)
                    row.update(linked(args.plugins,args.method,shape))
                except (AssertionError,OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
                    row = dict(implementation=args.method, **dict(zip(routes.DIMS,shape)),passed=0,error=str(exc))
                write.writerow(row)
                failed |= row["passed"] != 1
        else:
            # New processes keep all ordinary BLAS bindings independent.
            for name in methods:
                part = args.plugins/".release-public"/(name+"-routes.csv")
                sub = subprocess.run([sys.executable,__file__,str(args.panel),str(args.plugins),str(part),"--method",name])
                failed |= sub.returncode != 0
                if not part.is_file():
                    failed = True
                    continue
                with part.open(newline="") as one:
                    rows=list(csv.DictReader(one))
                write.writerows(rows)
                failed |= len(rows)!=len(shapes)
    if failed: raise SystemExit("linked public or native leaf fixture failed; see route CSV")

if __name__=="__main__": main()
