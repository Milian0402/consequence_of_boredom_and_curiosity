"""Check the public HK entry, packed views and external FastMMW on the fixed panel."""
import argparse
import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

BASE = "cob-hk15-parent-960"
NEW = "hk-packed-forms"
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("leaf_routes", HERE / "leaf-wave-routes.py")
leaf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaf)
PANEL = [(2048,2880,3264,3), (2034,3189,2997,11), (1964,2895,3162,5), (3189,2034,2997,11)]
HK_FIELDS = ["route", "observed", "expected", "packed_products", "packed_views", "original_forms"]
FAST_FIELDS = ["leaf", "leaf_calls", "padded", "leaf_evidence"]


def project(name, shape, directory, matrices):
    m,n,k,_ = shape
    engine, trace = [leaf.parent.fixture.load_method((directory/f"{name}{suffix}.so").resolve())
                     for suffix in ("", ".trace")]
    if any(hasattr(engine, symbol) for symbol in [*leaf.SYMBOLS, "rank_hk_packed_forms"]):
        raise ValueError("timed binary exposes diagnostics")
    if hasattr(trace, "rank_hk_packed_forms") != (name == NEW):
        raise ValueError("packed counter ownership differs from the frozen method")
    data = leaf.parent.outputs(*matrices,m,n,k,engine,trace)
    h,w,d = m//2,n//3,k//3
    active = (m%2 == n%3 == k%3 == 0 and min(h,w,d)>=768 and max(h,w,d)<=1280
              and 4*max(h,w,d)<=5*min(h,w,d))
    area = h*w + h*d + w*d
    fused = active and 3*h*w*d >= 960*area
    scoped = fused and 3*h*w*d < 1024*area
    expected = [15*int(active),15*int(fused),15*int(active and not fused),int(not active),15*int(scoped)]
    route = leaf.getter(trace,"rank_hk15_route_count",indexed=True)
    observed = [route(i) for i in range(5)]
    good = observed == expected and all(data[key] == 1 for key in leaf.EXACT[1:])
    data.update(route="hk-packed" if name == NEW else "hk-parent", observed="|".join(map(str,observed)),
                expected="|".join(map(str,expected)))
    if name == NEW:
        counts = leaf.getter(trace,"rank_hk_packed_forms",indexed=True)
        packed = [counts(i) for i in range(3)]
        good &= packed == [4*int(fused),28*int(fused),4*int(active and not fused)]
        data.update(zip(HK_FIELDS[-3:],packed))
    data.update(passed=int(good),error="" if good else "incorrect route, exact output, or guards")
    return data


def measure_one(args, method, columns):
    failed = False
    with args.output.open("w", newline="") as out:
        writer = csv.DictWriter(out,fieldnames=columns)
        writer.writeheader()
        for shape in PANEL:
            row = dict(zip(leaf.DIMS,shape),implementation=method)
            try:
                m,n,k,p = shape
                matrices = [leaf.parent.fixture.Matrix(r,c,p) for r,c in ((m,k),(k,n),(m,n))]
                if method in leaf.FAST:
                    row.update(leaf.check(method,shape,args.vendor,matrices))
                else:
                    row.update(project(method,shape,args.plugins,matrices))
                cpus = os.sched_getaffinity(0)
                threads = len(os.listdir("/proc/self/task"))
                if len(cpus)!=1 or threads!=1:
                    raise ValueError("route checker requires one thread and one allowed CPU")
                row.update(cpu=next(iter(cpus)),threads=threads)
            except (AttributeError, KeyError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                row.update(passed=0,error=str(error))
            writer.writerow(row)
            out.flush()
            failed |= row["passed"] != 1
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel","plugins","vendor","dest"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--method",choices=(BASE,NEW,*leaf.FAST))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    with args.panel.open(newline="") as f:
        if [tuple(int(r[k]) for k in leaf.DIMS) for r in csv.DictReader(f)] != PANEL:
            parser.error("unexpected panel geometry or ordering")
    def fields(name):
        return ["implementation",*leaf.DIMS,*(FAST_FIELDS if name in leaf.FAST else HK_FIELDS),
                "cpu","threads",*leaf.EXACT,"passed","error"]
    if args.method:
        if not args.output:
            parser.error("--method requires --output")
        return measure_one(args,args.method,fields(args.method))
    failed = False
    for filename,group in (("routes.csv",(BASE,NEW)),("fastmmw-routes.csv",tuple(leaf.FAST))):
        with (args.dest/filename).open("w",newline="") as out:
            writer = csv.DictWriter(out,fields(group[0]))
            writer.writeheader()
            for name in group:
                part = args.plugins/(name+"-routes.csv")
                result = subprocess.run([sys.executable,__file__,args.panel,args.plugins,args.vendor,args.dest,
                                         "--method",name,"--output",part])
                records = []
                if part.is_file():
                    with part.open(newline="") as inp:
                        records = list(csv.DictReader(inp))
                for shape in PANEL[len(records):]:
                    records.append(dict(zip(leaf.DIMS,shape),implementation=name,passed=0,
                                        error=f"route process exited {result.returncode} before this case"))
                writer.writerows(records)
                out.flush()
                failed |= result.returncode != 0 or len(records)!=len(PANEL)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
