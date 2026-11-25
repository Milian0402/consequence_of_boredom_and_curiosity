"""Check public products, original matrix regions and all vendor kernel calls."""
import argparse
import csv
import ctypes as ct
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

DIMS = ["m", "n", "k", "pad"]
EXACT = ["output_elements", "timed_binary_exact", "trace_binary_exact",
         "timed_inputs_unchanged", "trace_inputs_unchanged"]
FIELDS = ["implementation", *DIMS, "provider", "provider_identity", "core_m", "core_n", "core_k",
          "beta0", "beta1", "expected_beta0", "expected_beta1", "core_calls", "edge_regions",
          "internal_core_shapes_ok", "edge_pointers_ok", "cpu", "threads", *EXACT, "passed", "error"]


def fixture(plugins):
    helpers = Path((plugins / ".vendor-public" / "helpers.path").read_text().strip())
    spec = importlib.util.spec_from_file_location("parent_fixture", helpers / "hk-parent-routes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probes(plugins, name, helper):
    plain, trace = [helper.fixture.load_method((plugins / f"{name}{tag}.so").resolve()) for tag in ("", ".trace")]
    assert not hasattr(plain, "rank_vendor_leaf_count") and not hasattr(plain, "rank_vendor_leaf_field")
    ids = []
    for dll in (plain, trace):
        dll.rank_vendor_provider.argtypes, dll.rank_vendor_provider.restype = [], ct.c_char_p
        ids.append(dll.rank_vendor_provider().decode())
    provider = name.split("-")[1]
    expected = {"mkl": "libmkl_intel_lp64.so.3", "armpl": "libarmpl_lp64.so", "openblas": "OpenBLAS"}
    assert ids[0] == ids[1]
    assert ids[0].startswith(expected[provider]) if provider == "openblas" else Path(ids[0]).name == expected[provider]
    trace.rank_vendor_leaf_count.argtypes, trace.rank_vendor_leaf_count.restype = [ct.c_int], ct.c_uint
    trace.rank_vendor_leaf_field.argtypes, trace.rank_vendor_leaf_field.restype = [ct.c_uint, ct.c_uint], ct.c_uint64
    return plain, trace, provider, ids[0]


def check(helper, pair, name, shape):
    plain, trace, provider, identity = pair
    m, n, k, pad = shape
    matrices = (helper.fixture.Matrix(m,k,pad), helper.fixture.Matrix(k,n,pad), helper.fixture.Matrix(m,n,pad))
    a, b, c = matrices
    levels = 1 if name.endswith("-one") else 2
    split = 1 << levels
    aligned = name.endswith("-aligned-two")
    amn, ak = (64,32) if aligned else (split,split)
    cm, cn, ck = m-m%amn, n-n%amn, k-k%ak
    overwrites = 7**levels + int(cm!=m) + int(cn!=n)
    additions = int(aligned and ck!=k)
    affinity = os.sched_getaffinity(0)
    assert len(affinity) == 1 and len(list(Path("/proc/self/task").iterdir())) == 1
    pre = [trace.rank_vendor_leaf_count(i) for i in range(3)]
    exact = helper.outputs(*matrices,m,n,k,plain,trace)
    post = [trace.rank_vendor_leaf_count(i) for i in range(3)]
    zeros, ones, total = [after-before for after,before in zip(post,pre)]
    tasks = len(list(Path("/proc/self/task").iterdir()))
    assert affinity == os.sched_getaffinity(0) and tasks == 1
    ops = [[trace.rank_vendor_leaf_field(i,j) for j in range(11)] for i in range(pre[2],post[2])]
    core = 7**levels
    core_ok = len(ops) >= core and all(r[:3]==[cm//split,cn//split,ck//split] and
                r[3]>=r[2] and r[4]>=r[1] and r[5]>=r[1] and r[9:]==[0,1] for r in ops[:core])
    expected_edges = []
    def edge(label, dims, offsets, beta):
        addresses = [v.address + o*ct.sizeof(ct.c_float) for v,o in zip(matrices, offsets)]
        expected_edges.append((label, [*dims,a.ld,b.ld,c.ld,*addresses,beta,1]))
    if additions:
        edge("depth", (cm,cn,k-ck), (ck,ck*b.ld,0), 1)
    if n!=cn:
        edge("right", (cm,n-cn,k), (0,cn,cn), 0)
    if m!=cm:
        edge("bottom", (m-cm,n,k), (cm*a.ld,0,cm*c.ld), 0)
    edges_ok = ops[core:] == [row for _,row in expected_edges]
    passed = (all(exact[key] == 1 for key in EXACT[1:]) and exact["output_elements"]==m*n and
              (zeros,ones,total) == (overwrites,additions,overwrites+additions) and core_ok and edges_ok)
    return dict(implementation=name, **dict(zip(DIMS,shape)), provider=provider, provider_identity=identity,
                core_m=cm,core_n=cn,core_k=ck,beta0=zeros,beta1=ones,expected_beta0=overwrites,
                expected_beta1=additions,core_calls=core,edge_regions="|".join(s for s,_ in expected_edges),
                internal_core_shapes_ok=int(core_ok),edge_pointers_ok=int(edges_ok),cpu=next(iter(affinity)),
                threads=tasks,**exact,passed=int(passed),error="" if passed else "route, output or guard mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", type=Path)
    parser.add_argument("plugins", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method")
    args = parser.parse_args()
    with args.panel.open(newline="") as f:
        shapes = [tuple(int(r[d]) for d in DIMS) for r in csv.DictReader(f)]
    providers = ["openblas", {"x86_64":"mkl","aarch64":"armpl"}[platform.machine()]]
    methods = [f"cob-{p}-{v}" for p in providers for v in ("one","two","aligned-two")]
    if args.method:
        assert args.method in methods
        helper = fixture(args.plugins)
        pair = probes(args.plugins,args.method,helper)
    failed = 0
    with args.output.open("w",newline="") as out:
        write = csv.DictWriter(out, fieldnames=FIELDS)
        write.writeheader()
        if args.method:
            for shape in shapes:
                row = check(helper,pair,args.method,shape)
                write.writerow(row)
                failed += 1-int(row["passed"])
        else:
            for name in methods:
                part = args.plugins / ".vendor-public" / (name+"-checks.csv")
                run = subprocess.run([sys.executable,__file__,str(args.panel),str(args.plugins),str(part),"--method",name])
                failed += int(run.returncode != 0)
                if part.is_file():
                    with part.open(newline="") as one:
                        rows = list(csv.DictReader(one))
                    write.writerows(rows)
                    failed += int(len(rows)!=len(shapes))
                else:
                    failed += 1
    if failed:
        raise SystemExit(f"{failed} failed or incomplete method/route checks")

if __name__=="__main__":
    main()
