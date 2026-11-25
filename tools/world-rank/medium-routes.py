"""Exercise the actual public FP32 entries and prove the observed native tile per layout."""
import argparse
from array import array
import csv
import ctypes as ct
import importlib.util
import math
import os
from pathlib import Path
import platform
import random
import sys

OLD = "medium-incumbent"
NEW = "medium-transposed"
COUNTER = "cob_medium_transpose_count"
OLD_COUNTER = "rank_old_medium_successes"
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("parent_routes", HERE / "hk-parent-routes.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
DIMS = fixture.DIMENSIONS
COUNT_FIELDS = ["selected", "mr", "nr", "m_tiles", "n_tiles", "k_panels"]
FIELDS = ["implementation", *DIMS, *COUNT_FIELDS, "expected", "old_medium_calls", "old_medium_expected",
          "control_seed","seeded_control_equal","cpu", "threads", *fixture.EXACT, "passed", "error"]


def expected_counts(dims, machine, features):
    m,n,k,_ = dims
    eligible = 48 <= m <= 256 and 512 <= n <= 4096 and 96 <= k <= 768 and n >= 4*m and m*k <= 98304
    if machine == "aarch64" and "asimd" in features:
        mr,nr = 8,12
    elif machine == "x86_64" and {"avx512f", "fma"} <= features:
        mr,nr = 32,12
    elif machine == "x86_64" and {"avx2", "fma"} <= features:
        mr,nr = 16,6
    else:
        eligible = False
    if not eligible:
        return [0] * 6
    return [1,mr,nr,(m+mr-1)//mr,(n+nr-1)//nr,(k+95)//96]


def engines(plugins, name):
    plain = fixture.fixture.load_method((plugins/f"{name}.so").resolve())
    if hasattr(plain,COUNTER) or hasattr(plain,OLD_COUNTER):
        raise RuntimeError("timed entry exposes trace counters")
    trace = fixture.fixture.load_method((plugins/f"{name}.trace.so").resolve())
    old = getattr(trace,OLD_COUNTER)
    old.argtypes, old.restype = [],ct.c_int
    get = getattr(trace,COUNTER,None)
    if name == NEW:
        if get is None:
            raise ValueError("missing candidate route counter")
        get.argtypes, get.restype = [ct.c_int], ct.c_int
    elif get is not None:
        raise ValueError("incumbent exposes candidate diagnostics")
    return plain,trace,get,old


def seed_check(shape,seed,incumbent,candidate):
    m,n,k,pad = shape
    rng = random.Random(seed)
    pos = 16
    guard = array("I",[0x7fc4a93b]).tobytes()
    def allocate(r,c,inputs):
        stride = c+pad
        data = bytearray(guard*((pos*2)+r*stride))
        if inputs:
            for i in range(r):
                at = (pos+i*stride)*4
                data[at:at+4*c] = array("f",(rng.uniform(-1,1) for _ in range(c))).tobytes()
        return data,ct.cast(ct.addressof(ct.c_char.from_buffer(data))+pos*4,fixture.fixture.FP)
    a,ap = allocate(m,k,True)
    b,bp = allocate(k,n,True)
    saved_a,saved_b = bytes(a),bytes(b)
    results = []
    for engine in (incumbent,candidate):
        c,cp = allocate(m,n,False)
        engine.rank_vendor_sgemm(m,n,k,ap,k+pad,bp,n+pad,cp,n+pad)
        if saved_a!=a or saved_b!=b or c[:4*pos]!=guard*pos or c[-4*pos:]!=guard*pos:
            return False
        for i in range(m):
            start = 4*(pos+i*(n+pad))
            at = start+4*n
            if c[at:at+4*pad]!=guard*pad or not all(map(math.isfinite,memoryview(c)[start:at].cast("f"))):
                return False
        results.append(c)
    return results[0] == results[1]


def check_pair(name, shape, pair, features):
    m,n,k,pad = shape
    plain,trace,get,old = pair
    matrices = [fixture.fixture.Matrix(rows,cols,pad) for rows,cols in ((m,k),(k,n),(m,n))]
    before = old()
    data = fixture.outputs(*matrices,m,n,k,plain,trace)
    old_calls = old()-before
    counts = [get(i) for i in range(6)] if get else [0]*6
    want = expected_counts(shape,platform.machine(),features) if name == NEW else [0]*6
    cpus = os.sched_getaffinity(0)
    tasks = len(os.listdir("/proc/self/task"))
    old_wanted = int(not want[0] and platform.machine()=="x86_64" and "fma" in features and
                     bool({"avx2","avx512f"} & features) and 64<=m<=256 and 128<=n<=4096 and 128<=k<=1024
                     and n>=m and (2*n>=3*m or 2*k>=3*m) and (m>128 or n!=128 or k!=128))
    passed = counts == want and old_calls == old_wanted and len(cpus)==1 and tasks==1 and all(data[key] == 1 for key in fixture.EXACT[1:])
    data.update(zip(COUNT_FIELDS,counts))
    data.update(expected="|".join(map(str,want)),cpu=next(iter(cpus)) if len(cpus)==1 else "",
                old_medium_calls=old_calls,old_medium_expected=old_wanted,threads=tasks,
                passed=int(passed),error="" if passed else "tile, original dispatch, exact output, input, guard or affinity mismatch")
    return data


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("panel","plugins","output"):
        cli.add_argument(name,type=Path)
    args = cli.parse_args()
    cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
    features = next((set(line.partition(":")[2].split()) for line in cpuinfo
                     if line.partition(":")[0].strip() in ("flags","Features")), set())
    with args.panel.open(newline="") as f:
        cases = [(r["family"],tuple(int(r[k]) for k in DIMS)) for r in csv.DictReader(f)]
    failed = False
    with args.output.open("w",newline="") as out:
        writer = csv.DictWriter(out,FIELDS)
        writer.writeheader()
        for name in (OLD,NEW):
            try:
                pair = engines(args.plugins,name)
            except (AttributeError,OSError,RuntimeError) as error:
                pair = error
            for family,shape in cases:
                row = dict(zip(DIMS,shape),implementation=name)
                try:
                    if isinstance(pair,Exception):
                        raise pair
                    row.update(check_pair(name,shape,pair,features))
                    if family.startswith("control-") and name == NEW:
                        incumbent = fixture.fixture.load_method((args.plugins/f"{OLD}.so").resolve())
                        ok = seed_check(shape,68539041,incumbent,pair[0])
                        row.update(control_seed=68539041,seeded_control_equal=int(ok))
                        if not ok:
                            row.update(passed=0,error="seeded fallback differed or wrote to inputs, padding or guards")
                except (AttributeError,OSError,RuntimeError,ValueError) as error:
                    row.update(passed=0,error=str(error))
                writer.writerow(row)
                out.flush()
                failed |= row["passed"] != 1
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
