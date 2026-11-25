"""Measure the two installed public archives and nine independent, strided FP32 modes."""
import csv
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

BUILD, VENDOR, OUT = (Path(p).resolve() for p in sys.argv[1:4])
CPU = int(sys.argv[4])
HERE = Path(__file__).resolve().parent
SEEDS = globals().get("seed_override", (267491683, 1893702713))
CASES = (("random", SEEDS[0]), ("random", SEEDS[1]), ("cancel", SEEDS[0]), ("scaled", SEEDS[1]))
METHODS = ("openblas", "mkl", "mkl-jit", "aocl", "blasfeo", "onednn-oneshot", "onednn-cached-plan",
           "libxsmm-dispatch", "libxsmm-retained", "stream-incumbent", "grouped-b")
METHODS = globals().get("method_override", METHODS)
SUBJECTS = globals().get("subject_override", METHODS[-2:])
ASSISTED = globals().get("assisted_override", ())
GEOMETRY = "m n k lda ldb ldc ar ac br bc cr cc".split()
MEASURED = "checked guards a_mod64 b_mod64 c_mod64 cpu threads error first_s per_call_s lda ldb ldc".split()
HANDLE_METHODS = globals().get("handle_methods", ())
RETAINED = globals().get("retained_override", ("libxsmm-retained", "onednn-cached-plan"))
PAGE = os.sysconf("SC_PAGE_SIZE")
with (OUT / "panel.csv").open() as panel:
    shapes = list(csv.DictReader(panel))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def writer(name, fields):
    handle = (OUT / name).open("w", newline="", buffering=1)
    table = csv.DictWriter(handle, fields.split(), lineterminator="\n")
    table.writeheader()
    return table


def check(ok, message):
    if not ok:
        raise ValueError(message)


def source(method):
    if method in SUBJECTS:
        return BUILD / (method + ".worker")
    if method in ASSISTED:
        return BUILD / "prefix" / (method + ".so")
    if method.startswith("onednn-"):
        return VENDOR / "onednn" / (method.replace("onednn-", "onednn-f32-") + ".so")
    return VENDOR / (method + ".so")


def fp_fields(shape, values, calls):
    m, n, _, lda, ldb, ldc, ar, ac, br, bc, cr, cc = (int(shape[key]) for key in GEOMETRY)
    guards = (((cr + m + 2) * ldc * 4 + PAGE - 1) // PAGE) * (PAGE // 4) - m*n
    residues = [(r*ld+c)*4 % 64 for r, c, ld in ((ar, ac, lda), (br, bc, ldb), (cr, cc, ldc))]
    check(int(values["checked"]) == m*n and int(values["guards"]) == guards, "incorrect extent or guards")
    check([int(values[k]) for k in ("a_mod64","b_mod64","c_mod64")] == residues, "incorrect pointer residues")
    check([int(values[k]) for k in ("lda","ldb","ldc")] == [lda,ldb,ldc], "incorrect observed strides")
    check(int(values["threads"]) == 1 and int(values["cpu"]) == CPU, "not pinned to one CPU")
    check(0 <= float(values["error"]) <= 0.002 and 0 < float(values["first_s"]) < math.inf, "invalid numeric/first call")
    elapsed = float(values["per_call_s"])
    check((calls and 0 < elapsed < math.inf) or (not calls and elapsed == 0), "invalid warm timing")


raw = writer("raw.csv", "method family layout mode seed round place calls " + " ".join(MEASURED))
methods = writer("methods.csv", "method role policy status sha256")
admit = writer("admissions.csv", "method family layout qualified reason accuracy_records timing_records")
xsmm = writer("libxsmm-routes.csv", "method family layout m n k lda ldb ldc code_bytes admitted reason")
dnn = writer("onednn-routes.csv", "method family layout mode seed round implementation plans cached")
if HANDLE_METHODS:
    handles = writer("handle-observations.csv","method family layout mode seed round first_status last_status prepared_status prepare_s")
errors = writer("failures.csv", "method family layout reason")
failures = {}
if HANDLE_METHODS:
    with (OUT/"handle-route-exclusions.csv").open() as proof:
        for item in csv.DictReader(proof):
            key=(item["method"],item["family"],item["layout"])
            check(key[0] in HANDLE_METHODS and any((s["family"],s["layout"])==key[1:] for s in shapes),
                  "invalid explicit handle exclusion")
            check(key not in failures and item["reason"], "empty or duplicate route failure")
            failures[key]="native_route_excluded: "+item["reason"]
counts = {}
xsmm_inspect = {}
paths = {method: source(method) for method in METHODS}
available = {}
for method, path in paths.items():
    available[method] = path.is_file() and (not method.startswith("onednn-") or (VENDOR / "onednn/qualified").is_file())
    if available[method] and method.startswith("libxsmm-"):
        lib = ct.CDLL(str(path))
        if lib.rank_vendor_init() == 0:
            inspect = lib.rank_xsmm_code_size
            inspect.restype = ct.c_size_t
            inspect.argtypes = [ct.c_int] * 6
            xsmm_inspect[method] = inspect
        else:
            available[method] = False
    methods.writerow(dict(method=method, role="standalone" if method in SUBJECTS else ("vendor-assisted" if method in ASSISTED else "independent"),
        policy="retained" if method in RETAINED else "one-shot",
        status="available" if available[method] else "unavailable", sha256=digest(path) if path.is_file() else ""))

receipt = dict(production="a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6",
    benchmark=subprocess.check_output(["git","-C",str(HERE),"rev-parse","HEAD"],text=True).strip(),
    run=int(os.environ["GITHUB_RUN_ID"]), attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
    replica=int(os.environ["FRONTIER_REPLICA"]), cpu=CPU, page=PAGE,
    source_files={name:digest(OUT/name) for name in globals().get("receipt_files", ("panel.csv","protocol.md","novelty.json","candidate.sha256","methods.csv"))})
(OUT/"run.json").write_text(json.dumps(receipt, indent=2)+"\n")


def invoke(shape, method, mode, seed, round_id, place, calls):
    key = (method, shape["family"], shape["layout"])
    if key in failures:
        return
    if not available[method]:
        failures[key] = "registered_binary_unavailable"
        return
    try:
        is_subject = method in SUBJECTS
        executable = paths[method] if is_subject else paths["stream-incumbent"]
        args = [str(executable), "public" if is_subject else str(paths[method]), mode, str(seed), str(calls)]
        command = subprocess.run(args + [shape[k] for k in GEOMETRY], capture_output=True, text=True)
        check(command.returncode == 0, f"worker exit {command.returncode}: {command.stderr[-320:].strip()}")
        lines = command.stdout.splitlines()
        rows = [line.split(",")[1:] for line in lines if line.startswith("VIEW,")]
        check(len(rows) == 1, "missing observation")
        values = dict(zip(MEASURED, rows[0], strict=True))
        fp_fields(shape, values, calls)
        ids = dict(method=method, family=shape["family"], layout=shape["layout"], mode=mode, seed=seed, round=round_id)
        if method in HANDLE_METHODS:
            receipts = [line.split(",")[1:] for line in lines if line.startswith("HANDLE,")]
            check(len(receipts)==1 and len(receipts[0])==4,"missing explicit SGEMM observation")
            fs,ls,ps,elapsed=receipts[0]
            check(fs in ("0","1") and ls==fs,"unsupported or inconsistent explicit SGEMM status")
            if method=="mkl-handle-prepared":
                check(ps==fs and 0<float(elapsed)<math.inf,"missing explicit handle preparation")
            else:
                check(ps=="-1" and float(elapsed)==0,"oneshot unexpectedly held a plan")
            handles.writerow(dict(ids,first_status=fs,last_status=ls,prepared_status=ps,prepare_s=elapsed))
        if method.startswith("onednn-"):
            traits = [line.rsplit(",",2) for line in lines if line.startswith("DNN,")]
            check(len(traits) == 1, "missing oneDNN measured-route receipt")
            impl, plans, cached = traits[0]
            impl = impl[4:]
            keep = int(method == "onednn-cached-plan")
            check(bool(impl) and int(cached) == keep and int(plans) == (1 if keep else calls+1), "wrong oneDNN plans or policy")
            dnn.writerow(dict(ids, implementation=impl, plans=plans, cached=cached))
        raw.writerow(dict(ids, place=place, calls=calls, **values))
        count = counts.setdefault(key, [0,0])
        count[int(round_id >= 0)] += 1
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        failures[key] = str(exc).replace("\n"," ")
        errors.writerow(dict(method=method, family=key[1], layout=key[2], reason=failures[key]))


for shape in shapes:
    for method in METHODS:
        if method.startswith("libxsmm-"):
            dims = {k:int(shape[k]) for k in GEOMETRY[:6]}
            size = xsmm_inspect[method](*dims.values()) if method in xsmm_inspect else 0
            reason = "verified_nonreference_jit" if size else "no_verified_nonreference_kernel"
            xsmm.writerow(dict(method=method,family=shape["family"],layout=shape["layout"],**dims,
                               code_bytes=size,admitted=int(bool(size)),reason=reason))
            if not size:
                failures[(method,shape["family"],shape["layout"])] = reason
        for mode, seed in CASES:
            invoke(shape, method, mode, seed, -1, -1, 0)

for r in range(6):
    order = shapes[r%3:] + shapes[:r%3]
    if r>=3:
        order.reverse()
    for shape in order:
        start = (shapes.index(shape)+2*(r%3)) % len(METHODS)
        roster = list(METHODS[start:]+METHODS[:start])
        if r>=3:
            roster.reverse()
        for place, method in enumerate(roster):
            invoke(shape, method, "random", SEEDS[r//3], r, place, 32)

for shape in shapes:
    for method in METHODS:
        key = (method,shape["family"],shape["layout"])
        counts_here = counts.get(key,[0,0])
        reason = failures.get(key,"qualified" if counts_here == [4,6] else "incomplete")
        admit.writerow(dict(method=method,family=key[1],layout=key[2],qualified=int(reason=="qualified"),
                            reason=reason,accuracy_records=counts_here[0],timing_records=counts_here[1]))
