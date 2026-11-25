"""Independent acceptance audit for an installed x86-64 COB release and its research comparison."""
import argparse
import csv
import hashlib
import json
import math
import re
import shlex
from collections import defaultdict
from pathlib import Path
from statistics import median, geometric_mean

RUN = 45695245069
BENCHMARK = "bcc56044efeb37f2ab67bd45c6dae7a42623c292"
PROTOCOL = "ee1fc6d8cf4da9021d11a09f7e0a7eaf58dc49e7d3745236a6fe9f76095dfd6f"
RELEASE = "a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6"
TREE = "18f22622de1a4b1b7aebbdb66985741aa38c6e9e"
HEADER = "08e0f18ed5e275d622840f26cd09201e9c9610c312986e731901a2567dbc7149"
PARENT = "b218beff1eaf8fd606d0934405cdc1e28ecdfe15"
RESEARCH = "b2cdc6a2985726cb1658dc7cc730b5a1c4b99680"
PANEL = "466c2b51ee6df5fd737f3f4b1977c5b4ad75cdb7ae61bbaec7bc5919530f18fd"
RESEARCH_INPUTS = ("ce8377fd1574187615e356cf59425127deacd2d532bce602c41ac932e4dff440", "9acc944c3b022ef22611a9e38c52e97c8feea49614597683505e2ab60d9d5646")
RESEARCH_PATCH = "3fd0a13fc41d9f04441fb2f58981f49354e51c2856e2512972f3883778420d43"
EXT = ("openblas", "mkl", "mkl-jit", "aocl")
BUILD = ("cmake-default", "cmake-att", "cmake-intel", "make-default", "make-att", "make-intel", "cmake-native-default", "cmake-native-on")
LAB = ("research-incumbent", "research-candidate")
METHODS = EXT + BUILD + LAB
SEEDS = (348271759, 710438957)
SLOTS = [(b, j) for b in (1, 2) for j in (1, 2, 3)]
FIX = {("random", 1, SEEDS[0]), ("random", 2, SEEDS[1]), ("cancel", 0, SEEDS[0]), ("scaled", 0, SEEDS[0])}
AXES = ("m", "n", "k", "pad")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rows(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def indexed(records, key):
    unique = {}
    for r in records:
        k = key(r)
        require(k not in unique, f"Repeated {k}")
        unique[k] = r
    return unique


def sha(data):
    return hashlib.sha256(data).hexdigest()


def good_sha(value):
    return bool(re.fullmatch("[0-9a-f]{64}", value))


def read_json(p):
    return json.loads(p.read_text())


def shape(r):
    return tuple(int(r[k]) for k in AXES)


def positive(v):
    t = float(v)
    require(math.isfinite(t) and t > 0, "Invalid time " + str(v))
    return t


def fields(raw, sep):
    return dict(x.split("=", 1) for x in raw.split(sep) if "=" in x)


def accepted(m, n, k):
    return 24 <= m <= 160 and 1024 <= n <= 4096 and 96 <= k <= 384 and n >= 8*m and m*k <= 49152


def numeric(root, panel, bounds):
    cohort = [(f,n) for f in panel for n in METHODS]
    fkey = lambda r:(r["family"], r["implementation"], r["mode"], int(r["block"]), int(r["seed"]))
    accuracy = indexed(rows(root / "accuracy.csv"), fkey)
    first = indexed(rows(root / "first-calls.csv"), fkey)
    require(accuracy.keys() == first.keys() == {(f,n,*v) for f,n in cohort for v in FIX}, "Four fixtures and matching first calls per layout/method")
    cpus = set()
    errors, t0, times = defaultdict(float), defaultdict(dict), defaultdict(dict)
    valid, exclusions, cells = defaultdict(set), [], 0
    for token, r in accuracy.items():
        f, n, mode, block, seed = token
        m, w, _, _ = panel[f]
        d = fields(r["detail"], ",")
        e = float(r["max_abs_error"])
        low, high = (float(d[k]) for k in ["signed_min_error", "signed_max_error"])
        require(shape(r) == shape(first[token]) == panel[f] and r["phase"] == "stream-release" and
                r["stress"] == str(int(mode != "random")) and r["threads"] == d["threads"] == "1" and
                r["cpu"].isdigit() and r["cpu"] == d["cpu"] and int(r["checked"]) == m*w and
                r["guard_changes"] == r["nonfinite"] == d["input_changes"] == "0", f"Incomplete FP64/guard/thread record {token}")
        require(math.isfinite(e) and e >= 0 and low <= high and math.isclose(max(-low,high),e,rel_tol=1e-10,abs_tol=1e-14), f"Error signs {token}")
        ok = e <= 0.002
        require(r["verified"] == first[token]["verified"] == str(int(ok)), f"Numerical gate {token}")
        require(d.get("onednn_impl", "") == first[token]["onednn_impl"] == "", f"Undeclared runtime {n}")
        duration = positive(d["first_sgemm_seconds"])
        require(duration == positive(first[token]["first_sgemm_seconds"]), f"First-call mismatch {token}")
        if ok:
            valid[f,n].add((mode,block,seed))
        else:
            exclusions.append(dict(family=f, implementation=n, mode=mode, block=block, error=e))
        t0[f,n][mode,block,seed] = duration
        errors[n] = max(errors[n], e)
        cpus.add(int(r["cpu"]))
        cells += m*w
    time_rows = rows(root / "timings.csv")
    encounter = [(int(r["block"]),r["family"],int(r["round"]),int(r["position"])) for r in time_rows]
    actual_schedule = [(block, f, j, position) for block in (1,2)
                       for f in (list(panel) if block == 1 else list(reversed(panel)))
                       for j in (1,2,3) for position in range(1,len(METHODS)+1)]
    require(encounter == actual_schedule, "Observed cohort traversal differs from registered block/rotation order")
    for r in time_rows:
        f, n = key = r["family"], r["implementation"]
        b, j = token = int(r["block"]), int(r["round"])
        require(key in cohort and token in SLOTS and token not in times[key] and shape(r) == panel[f], f"Unexpected timing {key} {token}")
        m,w,k,p = panel[f]
        order = list(METHODS)
        offset = (list(panel).index(f)+j-1) % len(order)
        order = order[offset:] + order[:offset]
        if (j+b) % 2: order.reverse()
        require(r["phase"] == "stream-release" and r["seed"] == str(SEEDS[b-1]) and r["stress"] == "0" and r["mode"] == "random" and
                r["threads"] == "1" and r["cpu"].isdigit() and r["resident"] in ("0","1") and
                int(r["calls"]) == max(32, min(512, math.ceil(8e7/(2*m*w*k)))) and int(r["position"]) == order.index(n)+1,
                f"Counterbalance or call batch {key} {token}")
        cpus.add(int(r["cpu"]))
        times[key][token] = positive(r["seconds"])
    require(len(cpus) == 1 and (not bounds or cpus == bounds), "Logical CPU mismatch")
    cases = indexed(rows(root / "case-eligibility.csv"), lambda r:(r["family"],r["implementation"]))
    require(set(cases) == set(cohort), "Case admission matrix")
    for pair, row in cases.items():
        ok = valid[pair] == FIX and set(times[pair]) == set(SLOTS)
        require(shape(row) == panel[pair[0]] and (int(row["verified_cases"]),int(row["warm_pairs"]),int(row["qualified"])) ==
                (len(valid[pair]), len(times[pair]), int(ok)), f"Per-case qualification {pair}")
        require(ok and row["reason"] == "all_checks_passed", f"Acceptance is not fully qualified on {pair}")
    n_warm = sum(len(v) for v in times.values())
    return times, t0, dict(errors), dict(fixtures=len(accuracy), first_calls=len(first), numerical_exclusions=exclusions, fp64_cells=cells,
            qualified_cases=len(cohort), warm_batches=n_warm, warm_calls=sum(int(r["calls"]) for r in time_rows), logical_cpu=cpus.pop())


def routes(root, panel, width):
    n_product,n_lab=0,0
    cpus=set()
    product=indexed(rows(root/"routes.csv"), lambda r:(r["family"],r["implementation"]))
    require(set(product)=={(f,n) for f in panel for n in BUILD},"Installed per-case route coverage")
    for (f,n),r in product.items():
        m,w,k,_=panel[f]
        use=width if not n.endswith("default") and accepted(m,w,k) else 0
        require(shape(r)==panel[f] and int(r["width"])==use and (int(r["avx2"]),int(r["avx512"]))==(use==32,use==64) and
                r["guards"]==r["threads"]==r["passed"]==r["inputs_unchanged"]=="1" and int(r["checked"])==m*w, f"Installed route {f} {n}")
        n_product+=2*m*w
    lab=indexed(rows(root/"research-routes.csv"),lambda r:(r["family"],r["implementation"]))
    research_methods={"stream-native","stream-incumbent","stream-avx2","stream-avx2-incumbent"}
    if width==64:research_methods.update(("stream-avx512","stream-avx512-incumbent"))
    require(set(lab)=={(f,n) for f in panel for n in research_methods},"Original research route coverage")
    for (f,n),r in lab.items():
        m,w,k,_=panel[f]
        use="incumbent" not in n and accepted(m,w,k)
        old=not use and m>=64 and k>=128
        blocked=not use and not old
        nr=(32 if "avx2" in n else width) if use else 0
        mr={0:0,32:3,64:6}[nr]
        packs=math.ceil(w/nr) if use else 0
        require(shape(r)==panel[f] and tuple(int(r[z]) for z in ("selected","mr","nr","packs","tiles","old_medium","blocked"))==
                (use,mr,nr,packs,packs*math.ceil(m/mr) if use else 0,old,blocked) and int(r["linked_checked"])==int(r["trace_checked"])==m*w and
                r["guards"]==r["threads"]==r["passed"]==r["inputs_unchanged"]=="1" and r["cpu"].isdigit(),f"Research route {f} {n}")
        n_lab+=2*m*w
        cpus.add(int(r["cpu"]))
    return {"installed_cases":len(product),"installed_exact_outputs":n_product,"research_cases":len(lab),"research_exact_outputs":n_lab}, cpus


def results(panel, times, first):
    five=[f for f in panel if f.startswith("fresh-")]
    require(len(five)==5,"Five historical holdouts")
    comparisons={}
    def compare(f,new,old):
        rs=[times[f,old][t]/times[f,new][t] for t in SLOTS]
        cold=[first[f,old]["random",b,s]/first[f,new]["random",b,s] for b,s in enumerate(SEEDS,1)]
        return dict(candidate_us=1e6*median(times[f,new].values()),peer=old,peer_us=1e6*median(times[f,old].values()),
                    median=median(rs),wins=sum(r>1 for r in rs),ratios=rs,seed_medians=[median(rs[:3]),median(rs[3:])],first_random_ratios=cold)
    for new,old in [("cmake-att","cmake-default"),("cmake-native-on","research-candidate"),("cmake-native-on","cmake-native-default"),
                    ("cmake-native-on","research-incumbent"),("make-att","make-default"),("cmake-intel","cmake-att"),("make-intel","make-att"),("research-candidate","research-incumbent")]:
        matched={f:compare(f,new,old) for f in panel}
        m=[matched[f]["median"] for f in five]
        group=dict(geometric_mean=geometric_mean(m),minimum=min(m),faster_layouts=sum(x>1 for x in m),
                   seed_gms=[geometric_mean(matched[f]["seed_medians"][i] for f in five) for i in (0,1)],
                   paired_wins=sum(matched[f]["wins"] for f in five))
        comparisons[new+"_vs_"+old]=dict(holdout_replay=group,layouts=matched)
    leaders={}
    for new in ("cmake-att","cmake-native-on","research-candidate"):
        leaders[new]={}
        for f in panel:
            strongest=min(EXT,key=lambda n:(median(times[f,n].values()),n))
            peers={n:compare(f,new,n) for n in EXT}
            full=min(METHODS,key=lambda n:(median(times[f,n].values()),n))
            leaders[new][f]=dict(best_direct=peers[strongest],direct=peers,fastest_method=full)
    gen=comparisons["cmake-att_vs_cmake-default"]["holdout_replay"]
    parity=comparisons["cmake-native-on_vs_research-candidate"]["holdout_replay"]
    return dict(comparisons=comparisons,leaders=leaders,gates=dict(
        generic_speed=gen["geometric_mean"]>=1.03 and gen["minimum"]>=.98 and gen["faster_layouts"]>=4 and min(gen["seed_gms"])>1,
        matched_native_fidelity=parity["geometric_mean"]>=.98 and parity["minimum"]>=.95))


def compile_flags(root, method):
    if method.startswith("cmake-"):
        flags = read_json(root / (method + "-flags.json"))
    else:
        targets = {"gemm.o":"public","stream_b_avx2.o":"avx2","stream_b_avx512.o":"avx512"}
        flags = {}
        for line in (root / (method + "-commands.txt")).read_text().splitlines():
            op = shlex.split(line)
            target = targets.get(Path(op[op.index("-o")+1]).name)
            if not target:
                continue
            require(target not in flags, f"Duplicate archive member compilation {method} {target}")
            filename = "gemm.c" if target == "public" else "cob_stream_b_kernel.c"
            require(Path(op[op.index("-c")+1]).name == filename, f"Make source target {method}")
            flags[target] = [a for a in op if a.startswith(("-D","-m","-O"))]
    on = not method.endswith("default")
    native = "-native-" in method
    dialect = "intel" if method.endswith("intel") else "att"
    require(set(flags) == ({"public","avx2","avx512"} if on else {"public"}), f"Archive target membership {method}")
    for obj, opts in flags.items():
        arches = [v[7:] for v in opts if v.startswith("-march=")]
        require(arches and arches[-1] == ("native" if native and obj=="public" else "x86-64") and
                (native or "native" not in arches) and "-masm="+dialect in opts and "-O3" in opts, f"Target architecture/optimization {method} {obj}")
        widths = {v for v in opts if v.startswith(("-mavx","-mfma","-msse","-mno-avx"))}
        if obj == "public":
            require(not widths and ("-DCOB_ENABLE_X86_STREAM_B=1" in opts) == on, f"Generic dispatcher ISA/switch {method}")
            for target in ("PORTABLE_STRASSEN","RECURSIVE_STRASSEN","X86_MEDIUM_ROUTE"):
                require(("-DCOB_ENABLE_"+target+"=1" in opts)==native, f"Research-tuned switch in {method}")
        else:
            wide = obj=="avx512"
            require(widths=={"-mfma","-mavx512f" if wide else "-mavx2"} and
                    "-DCOB_STREAM_B_AVX512="+str(int(wide)) in opts, f"Isolated kernel ISA {method} {obj}")
    return flags


def build_proofs(root, s, width, ranking):
    release_manifest_sha="31c88bd81cea91d33400d4c7cbeadb41ebef749bc6db081e28af17bde1e583ef"
    require(s["release_tree"]==TREE and sha((root/"production-files.sha256").read_bytes())==s["source_manifest_sha256"]==release_manifest_sha and
            (root/"release-revision.txt").read_text().strip()==RELEASE, "Exact 22-file production snapshot, full release tree and revision")
    lab = read_json(root/"research/sources.json")
    canonical=lambda d:sha(json.dumps(d,sort_keys=True,separators=(",",":")).encode())
    require(lab["base"]==PARENT and lab["patch"]==RESEARCH_PATCH and
            (canonical(lab["original"]),canonical(lab["candidate"]))==RESEARCH_INPUTS, "Frozen 30/31-file research object inputs")
    timed = indexed(rows(root/"timed-binaries.csv"), lambda r:r["implementation"])
    products=indexed(rows(root/"product-builds.csv"),lambda r:r["implementation"])
    linked=indexed(rows(root/"linked-products.csv"),lambda r:r["implementation"])
    research=indexed(rows(root/"research/builds.csv"),lambda r:r["implementation"])
    tests=indexed([fields(line," ") for line in (root/"generic-build-tests.txt").read_text().splitlines() if line],lambda r:r["method"])
    require(set(timed)==set(METHODS) and set(linked)==set(products)==set(tests)==set(BUILD),"Ordinary, archive and test inventory")
    fingerprints={}; flags={}; guarded={}
    original_name = dict(zip(LAB,("stream-incumbent","stream-native")))
    for n in METHODS:
        mode, path = ranking["implementations"][n]
        key=original_name.get(n,n)
        filename=key+(".so" if n in EXT else ".worker")
        h=timed[n]["sha256"]
        require(good_sha(h) and mode==("direct" if n in EXT else "linked-public") and Path(path).name==filename,
                f"Timed ABI/path/digest {n}")
        fingerprints[n] = h
        if n in LAB:
            rec=research[key]
            v="original" if n=="research-incumbent" else "candidate"
            require(rec["worker_sha256"]==h and rec["source_sha256"]==lab[v]["src/gemm.c"] and rec["status"]=="qualified" and
                    ("-DCOB_ENABLE_X86_STREAM_B=1" in rec["flags"].split())==(v=="candidate") and "-march=native" in rec["flags"].split(),
                    f"Frozen research regular executable {n}")
        if n not in BUILD:
            continue
        on=not n.endswith("default"); cmake=n.startswith("cmake")
        option="ON" if on else "OFF"; dialect="intel" if n.endswith("intel") else "att"
        l,b=linked[n],products[n]
        require(b["build_system"]==("cmake" if cmake else "make-objects") and b["option"]==option and b["dialect"]==dialect and
                b["public_symbol"]==l["public_symbol"]=="cob_sgemm_rowmajor" and l["header_sha256"]==HEADER and
                h==b["worker_sha256"]==l["worker_sha256"] and good_sha(l["trace_sha256"]) and h!=l["trace_sha256"] and
                good_sha(l["build_archive_sha256"]) and l["build_archive_sha256"]==l["linked_archive_sha256"]==b["archive_sha256"] and
                int(l["trace_wrappers"])==2*int(on) and b["has_avx2"]==b["has_avx512"]==str(int(on)), f"Installed ABI, archive or wrappers {n}")
        require(tests[n]["test"]=="pass" and tests[n]["option"]==option and tests[n]["dialect"]==dialect, f"Production CTest/Make pass {n}")
        if on:
            g=tests[n]
            worst=float(g["max_abs_error"])
            require(int(g["native_width"])==width and int(g["AVX2_width"])==32 and
                    int(g["checked"])==576909 and 0<=worst<=.002, f"Width, guard pages, fallback and allocation tests {n}")
            guarded[n]={"checked":int(g["checked"]),"max_abs_error":worst}
        flags[n]=compile_flags(root,n)
        libs=(root/(n+"-libraries.txt")).read_text()
        require("libc.so" in libs and not re.search(r"lib(?:openblas|blas|mkl|gomp|iomp|dnnl|blis)|not found",libs,re.I),
                f"Standalone timed executable linkage {n}")
    for optout,optin in [("cmake-default","cmake-att"),("make-default","make-att"),("cmake-native-default","cmake-native-on")]:
        require(fingerprints[optout]!=fingerprints[optin] and products[optout]["archive_sha256"]!=products[optin]["archive_sha256"],"Option did not change binary")
        remove=lambda z:[x for x in z if x!="-DCOB_ENABLE_X86_STREAM_B=1"]
        require(remove(flags[optout]["public"])==remove(flags[optin]["public"]),f"More than opt-in differed in public compilation {optin}")
    for original in ("stream-native","stream-avx2","stream-avx512"):
        if research[original]["status"]=="qualified":
            receipt=(root/"research"/(original+"-fixtures.txt")).read_text()
            found=re.search(r"PASS tile=(\d+)x(\d+) checked=(\d+) max_abs_error=(\S+)",receipt)
            isa=research[original]["isa"]
            require(found and tuple(map(int,found.groups()[:3]))==((6,64,435666) if isa=="avx512" else (3,32,435666))
                    and 0<=float(found[4])<=.002,f"Research protected fixture {original}")
    return {"timed_sha256":fingerprints,"installed_archives":{n:r["linked_archive_sha256"] for n,r in linked.items()},
            "guarded_dispatch_fixtures":guarded,"compiler_targets":flags}


def provider_proofs(root):
    proof={}
    for n in ("mkl","mkl-jit"):
        info = (root/(n+"-build.txt")).read_text()
        require("vendor="+n in info and "version=2026.2.1" in info and "oneMKL LP64 sequential" in info and
                "libmkl_intel_lp64" in info and "libmkl_sequential" in info, f"Sequential oneMKL build and runtime {n}")
        if n=="mkl-jit":
            require("mkl_sgemm_jit" in info and "sgemm_direct" in info, "Dual Intel direct-call dispatch import")
        proof[n] = {"version":"2026.2.1","api":"MKL_DIRECT_CALL_SEQ_JIT" if n=="mkl-jit" else "cblas_sgemm"}
    aocl=(root/"aocl-configuration.txt").read_text()
    require("release_tag=6.0.0" in aocl and "source_sha=e5f75c24bdb8670ff648da7e483eb7e51b28a1db" in aocl and
            "threading_compiled=0" in aocl and "threads=1" in aocl,"Sequential AOCL release evidence")
    proof["aocl"]={"version":"6.0.0","source":"e5f75c24bdb8670ff648da7e483eb7e51b28a1db"}
    version=(root/"toolchain.txt").read_text().strip().splitlines()[-1]
    require(re.fullmatch(r"\d+\.\d+\.\d+.*",version),"Distro OpenBLAS version")
    proof["openblas"]={"pkg_config_version":version}
    runtime=next((p for p in (root/"peer-runtime.txt",root/"worker.log") if p.exists()),None)
    if runtime:
        lines={line.strip() for line in runtime.read_text().splitlines() if line.startswith("openblas_core=")}
        require(lines and all(re.search(r" threads=1$",line) for line in lines),"OpenBLAS provider or single-thread control")
        proof["openblas"]["measured_core"]=sorted({line.split()[0].split("=",1)[1] for line in lines})
    return proof


def verify(root, benchmark=None, run=None):
    s=read_json(root/"sources.json")
    phase="stream-release"
    require(s["base"]==PARENT and s["release"]==RELEASE and s["research"]==RESEARCH and
            sha((root/"protocol.md").read_bytes())==s["protocol_sha256"]==PROTOCOL and s["panel_sha256"]==PANEL==sha((root/"panel.csv").read_bytes()),
            "Production, benchmark, panel and protocol registration")
    require(s["benchmark"]==BENCHMARK and (benchmark is None or s["benchmark"]==benchmark) and
            int(s["github_attempt"])>0 and int(s["github_run"])==RUN and (run is None or int(s["github_run"])==run),"CI run identity")
    feats=(root/"native-features.txt").read_text().split()
    require(feats in (["1","0"],["1","1"]),"Unsupported CPU")
    width=64 if feats[1]=="1" else 32
    host=fields((root/"host.txt").read_text().replace(":","="),"\n")
    require(host["Architecture"].strip()=="x86_64","Host architecture")
    ranking=read_json(root/"ranking.json")
    require(ranking["phase"]==phase and ranking["full_output_tolerance"]==.002 and ranking["seeds"]==list(SEEDS) and ranking["rounds_per_seed"]==3 and
            tuple(ranking["implementations"])==METHODS and ranking["panel_sha256"]==PANEL,"Frozen method order, seeds and ranking")
    panel={r["family"]:shape(r) for r in rows(root/"panel.csv")}
    identities=build_proofs(root,s,width,ranking)
    inspected,cpus=routes(root,panel,width)
    times,first,errors,coverage=numeric(root,panel,cpus)
    require(coverage["logical_cpu"]==s["cpu"],"Selected CPU differs from pinned workers")
    return dict(phase=phase,host=host["Model name"].strip(),cpu=coverage["logical_cpu"],kernel_width=width,production=RELEASE,
                benchmark=s["benchmark"],run=int(s["github_run"]),attempt=int(s["github_attempt"]),research=RESEARCH,applicable_external=list(EXT),
                coverage={**coverage,**inspected},max_abs_errors=errors,providers=provider_proofs(root),builds=identities,**results(panel,times,first))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw",type=Path)
    cli.add_argument("--benchmark")
    cli.add_argument("--run",type=int)
    args=cli.parse_args()
    print(json.dumps(verify(args.raw,args.benchmark,args.run),indent=2,allow_nan=False))
