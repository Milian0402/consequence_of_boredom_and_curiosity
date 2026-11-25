"""Verify ordinary/timed source and archive identities, actual kernel selections and bypasses."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from measurements import demand, rows, shape, single

PRIOR="frontier-incumbent"
NEW="stream-packed-a"
HEADER="08e0f18ed5e275d622840f26cd09201e9c9610c312986e731901a2567dbc7149"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def valid_sha(value):
    return bool(re.fullmatch(r"[0-9a-f]{64}",value))

def eligible(s):
    m,n,k,_=s
    return 24<=m<=160 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and m*k<=49152

def pad_shape(s):
    m,n,k,_=s
    extra=(-m)%8
    return 24<=m<=96 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and 0<extra and 8*extra<=m

def host(root):
    content=(root/"host.txt").read_text()
    def field(key):
        m=re.search(r"^"+re.escape(key)+r":\s*([^\n]+)",content,re.M)
        return m[1].strip() if m else ""
    model=field("Model name");flags=field("Flags").split()
    vendor=field("Vendor ID") or ("GenuineIntel" if "intel" in model.lower() else "AuthenticAMD" if "amd" in model.lower() else "")
    demand(field("Architecture")=="x86_64" and model and vendor in ("GenuineIntel","AuthenticAMD") and {"avx2","fma"}<=set(flags),
           "Missing native host model, vendor, AVX2 and FMA")
    features=(root/"frontier-native-features.txt").read_text().split()
    demand(features==["1",str(int("avx512f" in flags))],"Runtime ISA differs from host flags")
    return model, vendor, ("avx512" if features[-1]=="1" else "avx2")


def products(root, contract, methods, timed):
    source=json.loads((root/"frontier-sources.json").read_text())
    declared=contract["source"]
    delta=["src/cob_stream_b_dispatch.inc","src/cob_stream_b_kernel.c","src/cob_stream_packed_a.inc"]
    demand(source["release"]==contract["release"] and source["tree"]==declared["tree"] and source["patch_sha256"]==declared["patch"] and
           source["changed_paths"]==delta, "Wrong baseline, patch or production source tree")
    identities=source["methods"]
    before=identities[PRIOR]["files"];after=identities[NEW]["files"]
    demand(sorted(f for f in (set(before)|set(after)) if before.get(f)!=after.get(f))==delta and before["include/cob_gemm.h"]==after["include/cob_gemm.h"]==HEADER,
           "Changed source files or public header outside the frozen set")
    build=single(rows(root/"frontier-builds.csv"),lambda r:r["implementation"])
    demand(set(build)==set(identities)=={PRIOR,NEW}, "Incomplete ordinary/trace pair")
    flags={}
    for n,r in build.items():
        files=identities[n]["files"]
        demand(all(valid_sha(x) for x in files.values()) and all(not Path(f).is_absolute() and ".." not in Path(f).parts for f in files),
               "Invalid source file inventory")
        files_sha=sha(json.dumps(files,sort_keys=True,separators=(",",":")).encode())
        demand(files_sha==r["source_manifest_sha256"]==identities[n]["sha256"]==declared["methods"][n] and
               r["source_revision"]==contract["release"] and r["patch_sha256"]==(declared["patch"] if n==NEW else "none"),
               f"Compiler inputs differ for {n}")
        demand(r["archive_sha256"]==r["installed_archive_sha256"] and r["header_sha256"]==HEADER
               and r["public_symbol"]=="cob_sgemm_rowmajor" and r["worker_sha256"]==timed[n] and
               all(valid_sha(r[k]) for k in ("archive_sha256","trace_sha256","worker_sha256")) and r["worker_sha256"]!=r["trace_sha256"],
               f"Ordinary installed worker and separate instrumentation {n}")
        ffile=root/(n+"-flags.json")
        flags[n]=json.loads(ffile.read_text())
        demand(r["flags_sha256"]==sha(ffile.read_bytes()) and set(flags[n])=={"public","avx2","avx512"},"ISA compilation targets")
        for target,opts in flags[n].items():
            arches=[v for v in opts if v.startswith("-march=")]
            demand(arches and arches[-1]==("-march=native" if target=="public" else "-march=x86-64") and "-O3" in opts and "-masm=att" in opts,"Native compiler flags")
            widths={v for v in opts if v.startswith(("-mavx","-mfma","-mno-avx"))}
            demand(widths==(set() if target=="public" else {"-mfma","-m"+target+("f" if target=="avx512" else "")}),"Isolated ISA subset")
        methods[n].update(source_manifest_sha256=files_sha,archive_sha256=r["archive_sha256"],trace_sha256=r["trace_sha256"])
    switch="-DCOB_ENABLE_X86_PACKED_A_FRONTIER=1"
    for t in flags[NEW]:
        demand(switch not in flags[PRIOR][t] and flags[NEW][t].count(switch)==1 and
               [x for x in flags[NEW][t] if x!=switch]==flags[PRIOR][t],f"Unmatched lane flags {t}")
    demand(build[NEW]["archive_sha256"]!=build[PRIOR]["archive_sha256"] and timed[NEW]!=timed[PRIOR],"Experimental archive unchanged")
    return {"release":source["release"],"tree":source["tree"],"patch":source["patch_sha256"]}


def witness(root, panel, isa):
    gate={}; cases={}
    records=single(rows(root/"frontier-routes.csv"),lambda r:(r["family"],r["implementation"]))
    demand(set(records)=={(f,n) for f in panel for n in (NEW,PRIOR)},"Unmatched executable native routes")
    pins=set()
    counters=("avx2","avx512","packed_avx2","packed_avx512")
    def validate(f,n,r,rescue=False):
        s=panel[f];m,w,k,p=s
        selected=eligible(s) and not rescue
        active=n==NEW and selected and m<=96
        expected=dict.fromkeys(counters,0)
        if selected: expected[("packed_" if active else "")+isa]=1
        route="allocation-fallback" if rescue else "packed-a" if active else "incumbent" if selected else "bypass"
        demand(shape(r)==s and r["isa"]==isa and all(int(r[k])==v for k,v in expected.items()) and
               int(r["candidate_calls"])==int(active) and r["route"]==route and int(r["checked"])==int(r["trace_checked"])==m*w and
               all(r[k]=="1" for k in ("guards","inputs_unchanged","threads","passed")) and r["cpu"].isdigit(),
               f"Unsupported observed route {f} {n} {route}")
        pins.add(int(r["cpu"]))
        return route
    for (f,n),r in records.items():
        cases.setdefault(f,{})[n]=validate(f,n,r)
        gate[f,n]=True
    fallback=single(rows(root/"frontier-fallbacks.csv"),lambda r:r["implementation"])
    demand(set(fallback)=={NEW,PRIOR},"Both ordinary implementations must recover from failed allocation")
    for n,r in fallback.items():
        f=r["family"]
        demand(eligible(panel[f]) and panel[f][0]<=96 and r["allocation_refusals"]=="1","Forced allocation rescue")
        validate(f,n,r,True)
    demand(len(pins)==1,"Mixed route affinity")
    return gate,cases,pins.pop()


def controls(root,panel,track):
    excluded={f for f,route in track.items() if route[NEW]!="packed-a"}
    data=single(rows(root/"frontier-control-parity.csv"),lambda r:r["family"])
    demand(excluded==set(data) and len(data)==2,"Dense-output fallback controls")
    for f,row in data.items():
        m,n,k,p=panel[f]
        demand(shape(row)==panel[f] and int(row["checked"])==m*n and row["passed"]=="1" and
               valid_sha(row["incumbent_sha256"]) and row["incumbent_sha256"]==row["candidate_sha256"],
               "Dense original/candidate float output mismatch on "+f)
    return {f:v["incumbent_sha256"] for f,v in data.items()}


def peers(root,panel,methods,vendor,logical_cpu,contract):
    underlying="mkl" if vendor=="GenuineIntel" else "aocl"
    data=single(rows(root/"host-pad-routes.csv"),lambda r:(r["family"],r["implementation"]))
    wrappers={n for n in methods if n in ("host-direct","host-pad8")}
    demand(set(data)=={(f,n) for f in panel for n in wrappers},"Missing hosted provider wrapper routing")
    registry=(root/"host-pad-provenance.txt").read_text().splitlines() if wrappers else []
    demand(not wrappers or registry[0]=="provider="+underlying,"Vendor-assisted CPUID result differs")
    hashes={}
    for l in registry[1:]:
        h,n=l.split(maxsplit=1)
        demand(n not in hashes and valid_sha(h),"Invalid wrapper input manifest")
        hashes[n]=h
    for name in ({*wrappers,underlying} if wrappers else set()):
        demand(hashes[name+".so"]==methods[name]["sha256"],"Wrong wrapper or forwarded provider binary "+name)
    demand(not wrappers or hashes["panel.csv"]==sha((root/"panel.csv").read_bytes()),"Wrapper table differs")
    demand(not wrappers or all(hashes[n]==h for n,h in contract["source"]["assisted_files"].items()),
           "Wrapper/proxy/source registration differs")
    reported={}
    for (f,n),r in data.items():
        m,w,k,p=s=panel[f]; padded=n=="host-pad8" and pad_shape(s)
        expected=[(m+7)//8*8 if padded else m,(k+15)//16*16 if padded else k+p,w+p,(w+15)//16*16 if padded else w+p]
        found=[int(r[col]) for col in ("provider_m","provider_lda","provider_ldb","provider_ldc")]
        demand(shape(r)==s and found==expected and r["provider"]==underlying and int(r["padded"])==padded and
               int(r["checked"])==m*w and int(r["cpu"])==logical_cpu and r["passed"]==r["traced"]=="1","Actual forwarded geometry "+str((f,n)))
        reported.setdefault(f,{})[n]={"padded":padded,"provider":underlying,"effective_dimensions":found}
    jit=single(rows(root/"libxsmm-routes.csv"),lambda r:(r["family"],r["implementation"]))
    wanted={n for n in methods if n in ("libxsmm-dispatch","libxsmm-retained")}
    demand(set(jit)=={(f,n) for f in panel for n in wanted},"Missing per-layout JIT admission")
    prefix={}
    for (f,n),r in jit.items():
        size=int(r["code_bytes"]);good=size>0
        demand(shape(r)==panel[f] and size>=0 and r["admitted"]==str(int(good)) and r["reason"]==("nonreference_jit_exact_shape" if good else "no_verified_nonreference_kernel"),"Unsupported JIT identity "+str((f,n)))
        prefix[f,n]=good
        reported.setdefault(f,{})[n]={"admitted":good,"code_bytes":size,"reason":r["reason"]}
    for kind,cached,plans in (("oneshot",0,11),("cached-plan",1,6)):
        n="onednn-"+kind
        if n in methods:
            log=(root/("onednn-qualified-"+kind+".log")).read_text()
            demand(re.search(rf"^runtime=3\.14\.1 cpu_runtime=1 gpu_runtime=0 cached_plan={cached} affinity=1 lib=.+$",log,re.M) and
                   "strict_fp32_residual=0x1p-23 subnormal=0x1p-149" in log and
                   re.search(rf"^PASS full_fp64_outputs=73096 created_plans={plans}$",log,re.M),
                   "Strict sequential oneDNN ordinary/retained semantics "+n)
            methods[n].update(native_fp32_smoke=73096,smoke_created_plans=plans)
    return prefix,reported,underlying
