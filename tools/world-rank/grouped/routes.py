"""Verify installed normal/observed calls using the exact positive-stride views."""
import csv
import hashlib
import tempfile
import os
from pathlib import Path
import shutil
import subprocess
import sys

def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

layout_fields="family layout m n k lda ldb ldc ar ac br bc cr cc".split()
view_fields="checked guard_cells a_mod64 b_mod64 c_mod64 cpu threads error first_s batch_s lda ldb ldc".split()
calls="avx2 avx512 grouped_avx2 grouped_avx512".split()

def observe(binary,row,fail=False,mode="random",seed=267491683,dump=None):
    env=dict(os.environ)
    if fail: env["GROUPED_REFUSE_SCRATCH"]="1"
    else: env.pop("GROUPED_REFUSE_SCRATCH",None)
    if dump is None: env.pop("RANK_VIEW_DIGEST_PATH",None)
    else: env["RANK_VIEW_DIGEST_PATH"]=str(dump)
    cmd=[binary,"public",mode,str(seed),"0",*(row[k] for k in layout_fields[2:])]
    data=subprocess.run(cmd,capture_output=True,text=True,check=True,env=env)
    lines=[line for line in data.stdout.splitlines() if line.startswith(("VIEW,","GROUPED,"))]
    result={}
    for line in lines:
        marker,*values=line.split(",")
        if marker in result: raise ValueError("Duplicate "+marker)
        if marker=="VIEW":
            if len(values)!=len(view_fields): raise ValueError("Incomplete observed views")
            result[marker]=dict(zip(view_fields,values))
        else: result[marker]={k:int(v) for k,v in (s.split("=",1) for s in values)}
    return result

panel,work,out=map(Path,sys.argv[1:])
builds=table(out/"grouped-builds.csv")
if [r["implementation"] for r in builds]!=["stream-incumbent","grouped-b"]:
    raise ValueError("Expected both production archives")
isa_flags=list(map(int,(out/"grouped-native-features.txt").read_text().split()))
affinity=os.sched_getaffinity(0)
if len(affinity)!=1 or isa_flags not in ([1,0],[1,1]): raise ValueError("One pinned AVX2/FMA CPU required")
cpu=min(affinity);page=os.sysconf("SC_PAGE_SIZE");isa="avx512" if isa_flags[1] else "avx2"
fields=["implementation",*layout_fields,"isa",*calls,"candidate_calls","route","first_bytes","first_align",
        "checked","trace_checked","guard_cells","a_mod64","b_mod64","c_mod64","args_ok","workspace_ok","guards",
        "inputs_unchanged","threads","cpu","error","trace_error","passed"]

def validate(actual,expected):
    if any(int(actual[k])!=v for k,v in expected.items()):raise ValueError((actual,expected))
    if not (0<=float(actual["error"])<=0.002 and float(actual["first_s"])>0 and float(actual["batch_s"])==0):
        raise ValueError(actual)

with (out/"grouped-routes.csv").open("w",newline="") as tf,(out/"grouped-fallbacks.csv").open("w",newline="") as ff:
    normal=csv.DictWriter(tf,fieldnames=fields);normal.writeheader()
    failure=csv.DictWriter(ff,fieldnames=[*fields,"allocation_refusals"]);failure.writeheader()
    for build in builds:
        method=build["implementation"]; refused=False; seen=set()
        for row in table(panel):
            m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc=(int(row[f]) for f in layout_fields[2:])
            offsets=dict(a_mod64=4*(ar*la+ac)%64,b_mod64=4*(br*lb+bc)%64,c_mod64=4*(cr*lc+cc)%64)
            lds=dict(lda=la,ldb=lb,ldc=lc)
            count_c=((cr+m+2)*lc*4+page-1)//page*(page//4)-m*n
            expected=dict(checked=m*n,guard_cells=count_c,cpu=cpu,threads=1,**lds,**offsets)
            orig=observe(work/(method+".worker"),row)
            if set(orig)!={"VIEW"}: raise ValueError("Unexpected ordinary worker output")
            validate(orig["VIEW"],expected)
            eligible=24<=m<=160 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and m*k<=49152
            grouped=method=="grouped-b" and eligible
            route="grouped-b" if grouped else ("incumbent" if eligible else "bypass")
            seen.add(route)
            for fail in ([False,True] if eligible and not refused else [False]):
                traced=observe(work/(method+".trace.worker"),row,fail)
                validate(traced["VIEW"],expected)
                if not fail and traced["VIEW"]["error"]!=orig["VIEW"]["error"]: raise ValueError("Numerical witness drift")
                witness=traced["GROUPED"]
                counts=dict.fromkeys(calls,0)
                if eligible and not fail:counts[("grouped_" if grouped else "")+isa]=1
                expect=dict(counts,public_calls=1,args_ok=1,alignment_ok=1,refused=int(fail),m=m,n=n,k=k,**lds,**offsets)
                if any(witness[key]!=value for key,value in expect.items()):raise ValueError((method,row,witness,expect))
                if eligible:
                    size=k*(64 if isa=="avx512" else 32)*4*(4 if grouped else 1)
                    if (witness["first_bytes"],witness["first_align"])!=(size,64):raise ValueError("Wrong workspace size/alignment")
                record=dict(row,implementation=method,isa=isa,**counts,candidate_calls=counts["grouped_avx2"]+counts["grouped_avx512"],
                    route="allocation-fallback" if fail else route,first_bytes=witness["first_bytes"],first_align=witness["first_align"],
                    checked=m*n,trace_checked=m*n,guard_cells=count_c,**offsets,args_ok=1,workspace_ok=1,
                    guards=1,inputs_unchanged=1,threads=1,cpu=cpu,error=orig["VIEW"]["error"],trace_error=traced["VIEW"]["error"],passed=1)
                if fail:failure.writerow(dict(record,allocation_refusals=1));refused=True
                else:normal.writerow(record)
        if seen!={"grouped-b" if method=="grouped-b" else "incumbent","bypass"} or not refused:
            raise ValueError("Missing active/bypass/ENOMEM coverage")
shutil.copyfile(out/"grouped-routes.csv",out/"routes.csv")

with (out/"grouped-control-parity.csv").open("w",newline="") as stream, tempfile.TemporaryDirectory(dir=work) as temp:
    proof=csv.DictWriter(stream,fieldnames=[*layout_fields,"mode","seed","checked","incumbent_sha256","candidate_sha256","passed"])
    proof.writeheader()
    for row in table(panel):
        if not row["family"].startswith("control-"): continue
        for mode,seed in (("random",267491683),("random",1893702713),("cancel",267491683),("scaled",1893702713)):
            hashes=[]
            count=int(row["m"])*int(row["n"])
            for build in builds:
                name=build["implementation"]
                path=Path(temp)/(row["family"]+"-"+name+"-"+mode+"-"+str(seed))
                result=observe(work/(name+".worker"),row,mode=mode,seed=seed,dump=path)
                if set(result)!={"VIEW"} or int(result["VIEW"]["checked"])!=count or path.stat().st_size!=4*count:
                    raise ValueError("Incomplete whole-output parity receipt")
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                path.unlink()
            if hashes[0]!=hashes[1]: raise ValueError("Bypass output bits changed")
            proof.writerow(dict(row,mode=mode,seed=seed,checked=count,incumbent_sha256=hashes[0],candidate_sha256=hashes[1],passed=1))
