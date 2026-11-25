"""Verify public calls from normal and wrapped drivers linked to the same archive."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))
def fixture(binary,dims,refuse=False):
    env=dict(os.environ)
    if refuse: env["FRONTIER_REFUSE_SCRATCH"]="1"
    else: env.pop("FRONTIER_REFUSE_SCRATCH",None)
    data=subprocess.run([binary,"fixture",*map(str,dims)],capture_output=True,text=True,check=True,env=env)
    return {r.split(",",1)[0]:dict(x.split("=",1) for x in r.split(",")[1:]) for r in data.stdout.splitlines()}
def eligible(m,n,k):
    return 24<=m<=160 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and m*k<=49152

panel,work,out=map(Path,sys.argv[1:])
modes=table(out/"frontier-builds.csv")
if {r["implementation"] for r in modes}!={"frontier-incumbent","stream-packed-a"} or len(modes)!=2:
    raise ValueError("Requires one build receipt per product")
features=list(map(int,(out/"frontier-native-features.txt").read_text().split()))
cpu=os.sched_getaffinity(0)
if len(cpu)!=1 or features not in ([1,1],[1,0]):
    raise ValueError("Expected a single pinned AVX2/FMA-capable logical CPU")
isa="avx512" if features[1] else "avx2"
fields=("implementation","family","m","n","k","pad","isa","avx2","avx512","packed_avx2","packed_avx512","candidate_calls",
        "route","checked","trace_checked","guards","inputs_unchanged","threads","cpu","passed")
counters=("avx2","avx512","packed_avx2","packed_avx512")
rows=table(panel)
with (out/"frontier-routes.csv").open("w",newline="") as trace, (out/"frontier-fallbacks.csv").open("w",newline="") as fallback:
    writer=csv.DictWriter(trace,fieldnames=fields);writer.writeheader()
    refusals=csv.DictWriter(fallback,fieldnames=(*fields,"allocation_refusals"));refusals.writeheader()
    for method in modes:
        name=method["implementation"]; seen=set(); refused=False
        for row in rows:
            m,n,k,pad=dims=tuple(int(row[f]) for f in ("m","n","k","pad"))
            expected={"checked":str(m*n),"output_exact":"1","inputs_unchanged":"1","guards":"1","threads":"1"}
            plain=fixture(work/(name+".worker"),dims)
            if plain!={"FIXTURE":expected}: raise ValueError(f"Uninstrumented output mismatch {name} {dims}")
            selected=eligible(m,n,k)
            new=name=="stream-packed-a" and selected and m<=96
            route=("packed-a" if new else "incumbent") if selected else "bypass"
            seen.add(route)
            for fail in ([False,True] if selected and not refused and m<=96 else [False]):
                data=fixture(work/(name+".trace.worker"),dims,fail)
                want={field:0 for field in counters}
                if selected and not fail: want[("packed_" if new else "")+isa]=1
                proof={key:int(v) for key,v in data["FRONTIER"].items()}
                if data["FIXTURE"]!=expected or proof!=dict(want,public_calls=1,args_ok=1,alignment_ok=1,refused=int(fail)):
                    raise ValueError(f"Unexpected witnessed route {name} {dims} refuse={fail}: {data}")
                result=dict(row,implementation=name,isa=isa,**want,candidate_calls=want["packed_avx2"]+want["packed_avx512"],
                    route="allocation-fallback" if fail else route,checked=m*n,trace_checked=m*n,guards=1,inputs_unchanged=1,
                    threads=1,cpu=min(cpu),passed=1)
                if fail:
                    refusals.writerow(dict(result,allocation_refusals=1));refused=True
                else: writer.writerow(result)
        required={"incumbent","bypass"}|({"packed-a"} if name=="stream-packed-a" else set())
        if seen!=required or not refused: raise ValueError(f"Missing route coverage {name}: {seen}")
shutil.copyfile(out/"frontier-routes.csv",out/"routes.csv")

with (out/"frontier-control-parity.csv").open("w",newline="") as stream:
    evidence=csv.DictWriter(stream,fieldnames=("family","m","n","k","pad","checked","incumbent_sha256","candidate_sha256","passed"))
    evidence.writeheader()
    control_count=0
    for row in rows:
        m,n,k,pad=dims=tuple(int(row[f]) for f in ("m","n","k","pad"))
        if eligible(m,n,k) and m<=96: continue
        results={}
        for mode in modes:
            name=mode["implementation"]
            values=subprocess.check_output([work/(name+".parity"),*map(str,dims)])
            if len(values)!=4*m*n: raise ValueError("Incomplete dense control "+name)
            results[name]=hashlib.sha256(values).hexdigest()
        if len(set(results.values()))!=1: raise ValueError("Dense bypass result changed "+row["family"])
        evidence.writerow(dict(row,checked=m*n,incumbent_sha256=results["frontier-incumbent"],
                               candidate_sha256=results["stream-packed-a"],passed=1))
        control_count+=1
    if control_count!=2: raise ValueError("Expected two unchanged-route controls")
