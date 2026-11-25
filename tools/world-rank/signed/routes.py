"""Check production and separately traced complete calls on every registered layout."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys

DIMS = ["m","n","k","pad"]
TRACE = ["core_m","core_n","core_k","calls","core_calls","core_overwrites","positive_updates","negative_updates",
         "depth_updates","right_strips","bottom_strips","quadrants_ok","edges_ok","provider_ok"]
FIELDS = ["implementation",*DIMS,"provider",*TRACE,"inputs_unchanged","guards_ok","linked_checked",
          "trace_checked","baseline_bitwise_equal","threads","cpu","passed","error"]

def parity_digest(worker,dims):
    digest=hashlib.sha256()
    length=0
    with subprocess.Popen([worker,*map(str,dims)],stdout=subprocess.PIPE) as process:
        for data in iter(lambda:process.stdout.read(1024*1024),b""):
            length+=len(data);digest.update(data)
        if process.wait() or length!=4*dims[0]*dims[1]:
            raise ValueError(f"incomplete dense public output from {worker.name}")
    return digest.hexdigest()

def read_output(worker, dims):
    result = subprocess.run([worker,"fixture",*map(str,dims)],capture_output=True,text=True)
    if result.returncode:
        raise ValueError(f"{worker.name} fixture returned {result.returncode}: {result.stderr[-250:]}")
    def extract(tag):
        hits=[line for line in result.stdout.splitlines() if line.startswith(tag+",")]
        if len(hits)!=1:
            raise ValueError(f"{worker.name}: expected one {tag} row")
        return {s.split("=",1)[0]:int(s.split("=",1)[1]) for s in hits[0].split(",")[1:]}
    return extract("FIXTURE"), extract("SIGNED_TRACE") if worker.name.endswith(".trace.worker") else None

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel",type=Path);cli.add_argument("build",type=Path);cli.add_argument("output",type=Path)
    args=cli.parse_args()
    cpu=min(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpu})
    native={"x86_64":"mkl","aarch64":"armpl"}[platform.machine()]
    with args.panel.open(newline="") as stream: panel=list(csv.DictReader(stream))
    errors=0
    matches={}
    with (args.output.parent/"baseline-parity.csv").open("w",newline="") as side:
        rows=csv.DictWriter(side,fieldnames=["provider",*DIMS,"elements","pristine_sha256","disabled_sha256","passed","error"])
        rows.writeheader()
        for p in (native,"openblas"):
            for r in panel:
                dims=tuple(int(r[d]) for d in DIMS)
                row=dict(provider=p,**dict(zip(DIMS,dims)),elements=dims[0]*dims[1],passed=0,error="")
                try:
                    row["pristine_sha256"]=parity_digest(args.build/("pristine-"+p+".parity"),dims)
                    row["disabled_sha256"]=parity_digest(args.build/("original-"+p+".parity"),dims)
                    if row["pristine_sha256"]!=row["disabled_sha256"]: raise ValueError("dense result digests differ")
                    row["passed"]=1
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    row["error"]=str(exc)
                matches[p,dims]=row["passed"];rows.writerow(row);side.flush()
    with args.output.open("w",newline="") as f:
        save=csv.DictWriter(f,fieldnames=FIELDS);save.writeheader()
        for provider in (native,"openblas"):
            for mode in ("original","signed"):
                label=mode+"-"+provider
                for r in panel:
                    m,n,k,pad=(int(r[d]) for d in DIMS)
                    row=dict(implementation=label,m=m,n=n,k=k,pad=pad,provider=provider,cpu=cpu,passed=0,error="")
                    try:
                        dims=m,n,k,pad
                        ordinary,_=read_output(args.build/(label+".worker"),dims)
                        traced,witness=read_output(args.build/(label+".trace.worker"),dims)
                        exact={"checked":m*n,"output_exact":1,"inputs_unchanged":1,"guards":1,"threads":1}
                        if ordinary!=exact or traced!=exact:
                            raise ValueError(f"public fixture evidence differs: {ordinary} {traced}")
                        row.update(inputs_unchanged=1,guards_ok=1,linked_checked=ordinary["checked"],
                                   trace_checked=traced["checked"],threads=1,**witness)
                        row["baseline_bitwise_equal"]=matches[provider,dims]
                        if not row["baseline_bitwise_equal"]: raise ValueError("pristine and disabled baseline did not match")
                        signed=mode=="signed"
                        expected=dict(core_m=m//64*64,core_n=n//64*64,core_k=k//32*32,
                            calls=49+(m%64!=0)+(n%64!=0)+(k%32!=0),core_calls=49,core_overwrites=35 if signed else 49,
                            positive_updates=7 if signed else 0,negative_updates=7 if signed else 0,
                            depth_updates=int(k%32!=0),right_strips=int(n%64!=0),bottom_strips=int(m%64!=0),
                            quadrants_ok=1,edges_ok=1,provider_ok=1)
                        if witness!=expected: raise ValueError(f"unexpected signed route: {witness} expected {expected}")
                        row["passed"]=1
                    except (OSError,ValueError,subprocess.SubprocessError) as exc:
                        row["error"]=str(exc);errors+=1
                    save.writerow(row);f.flush()
    if errors: sys.exit(f"{errors} failed production or signed trace fixtures")
if __name__=="__main__":main()
