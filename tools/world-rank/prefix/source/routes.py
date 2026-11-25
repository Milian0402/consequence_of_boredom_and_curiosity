"""Verify real incoming and provider-side pointers at every registered geometry."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
for arg in ("panel", "vendor", "build", "out"):
    parser.add_argument(arg, type=Path)
parser.add_argument("--mock", action="store_true", help="explicit local correctness fixture using a stand-in provider")
args = parser.parse_args()
panel, vendor, build, out = [getattr(args,x).resolve() for x in ("panel","vendor","build","out")]
out.mkdir(parents=True,exist_ok=True)
assert len(os.sched_getaffinity(0))==1
cpu=min(os.sched_getaffinity(0))
fields = "m n k lda ldb ldc ar ac br bc cr cc".split()
stats = "cpu_vendor intel_avx2 eligible route vector_calls provider_calls checked guard_cells a_mod64 b_mod64 c_mod64 a_element_offset b_element_offset c_element_offset cpu ordinary_error trace_error".split()
details = "kind index m n k lda ldb ldc a_offset b_offset c_offset".split()
with panel.open() as source:
    layouts=list(csv.DictReader(source))
names=("host-forward","host-split8","host-prefix8")
provider_hash=hashlib.sha256((vendor/"mkl.so").read_bytes()).hexdigest()

with (out/"prefix-routes.csv").open("w",newline="") as r,(out/"prefix-provider-calls.csv").open("w",newline="") as p,(out/"prefix-provider.txt").open("w") as proof:
    routes=csv.DictWriter(r,["method","family","layout",*fields,*stats,"threads","provider_sha256","passed"])
    parts=csv.DictWriter(p,["method","family","layout",*details])
    routes.writeheader(); parts.writeheader()
    for s in layouts:
        for mode,method in enumerate(names):
            command = [build/"prefix-check",str(mode),build/(method+".so"),build/(method+".trace.so"),
                build/"prefix-observer.so",vendor/"mkl.so",*(s[k] for k in fields)]
            run = subprocess.run(command,capture_output=True,text=True)
            if run.returncode or (not args.mock and "oneMKL LP64 sequential:" not in run.stderr):
                raise RuntimeError(f"{method} {s['family']} failed with {run.returncode}: {run.stderr}")
            parts_here=[]
            result=None
            for line in run.stdout.splitlines():
                tag,*values=line.split(",")
                if tag=="PART":
                    parts_here.append(dict(zip(details,values,strict=True)))
                elif tag=="RESULT" and result is None:
                    result=dict(zip(stats,values,strict=True))
                else:
                    raise ValueError(line)
            if result is None or int(result["cpu"])!=cpu or len(parts_here)!=int(result["vector_calls"])+int(result["provider_calls"]):
                raise ValueError("Incomplete provider route receipt")
            expected_offsets = [int(s[x+"r"])*int(s[ld])+int(s[x+"c"]) for x,ld in zip("abc",("lda","ldb","ldc"))]
            if [int(result[x+"_element_offset"]) for x in "abc"] != expected_offsets:
                raise ValueError("Observed cropped parent origin does not match the registration")
            if args.mock and result["eligible"] == "1" and mode and result["route"]=="forward":
                raise ValueError("Local host did not execute the assigned kernel")
            case=dict(method=method,family=s["family"],layout=s["layout"])
            routes.writerow(dict(case,**{key:s[key] for key in fields},**result,threads=1,provider_sha256=provider_hash,passed=1))
            for actual in parts_here:
                parts.writerow(dict(case,**actual))
            print(method,s["family"],run.stderr.rstrip(),file=proof)
