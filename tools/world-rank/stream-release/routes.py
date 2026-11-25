"""Record separate ordinary and observed calls using the exact CMake or Make archive."""
import csv
import os
from pathlib import Path
import subprocess
import sys

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def execute(binary, dims):
    result = subprocess.run([binary, "fixture", *map(str,dims)], text=True,
                            capture_output=True, check=True)
    return {line.split(",",1)[0]: dict(field.split("=") for field in line.split(",")[1:])
            for line in result.stdout.splitlines()}

build,out = map(Path,sys.argv[1:])
avx2,avx512 = map(int,(out/"native-features.txt").read_text().split())
cpu = os.sched_getaffinity(0)
if len(cpu)!=1 or not avx2:
    raise SystemExit("acceptance needs a single pinned AVX2/FMA-capable x86 CPU")
width = 64 if avx512 else 32
fields = ("implementation","family","m","n","k","pad","checked","width","avx2","avx512","guards","inputs_unchanged","threads","passed")
with (out/"routes.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for method in read(out/"product-builds.csv"):
        name=method["implementation"]
        for row in read(out/"panel.csv"):
            m,n,k,pad = dims = [int(row[key]) for key in ("m","n","k","pad")]
            observed=execute(build/(name+".trace.worker"),dims)
            plain=execute(build/(name+".worker"),dims)
            expected=dict(checked=str(m*n),output_exact="1",inputs_unchanged="1",guards="1",threads="1")
            eligible=24<=m<=160 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and m*k<=49152
            actual=observed["DISPATCH"]
            hit=width if eligible and method["option"]=="ON" else 0
            if (plain!={"FIXTURE":expected} or observed["FIXTURE"]!=expected or
                actual!={"avx2":str(int(hit==32)),"avx512":str(int(hit==64))}):
                raise ValueError(f"{name} {dims}: ordinary={plain}, instrumented={observed}, expected width {hit}")
            writer.writerow(dict(row,implementation=name,checked=m*n,width=hit,**actual,
                                 guards=1,inputs_unchanged=1,threads=1,passed=1))
