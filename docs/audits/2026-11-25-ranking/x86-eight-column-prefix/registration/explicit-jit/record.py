"""Capture genuine generated kernels and oneMKL's documented standard-GEMM fallback."""
import csv
import os
from pathlib import Path
import subprocess
import sys

exe, panel, output = map(Path,sys.argv[1:])
cpu = min(os.sched_getaffinity(0))
os.sched_setaffinity(0, {cpu})
keys = "m n k lda ldb ldc ar ac br bc cr cc".split()
with panel.open() as f:
    shapes=list(csv.DictReader(f))
small=dict(zip(keys,map(str,(16,16,16,16,16,16,0,0,0,0,0,0)),strict=True))
small.update(family="probe-small-16",layout="compact")
with output.open("w",newline="") as f:
    fields=["family","layout",*keys,"cpu","status","code","called","checked","guards"]
    writer=csv.DictWriter(f, fields)
    writer.writeheader()
    for shape in [small,*shapes]:
        command=subprocess.run([exe,*(shape[k] for k in keys)],text=True,capture_output=True,check=True)
        state,code,called,checked,guards=command.stdout.strip().split(",")
        if (state,code) not in {("MKL_JIT_SUCCESS","0"),("MKL_NO_JIT","1"),("MKL_JIT_ERROR","2")}:
            raise ValueError((state,code))
        if state!="MKL_JIT_ERROR" and (int(checked)!=int(shape["m"])*int(shape["n"]) or
                                      int(guards)<1 or int(called)!=1):
            raise ValueError("incomplete explicit-GEMM validation")
        writer.writerow({**{k:shape[k] for k in ["family","layout",*keys]},"cpu":cpu,"status":state,
                         "code":code,"called":called,"checked":checked,"guards":guards})
