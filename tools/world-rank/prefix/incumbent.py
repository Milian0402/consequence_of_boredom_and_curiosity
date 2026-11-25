"""Trace the installed COB comparison binary at each measured public layout."""
import csv
import os
from pathlib import Path
import subprocess
import sys
build, out = map(Path,sys.argv[1:])
cpu,=os.sched_getaffinity(0)
avx2,avx512=(int(x) for x in (out/"grouped-native-features.txt").read_text().split())
assert avx2 and avx512 in (0,1)
dims="m n k lda ldb ldc ar ac br bc cr cc".split()
with (out/"panel.csv").open() as src, (out/"incumbent-routes.csv").open("w",newline="") as dest:
    table=csv.writer(dest); table.writerow(["family","layout","public_calls","isa","kernel_calls","checked","cpu","forwarded_args","workspace_aligned","passed"])
    for s in csv.DictReader(src):
        run=subprocess.run([build/"stream-incumbent.trace.worker","public","random","1047891527","0",*(s[k] for k in dims)],capture_output=True,text=True,check=True)
        trace,=[x for x in run.stdout.splitlines() if x.startswith("GROUPED,")]
        view,=[x for x in run.stdout.splitlines() if x.startswith("VIEW,")]
        obs=dict(x.split("=") for x in trace.split(",")[1:]); v=view.split(",")[1:]
        m,n=int(s["m"]),int(s["n"])
        assert all(obs[k]==s[k] for k in dims[:6])
        assert [int(obs[k]) for k in ("avx2","avx512","grouped_avx2","grouped_avx512")]==[1-avx512,avx512,0,0]
        assert all(obs[k]=="1" for k in ("public_calls","args_ok","alignment_ok")) and obs["refused"]=="0"
        assert int(v[0])==m*n and int(v[5])==cpu and v[6]=="1" and float(v[7])<=0.002
        table.writerow([s["family"],s["layout"],1,"avx512" if avx512 else "avx2",1,m*n,cpu,1,1,1])
