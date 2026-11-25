"""Compare all logical FP32 output bits for the three forwarders on bypasses."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

build, vendor, out = map(Path, sys.argv[1:])
fields="m n k lda ldb ldc ar ac br bc cr cc".split()
with (out/"panel.csv").open() as f:
    panel=list(csv.DictReader(f))
intel=any("GenuineIntel" in line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("vendor_id"))
methods={"mkl":vendor/"mkl.so",**{m:build/"prefix"/(m+".so") for m in ("host-forward","host-split8","host-prefix8")}}
with (out/"prefix-bypass-parity.csv").open("w",newline="") as receipt,tempfile.TemporaryDirectory(dir=build) as tmp:
    w=csv.writer(receipt);w.writerow(["family","layout","mode","seed","method","checked","sha256","mkl_sha256","passed"])
    for s in panel:
        if intel and not s["family"].startswith("control-"):
            continue
        for mode,seed in (("random",1047891527),("random",1936510481),("cancel",1047891527),("scaled",1936510481)):
            golden=None
            for m,plugin in methods.items():
                path=Path(tmp)/"live-fp32"
                env=dict(os.environ,RANK_VIEW_DIGEST_PATH=str(path))
                args=[build/"stream-incumbent.worker",plugin,mode,str(seed),"0",*(s[k] for k in fields)]
                subprocess.run(args,env=env,capture_output=True,check=True)
                bits=path.read_bytes();path.unlink()
                cells=int(s["m"])*int(s["n"])
                if len(bits)!=4*cells: raise ValueError("Incomplete bypass buffer")
                digest=hashlib.sha256(bits).hexdigest()
                if m=="mkl": golden=digest
                if digest!=golden: raise ValueError(f"Bypass mismatch {s['family']} {m} {mode}")
                w.writerow([s["family"],s["layout"],mode,seed,m,cells,digest,golden,1])
