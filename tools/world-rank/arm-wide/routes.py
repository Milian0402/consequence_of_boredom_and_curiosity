"""Linkage is recorded by the C wrapper around the installed library."""
import csv
from pathlib import Path
import subprocess
import sys
from study import GEOMETRY, SEEDS, write_table

build, out = map(Path, sys.argv[1:])
rows = []
with (out/"panel.csv").open() as source, (out/"route-workers.log").open("w") as log:
    for shape in csv.DictReader(source):
        for method in ("incumbent", "candidate"):
            args = [build/(method+".trace"), "public", "random", str(SEEDS[0]), "0",
                    *(shape[k] for k in GEOMETRY)]
            worker = subprocess.run(args, text=True, capture_output=True)
            log.write(worker.stdout + worker.stderr)
            worker.check_returncode()
            (line,) = [l for l in worker.stdout.splitlines() if l.startswith("ARMSTREAM,")]
            rows.append(dict(family=shape["family"], method=method,
                             **dict(pair.split("=") for pair in line.split(",")[1:])))
write_table(out/"routes.csv", ["family", "method", "public_calls", "stream_calls", "args_ok"], rows)
