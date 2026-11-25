"""Replay the frozen selection rule using only the measurements in this checkout."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "signed-winograd" / "results-linux"


def write_host_rows(source, target, arch):
    with source.open(newline="") as inp, target.open("w", newline="") as out:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(row for row in reader if row["arch"] == arch)


def main():
    subprocess.run([sys.executable, "-B", str(EVIDENCE / "verify.py")], check=True)
    hosts = json.loads((EVIDENCE / "provenance.json").read_text())["hosts"]
    tables = {
        "methods.csv": "eligibility.csv",
        "coverage.csv": "case-eligibility.csv",
        "timings.csv": "timings.csv",
    }
    for arch, provider in (("x86_64", "mkl"), ("aarch64", "openblas")):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)
            info = {"native_provider": provider, "model": hosts[arch]["model"]}
            (path / "signed-score.json").write_text(json.dumps(info))
            for source, target in tables.items():
                write_host_rows(EVIDENCE / source, path / target, arch)
            result = subprocess.check_output([sys.executable, "-B", str(HERE / "select.py"), str(path)])
        decision = json.loads(result)
        if decision != json.loads((HERE / (arch + ".json")).read_text()):
            raise ValueError(f"Host selection differs for {arch}")
        print(f"{arch} reproduced {decision['chosen']}")


if __name__ == "__main__":
    main()
