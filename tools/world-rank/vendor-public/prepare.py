"""Apply the experimental provider and core policy to an isolated copy of COB."""
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("repo", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
for directory in ("src", "include"):
    shutil.copytree(args.repo / directory, args.destination / directory, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("* [0-9]*.*"))
expected = {
    "src/gemm.c": "4bf368c5b54e75d011f05ad883c1d4b3628bde2c885580de42ed85d963bcb8ff",
    "src/cob_openblas_winograd.inc": "103d80d88ce40c2fe6aa771b6ae7bc10c2db83f868714174813f60c8cbf5265a",
}
for name, digest in expected.items():
    if hashlib.sha256((args.destination / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"unexpected input revision for {name}")
patch = Path(__file__).with_name("source.patch").resolve()
subprocess.run(["patch", "--quiet", "-p1", "-i", str(patch)], cwd=args.destination, check=True)
