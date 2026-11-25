"""Bind the ordinary and signed build inputs to every file in the verified release."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

here = Path(__file__).resolve().parent
repo, build, out = (Path(s).resolve() for s in sys.argv[1:])
original = build / "source"
candidate = build / "source.signed"
subprocess.run([sys.executable, here.parent / "release-replay/prepare-sources.py", repo, original], check=True)
subprocess.run([sys.executable, here.parent / "release-public/apply-test-backport.py", original], check=True)
manifest = json.loads((here.parent / "medium/source.json").read_text())
base = "b218beff1eaf8fd606d0934405cdc1e28ecdfe15"
delta = here / "signed-leaves.patch"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files(tree):
    return {p.relative_to(tree).as_posix():digest(p) for p in tree.rglob("*") if p.is_file()}

expected = "c5044a79c02e82ec3dca393c8115fcdc2bb0ffcc321b656c033f68321e2e8a94"
if manifest["revision"] != base or files(original) != manifest["release_files"] or digest(delta) != expected:
    raise ValueError("Unrecognized release inventory or signed patch")
shutil.copytree(original, candidate)
subprocess.run(["patch", "-p1", "-i", delta], cwd=candidate, check=True)
prior, new = files(original), files(candidate)
if set(prior) != set(new) or {n for n in prior if prior[n]!=new[n]} != {"src/gemm.c","src/cob_openblas_winograd.inc"}:
    raise ValueError("Unexpected signed source changes")
report = dict(base_commit=base, patch_sha256=expected, original=prior, signed=new)
(out/"sources.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
