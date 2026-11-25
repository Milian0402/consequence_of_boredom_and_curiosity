"""Apply a pinned two-test correction without changing any production or build source."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

source=Path(sys.argv[1]).resolve()
here=Path(__file__).resolve().parent
proof=json.loads((here/"test-backport.json").read_text())
allowed={"tests/test_fused_strassen.c","tests/test_portable_strassen.c"}
if set(proof["files"])!=allowed:
    raise SystemExit("unexpected fixture-backport file set")

def inventory():
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*") if p.is_file()}

before=inventory()
for path in sorted(allowed):
    if before.get(path)!=proof["files"][path]["before"]:
        raise SystemExit("unrecognized original test: "+path)
delta=(here/"test-backport.patch.gz").read_bytes()
if hashlib.sha256(delta).hexdigest()!=proof["patch_sha256"]:
    raise SystemExit("fixture-backport delta digest mismatch")
subprocess.run(["git","apply","--whitespace=nowarn","-"],cwd=source,input=gzip.decompress(delta),check=True)
after=inventory()
expected=dict(before)
for path in sorted(allowed):
    expected[path]=proof["files"][path]["after"]
if after!=expected:
    raise SystemExit("fixture backport changed a production file or produced an unrecognized test")
print("Fixture-only correction",proof["test_commit"],"verified. All original production and build files identical.")
