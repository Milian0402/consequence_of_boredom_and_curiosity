"""Stage the exact release and its one preregistered source-only delta."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile

release = "a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6"
tree = "18f22622de1a4b1b7aebbdb66985741aa38c6e9e"
here = pathlib.Path(__file__).resolve().parent
repo, work, out = map(lambda raw: pathlib.Path(raw).resolve(),sys.argv[1:4])
checked = subprocess.check_output(["git","-C",repo,"rev-parse","HEAD","HEAD^{tree}"],text=True).splitlines()
if checked != [release,tree]:
    raise ValueError("Unexpected public production revision or source tree")
paths = ["CMakeLists.txt","Makefile","include","src","tests","third_party","bench"]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def inventory(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
def manifest(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()

original = work/"stream-incumbent"/"source"
candidate = work/"grouped-b"/"source"
original.mkdir(parents=True)
with subprocess.Popen(["git","-C",repo,"archive","--format=tar",release,*paths],stdout=subprocess.PIPE) as proc:
    with tarfile.open(fileobj=proc.stdout,mode="r|") as archive:
        archive.extractall(original,filter="data")
    if proc.wait():
        raise RuntimeError("git archive failed")
base=inventory(original)
if sys.argv[4:] == ["--incumbent-only"]:
    release_only = {"release":release, "tree":tree, "methods":{"stream-incumbent":{"sha256":manifest(base),"files":base}}}
    (out/"grouped-sources.json").write_text(json.dumps(release_only,indent=2,sort_keys=True)+"\n")
    sys.exit(0)
shutil.copytree(original,candidate)
patch=here/"candidate.patch"
patch_hash=digest(patch)
if patch_hash != (here/"candidate.sha256").read_text().split()[0]:
    raise ValueError("candidate patch fingerprint differs from registration")
# Use patch outside a Git checkout so paths are relative to the staged tree.
subprocess.run(["patch","--batch","--forward","--fuzz=0","-p1","--input",patch],cwd=candidate,check=True)
changed=inventory(candidate)
delta=sorted(p for p in base.keys() | changed.keys() if base.get(p)!=changed.get(p))
if delta!=["src/cob_stream_b_dispatch.inc","src/cob_stream_b_group.h","src/cob_stream_b_kernel.c"]:
    raise ValueError(f"Unexpected patch scope {delta}")
identities={"release":release,"tree":tree,"patch_sha256":patch_hash,"changed_paths":delta,
    "methods":{"stream-incumbent":{"sha256":manifest(base),"files":base},
               "grouped-b":{"sha256":manifest(changed),"files":changed}}}
(out/"grouped-sources.json").write_text(json.dumps(identities,indent=2,sort_keys=True)+"\n")
