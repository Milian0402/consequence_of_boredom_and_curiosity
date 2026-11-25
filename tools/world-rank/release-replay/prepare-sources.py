"""Reconstruct and verify the two original release source trees used for Linux builds."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo, target = map(lambda s: Path(s).resolve(), sys.argv[1:])
previous = target.with_name(target.name+".previous")
here = Path(__file__).parent
manifest = json.loads((here/"source-identity.json").read_text())
previous.mkdir()
basefiles = [p for p in manifest["scope"] if subprocess.run(["git","-C",str(repo),"cat-file","-e",manifest["base"]+":"+p],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0]
source = subprocess.check_output(["git","-C",str(repo),"archive",manifest["base"],*basefiles])
subprocess.run(["tar","-x","-C",str(previous)],input=source,check=True)
for variant, dest in (("previous",previous),("release",target)):
    if variant == "release":
        shutil.copytree(previous, dest)
    spec=manifest[variant]
    packed=(here/(variant+"-source.patch.gz")).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=spec["patch_sha256"]:
        raise SystemExit("source delta digest mismatch")
    # Applied in a disposable directory outside the checked-out Git worktree.
    subprocess.run(["git","apply","--whitespace=nowarn","-"],cwd=dest,input=gzip.decompress(packed),check=True)
    found={}
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            data=path.read_bytes()
            found[str(path.relative_to(dest))]=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    if found!=spec["files"]:
        raise SystemExit(f"{variant} source content or file inventory differs from its recorded Git tree")
    print(f"{variant}={spec['commit']} verified_git_blobs={len(found)}")
