"""Seal and reproduce a shareable frontier benchmark record. This module is standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

MANIFEST = "checksums.sha256"
OMIT = {".DS_Store", "__pycache__"}
BLOCKED_PATHS = tuple(("/" + name + "/").encode() for name in ("home/runner", "Users", "private/var", "tmp"))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root):
    paths={}
    for p in root.rglob("*"):
        name=p.relative_to(root)
        need(not p.is_symlink() and not any(part.startswith(".") or part in OMIT for part in name.parts), f"Unexpected member {name}")
        if not p.is_file() or name.as_posix()==MANIFEST:
            continue
        need(p.suffix.lower() in {".md",".csv",".txt",".json",".py",".sh",".sha256",".patch",".log",".inc",".c",".h"},
             f"Unexpected file type {p.name}")
        data=p.read_bytes()
        need(b"\0" not in data and not any(v in data for v in BLOCKED_PATHS), f"Binary data or local path in {name}")
        paths[name.as_posix()]=p
    return paths


def seal(root):
    return "".join(f"{sha(p)}  {n}\n" for n,p in sorted(files(root).items()))


def write_zip(root, out):
    raw=seal(root)
    (root/MANIFEST).write_text(raw)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(out) as z:
        need(z.testzip() is None and z.read(MANIFEST)==raw.encode(), "Zip CRC or digest inventory changed")
        for n,p in files(root).items():
            need(z.read(n)==p.read_bytes(), f"Packed member differs {n}")


def normalized_result(result):
    # An original archive receipt can only be rederived by supplying that complete archive.
    return {k:v for k,v in result.items() if k!="receipt"}


def replay(root):
    raw=(root/MANIFEST).read_text()
    need(seal(root)==raw, "Packaged evidence differs from its recorded file hashes")
    index=json.loads((root/"hosts.json").read_text())
    for record in index["hosts"]:
        folder=root/record["folder"]
        subprocess.run([sys.executable,"-B",str(root/"members.py"),str(folder)],check=True)
        expected=json.loads((folder/"analysis.json").read_text())
        with tempfile.TemporaryDirectory(prefix="frontier-validation-") as temp:
            check=Path(temp)/"analysis.json"
            cmd=[sys.executable,"-B",str(root/"verify.py"),str(folder/"raw"),"--contract",str(root/"contract.json"),"--output",str(check)]
            subprocess.run(cmd,check=True)
            got=json.loads(check.read_bytes())
        need(normalized_result(got)==normalized_result(expected), f"Worker-source, qualification or timing evidence changed in {folder.name}")
    fresh=subprocess.check_output([sys.executable,"-B",str(root/"hosts.py"),*[str(root/r["folder"]/"analysis.json") for r in index["hosts"]]])
    need(json.loads(fresh)==json.loads((root/"multi-host.json").read_bytes()),"Combined model/replication result differs")
    print(f"Verified {len(index['hosts'])} host record(s) and {len(files(root))} source/evidence files.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory",type=Path,nargs="?",default=Path(__file__).resolve().parent)
    args=cli.parse_args()
    replay(args.directory)
