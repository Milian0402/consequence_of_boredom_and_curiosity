"""Restore the b218bef release files, validate every path, then apply the one candidate patch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASELINE = "b218beff1eaf8fd606d0934405cdc1e28ecdfe15"
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(tree, hashes):
    entries = {p.relative_to(tree).as_posix(): sha(p) for p in tree.rglob("*") if p.is_file()}
    if entries != hashes:
        wrong = sorted(k for k in set(entries) | set(hashes) if entries.get(k) != hashes.get(k))
        raise ValueError(f"unexpected source inventory or hashes: {wrong}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("repository",type=Path)
    cli.add_argument("incumbent",type=Path)
    args = cli.parse_args()
    repository, incumbent = args.repository.resolve(), args.incumbent.resolve()
    manifest = json.loads((HERE/"source.json").read_text())
    if manifest["revision"] != BASELINE:
        raise ValueError("unexpected public release")
    candidate = incumbent.with_name(incumbent.name+".candidate")
    subprocess.run([sys.executable, HERE.parent/"release-replay/prepare-sources.py", repository, incumbent],check=True)
    subprocess.run([sys.executable, HERE.parent/"release-public/apply-test-backport.py", incumbent],check=True)
    verify(incumbent, manifest["release_files"])
    shutil.copytree(incumbent,candidate)
    delta = HERE/"candidate.patch"
    if sha(delta) != manifest["patch_sha256"]:
        raise ValueError("candidate patch digest mismatch")
    subprocess.run(["git","apply",str(delta)],cwd=candidate,check=True)
    verify(candidate,manifest["candidate_files"])
    for tree in (incumbent,candidate):
        # compile_entry expects these ABI wrappers beside the independent source snapshot.
        adapter = tree/"tools/world-rank"
        adapter.mkdir(parents=True,exist_ok=True)
        for name in ("vendor.h","cob_adapter.c"):
            shutil.copyfile(HERE.parent/name,adapter/name)
    print(f"baseline={BASELINE} files={len(manifest['release_files'])} candidate_sha256={sha(delta)}")


if __name__ == "__main__":
    main()
