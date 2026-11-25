"""Portable projections and member receipts for one authenticated Frontier SGEMM job."""
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile


PANEL = "b6d451644a0a979e1b09796ff22c6b4ff24b537a7a701e68afad72ada14d5190"
SEEDS = [856103219,1743857903]
PRIVATE_PATH = re.compile(r"(?:/(?:Users|home|private|var|opt|tmp|mnt|usr|sys|proc|runner|__w)/|"
                          r"(?<![\w])(?:[a-zA-Z]:[\\/]|work/world-rank/)|"
                          r"\b(?:fv-az\d+|runnervm[a-z0-9]+)\b)", re.IGNORECASE)


def check(ok, message):
    if not ok:
        raise ValueError(str(message))


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def encoded_json(value):
    return (json.dumps(value,indent=2,allow_nan=False) + "\n").encode()


def check_portable(data, label, hostname=None):
    text = data.decode("utf8")
    private = PRIVATE_PATH.search(text)
    check(not private, f"private path or host in {label}")
    if hostname:
        check(hostname not in text, f"original machine name in {label}")
    return data


def ranking(data):
    record = json.loads(data)
    entries = record["implementations"]
    for n,(variant,path) in entries.items():
        check(re.fullmatch(r"[A-Za-z0-9_.-]+",n) and isinstance(variant,str), "bad implementation descriptor")
        entries[n] = [variant,Path(path).name]
    return encoded_json(record)


def host(data, rev):
    lines = data.decode().splitlines()
    uname = next((line.split() for line in lines if line.startswith("Linux ")), [])
    check(len(uname) >= 3 and rev in lines, "missing recorded runner identity or source")
    fields = dict(line.split(":",1) for line in lines if ":" in line)
    # Preserve the native CPUID flags used to qualify each ISA. Restrict the
    # record to hardware facts. The raw uname line contains an ephemeral VM ID.
    keys = ["Architecture","CPU(s)","Vendor ID","Model name","CPU family","Model","Stepping",
            "Thread(s) per core","Core(s) per socket","Socket(s)","Flags"]
    saved = [f"{k}: {fields[k].strip()}" for k in keys if k in fields]
    check("Architecture" in fields and "Model name" in fields, "incomplete native CPU identification")
    return ("\n".join(saved+[rev])+"\n").encode(), uname[1], fields["Model name"].strip()


def digest_basenames(data):
    rows = []
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        digest,filename = line.split(maxsplit=1)
        check(re.fullmatch(r"[a-f0-9]{64}",digest) and filename, "malformed binary receipt")
        rows.append(f"{digest}  {Path(filename).name}")
    return ("\n".join(rows)+"\n").encode()


def provider_metadata(data):
    lines=[]
    for line in data.decode().splitlines():
        if line.startswith("runtime=") and " lib=" in line:
            settings,lib=line.split(" lib=",1)
            lines.append(settings+" lib="+Path(lib).name)
        elif line.startswith(("Upstream LIBXSMM ","Upstream OpenBLAS ","openblas_core=",
                              "source_url=","release_tag=","source_sha=","AOCL-BLAS ",
                              "source=https://github.com/giaf/blasfeo ","BLASFEO target=",
                              "ELF dependencies=","vendor=","version=","strict_fp32_residual=","PASS ")):
            lines.append(line)
    check(lines,"no reviewed provider identity")
    return ("\n".join(lines)+"\n").encode()


def project(data, transform, rev):
    if transform == "identity":
        return data
    if transform == "lscpu-hardware-and-source":
        return host(data,rev)[0]
    if transform == "implementation-basenames":
        return ranking(data)
    if transform == "onednn-qualification-basename":
        first, newline, rest = data.partition(b"\n")
        settings, marker, lib = first.rpartition(b" lib=")
        check(settings.startswith(b"runtime=") and marker and lib and newline, "malformed oneDNN qualification header")
        return settings+marker+lib.rsplit(b"/",1)[-1]+newline+rest
    if transform == "binary-basenames":
        return digest_basenames(data)
    if transform == "provider-version-and-qualification":
        return provider_metadata(data)
    raise ValueError("Unknown public projection: "+transform)


def verify_records(root, archive=None):
    evidence = json.loads((root/"provenance.json").read_text())
    members = json.loads((root/"members.json").read_text())
    paths = {p.relative_to(root/"raw").as_posix() for p in (root/"raw").rglob("*") if p.is_file()}
    check(paths == set(members), "public inventory mismatch")
    public_meta = ("members.json","provenance.json")
    for name in public_meta:
        check_portable((root/name).read_bytes(), name)
    for name,row in members.items():
        local = root/"raw"/name
        check(not Path(name).is_absolute() and ".." not in Path(name).parts and local.is_file(), "unsafe public record")
        public = check_portable(local.read_bytes(), name)
        check(hash_bytes(public) == row["published_sha256"], name)
        if row["transform"] == "identity":
            check(row["original_sha256"] == row["published_sha256"], name)
    if archive is not None:
        check(hash_bytes(archive.read_bytes()) == evidence["download_sha256"], "archive SHA-256")
        with ZipFile(archive) as original:
            check(original.testzip() is None, "invalid ZIP CRC")
            for name,row in members.items():
                data = original.read(row["archive_member"])
                check(hash_bytes(data)==row["original_sha256"] and
                      project(data,row["transform"],evidence.get("source_revision","")) == (root/"raw"/name).read_bytes(),
                      f"original-to-public projection mismatch: {name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--archive",type=Path)
    args = parser.parse_args()
    verify_records(args.directory,args.archive)
    print("Verified portable records and their member digests.")
