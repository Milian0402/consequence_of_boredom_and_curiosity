"""Replay both single-thread Linux studies offline using Python's standard library."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
STUDIES = {
    "certified-bf16": ("c609245c8d7359b1e484c7d0896e3ad9a32a97a9", "## Reproduce the comparison"),
    "checked-int8": ("6a9c6a696903926b83caf19364b4d19b23f8f2f6", "## Reproduction"),
}
ARCHES = ("x86_64", "aarch64")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_tree(root, skip):
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != skip:
                records.append(f"{sha(path.read_bytes())}  {relative}\n")
    return "".join(records)


def expand(path, destination):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        check(len(names) == len(set(names)) and archive.testzip() is None, "duplicate or corrupt ZIP member")
        for name in names:
            relative = PurePosixPath(name)
            check(not relative.is_absolute() and ".." not in relative.parts and not name.endswith("/"), "unsafe ZIP member")
            target = destination / relative
            check(not target.exists(), "overlapping ZIP members")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def rewrite_report(raw, study):
    """Adjust the sealed report's file links to the public tree and preserve its source receipts."""
    code, footer_heading = STUDIES[study]

    def source_link(match):
        label, local = match.groups()
        if local == "peer-comparisons.csv":
            return match[0]
        if local == "source/exact/fixtures/data/ATTRIBUTION.md":
            return f"[{label}](../fixtures/ATTRIBUTION.md)"
        if local.startswith("source/exact/") and not local.startswith("source/exact/fixtures/data/"):
            suffix = local.removeprefix("source/")
            url = f"https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/{code}/tools/world-rank/{suffix}"
            return f"[{label}]({url})"
        return f"{label} (`{local}` in [evidence.zip](evidence.zip))"

    before, separator, footer = raw.partition(footer_heading)
    check(bool(separator), "missing study source and archive receipts")
    body = re.sub(r"\[([^\]]+)\]\((?!https?://)([^)]+)\)", source_link, before)
    receipt_lines = [line for line in footer.splitlines() if "https://" in line]
    check(len(receipt_lines) == 3, "expected two job receipts and one source record")
    return (
        body.rstrip() + "\n\n## Reproduction\n\n"
        "Run `python3 -B verify.py` from the [Linux directory](../README.md). The [study archive](evidence.zip) "
        "contains frozen source, raw observations, qualification and comparison records. "
        "[All six paired ratios by peer](peer-comparisons.csv)\n\n"
        + "\n\n".join(receipt_lines) + "\n"
    )


def run_study(study, archives):
    with tempfile.TemporaryDirectory(prefix="sgemm-public-") as temporary:
        destination = Path(temporary)
        expand(ROOT / study / "evidence.zip", destination)
        expand(ROOT / "fixtures/matrices.zip", destination / "source/exact/fixtures/data")
        check((destination / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes(), "source license mismatch")

        # Regenerate the source packet's report and CSV before comparing the reader-facing copies.
        subprocess.run([sys.executable, "-B", str(destination / "render.py"), str(destination),
                        str(destination / "README.md")], check=True)
        rendered = (destination / "README.md").read_text()
        check(rewrite_report(rendered, study) == (ROOT / study / "README.md").read_text(), "reader report differs")
        check((destination / "peer-comparisons.csv").read_bytes() == (ROOT / study / "peer-comparisons.csv").read_bytes(),
              "reader CSV differs")
        (destination / "SHA256SUMS").write_text(hash_tree(destination, "SHA256SUMS"))

        # The underlying verification checks the source blobs, host receipts and independent scores.
        command = [sys.executable, "-B", str(destination / "verify.py")]
        for arch, path in archives.items():
            if path is not None:
                command += ["--archive-" + arch, str(path.resolve())]
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for study in STUDIES:
        for arch in ARCHES:
            parser.add_argument(f"--{study}-{arch}", type=Path, help=f"Original GitHub artifact ZIP for {study} {arch}")
    args = vars(parser.parse_args())
    check(hash_tree(ROOT, "SHA256SUMS") == (ROOT / "SHA256SUMS").read_text(), "public checksum inventory differs")
    for study in STUDIES:
        archives = {arch: args[study.replace("-", "_") + "_" + arch] for arch in ARCHES}
        run_study(study, archives)
    print("Verified both public Linux studies, four host receipts, qualified ranks, paired timings and reader tables.")


if __name__ == "__main__":
    main()
