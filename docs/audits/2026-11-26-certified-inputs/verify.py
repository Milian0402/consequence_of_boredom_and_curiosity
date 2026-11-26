"""Validate the complete evidence directory and replay each study offline."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS"


def verify_inventory():
    actual = {}
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symbolic link: {path.relative_to(ROOT)}")
        if path.is_file() and path != MANIFEST:
            actual[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        digest, filename = line.split("  ", 1)
        if filename in expected:
            raise ValueError(f"Repeated manifest path: {filename}")
        expected[filename] = digest
    if expected != actual:
        different = sorted(name for name in set(expected) | set(actual)
                           if expected.get(name) != actual.get(name))
        raise ValueError("Evidence inventory differs: " + ", ".join(different))
    return len(expected)


if __name__ == "__main__":
    print(f"Verified SHA-256 for {verify_inventory()} public files.", flush=True)
    for study in ("linux", "apple", "applications"):
        directory = ROOT / study
        print(f"Verifying {study} observations", flush=True)
        subprocess.run([sys.executable, "-B", str(directory / "verify.py")],
                       cwd=directory, check=True)
    print("Verified the two Linux trials, Apple M5 Max trials and application records.")
