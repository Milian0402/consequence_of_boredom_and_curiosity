"""Replay both hosted CPU results and both numerical supplements with Python 3.11+."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

DATA = Path(__file__).resolve().parent


def replay(archive: Path, directory: Path, program: str) -> None:
    with ZipFile(archive) as bundle:
        bundle.extractall(directory)
    subprocess.run([sys.executable, "-B", program], cwd=directory, check=True)


if __name__ == "__main__":
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        with (DATA / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {name}")
    with tempfile.TemporaryDirectory(prefix="bf16-replay-") as temporary:
        directory = Path(temporary)
        replay(DATA / "observations.zip", directory, "verify_all.py")
        supplements = directory / "supplement"
        replay(supplements / "preflight-evidence.zip", directory / "preflight", "replay.py")
        replay(supplements / "long-k-evidence.zip", directory / "long-k", "accuracy/long-k/replay.py")
