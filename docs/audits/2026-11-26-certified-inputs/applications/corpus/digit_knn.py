"""Use the exact dense product in a one-nearest-neighbor digit classifier."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


def binary_records(archive, suffix):
    compressed = archive.read("optdigits-orig." + suffix + ".Z")
    raw = subprocess.run(["gzip", "-cd"], input=compressed, check=True, capture_output=True).stdout
    lines = raw.splitlines()
    first = next(i for i, line in enumerate(lines) if len(line) == 32 and set(line) <= {48, 49})
    lines = lines[first:]
    assert len(lines) % 33 == 0
    images = []
    labels = []
    for i in range(0, len(lines), 33):
        pixels = b"".join(lines[i:i+32])
        assert len(pixels) == 1024 and set(pixels) <= {48, 49}
        images.append(np.frombuffer(pixels, dtype=np.uint8) - 48)
        labels.append(int(lines[i+32]))
    return np.stack(images), np.array(labels)


def count_records(archive, suffix):
    table = np.loadtxt(io.BytesIO(archive.read("optdigits." + suffix)), delimiter=",", dtype=np.uint8)
    return table[:, :-1], table[:, -1]


def run(name, test, train):
    a, expected = test
    b, labels = train
    m, k = a.shape
    n = b.shape[0]
    assert k * int(a.max()) * int(b.max()) <= 2**24
    # Binary64 gives each exact integer dot product within these bounds.
    dots = a.astype(np.float64) @ b.astype(np.float64).T
    a_norm = (a.astype(np.int32)**2).sum(axis=1)
    b_norm = (b.astype(np.int32)**2).sum(axis=1)
    distances = a_norm[:, None] + b_norm[None, :] - 2 * dots.astype(np.int32)
    assert (distances >= 0).all() and np.array_equal(dots, np.trunc(dots))
    nearest = np.argmin(distances, axis=1)
    for i in range(0, m, 79):
        direct = ((b.astype(np.int32) - a[i])**2).sum(axis=1)
        assert np.array_equal(direct, distances[i])
    return dict(workload=name, m=m, n=n, k=k, recognized=int(np.sum(labels[nearest] == expected)),
                test_records=m, f32_row_major_product_sha256=hashlib.sha256(dots.astype("<f4").tobytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="original UCI Optical Digits ZIP")
    args = parser.parse_args()
    expected = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Unexpected UCI Optical Digits archive SHA-256")
    with zipfile.ZipFile(args.archive) as z:
        print(json.dumps([
            run("pixel_count_features", count_records(z, "tes"), count_records(z, "tra")),
            run("original_bitmaps", binary_records(z, "windep"), binary_records(z, "tra")),
        ], indent=2))
