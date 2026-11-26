"""Audit sufficient conditions for exact one-product BF16 on public application inputs."""
import argparse
import csv
import hashlib
import io
import math
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


SOURCES = {
    "optdigits.zip": (
        "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip",
        "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0",
    ),
    "ml-100k.zip": (
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229",
    ),
    "har.zip": (
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
    ),
    "wine.zip": (
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4",
    ),
}


def archive(root, name, fetch):
    url, digest = SOURCES[name]
    path = root / name
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"Missing {name} in {root}. Supply --fetch to download pinned archives.")
        with urllib.request.urlopen(url) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Changed public archive: {name}")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Archive integrity failure: {name}")
    return zipfile.ZipFile(path)


def original_digits(z, member, rows):
    with tempfile.TemporaryDirectory() as temp:
        packed = Path(temp) / member
        packed.write_bytes(z.read(member))
        data = subprocess.check_output(["gzip", "-dc", str(packed)])
    lines = re.findall(rb"^[01]{32}$", data, flags=re.MULTILINE)
    assert len(lines) == 32 * rows
    return (np.frombuffer(b"".join(lines), dtype="u1") - 48).reshape(rows, 1024)


def cases(root, fetch=False):
    with archive(root, "optdigits.zip", fetch) as z:
        test = np.loadtxt(io.BytesIO(z.read("optdigits.tes")), delimiter=",", dtype="u1")[:, :-1]
        train = np.loadtxt(io.BytesIO(z.read("optdigits.tra")), delimiter=",", dtype="u1")[:, :-1]
        yield "optdigits_counts", test, train.T
        yield "optdigits_unit_intensity", test / 16, (train / 16).T
        test = original_digits(z, "optdigits-orig.windep.Z", 1797)
        train = original_digits(z, "optdigits-orig.tra.Z", 1934)
        yield "optdigits_original_binary", test, train.T

    with archive(root, "ml-100k.zip", fetch) as z:
        raw = np.loadtxt(io.BytesIO(z.read("ml-100k/u.data")), dtype="i4", usecols=(0, 1, 2))
        matrix = np.zeros((943, 1682), dtype="f4")
        matrix[raw[:, 0] - 1, raw[:, 1] - 1] = raw[:, 2]
        assert np.count_nonzero(matrix) == 100_000
        yield "movielens_user_raw_crossproducts", matrix, matrix.T
        mask = matrix != 0
        mean = matrix.sum(axis=1, dtype="f8") / mask.sum(axis=1)
        centered = np.where(mask, matrix - mean[:, None], 0).astype("f4")
        yield "movielens_user_centered_crossproducts", centered, centered.T

    with archive(root, "har.zip", fetch) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as z:
            def har(split):
                with z.open(f"UCI HAR Dataset/{split}/X_{split}.txt") as f:
                    return np.loadtxt(f, dtype="f4")
            yield "har_inertial_linear_kernel", har("test"), har("train").T

    with archive(root, "wine.zip", fetch) as z:
        for wine_type in ("red", "white"):
            with z.open(f"winequality-{wine_type}.csv") as f:
                # Follow a binary64 preprocessing pipeline, then cast the actual GEMM operands.
                x = np.loadtxt(f, delimiter=";", skiprows=1, dtype="f8")[:, :-1]
                x = (x - x.mean(axis=0)).astype("f4")
                yield f"wine_{wine_type}_covariance_numerator", x.T, x


def matrix_stats(x):
    x = np.ascontiguousarray(x, dtype="<f4")
    assert np.isfinite(x).all()
    maximum = float(np.abs(x).max())
    integer = bool(np.all(x == np.trunc(x)))
    bits = x.view("<u4").ravel() & 0x7fffffff
    bf16_fraction = float(np.mean((bits & 0xffff) == 0))
    nonzero = bits[bits != 0]
    if not len(nonzero):
        scale = 0
    else:
        exponent = ((nonzero >> 23) & 255).astype("i2")
        significand = (nonzero & 0x7fffff) | np.where(exponent > 0, 1 << 23, 0).astype("u4")
        low_bit = significand & (~significand + np.uint32(1))
        trailing = np.frexp(low_bit.astype("f8"))[1] - 1
        scale = int(np.min(np.maximum(exponent, 1) - 127 - 23 + trailing))
    return integer, maximum, bf16_fraction, float(len(nonzero) / bits.size), scale, hashlib.sha256(x.tobytes()).hexdigest()


def audit(name, a, b):
    m, k = a.shape
    assert k == b.shape[0] and k > 0
    ai, am, ab, ad, ae, ah = matrix_stats(a)
    bi, bm, bb, bd, be, bh = matrix_stats(b)
    # x == q * 2**exp, where q is an integer. All absolute integer partial sums
    # <= 2**24 prove exact FP32 accumulation under any product/summation order.
    scaled_am = math.ldexp(am, -ae)
    scaled_bm = math.ldexp(bm, -be)
    strict = k <= 4096 and ai and bi and max(am, bm) <= 63
    # Python integers do not overflow when multiplying data-dependent limits.
    wider = ai and bi and max(am, bm) <= 256 and k * int(am) * int(bm) <= 2**24
    dyadic = (max(scaled_am, scaled_bm) <= 256 and k * int(scaled_am) * int(scaled_bm) <= 2**24
              and -126 <= ae + be <= 103)
    return dict(workload=name, m=m, n=b.shape[1], k=k,
                strict_integer=int(strict), bounded_integer=int(wider), scaled_dyadic=int(dyadic),
                max_abs_a=am, max_abs_b=bm, all_integer_a=int(ai), all_integer_b=int(bi),
                bf16_exact_fraction_a=f"{ab:.8f}", bf16_exact_fraction_b=f"{bb:.8f}",
                nonzero_fraction_a=f"{ad:.8f}", nonzero_fraction_b=f"{bd:.8f}",
                grid_exponent_a=ae, grid_exponent_b=be, sha256_a_f32le=ah, sha256_b_f32le=bh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory of the four original archives")
    parser.add_argument("--fetch", action="store_true", help="download any missing originals from their pinned sources")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    rows = [audit(name, a, b) for name, a, b in cases(args.data, args.fetch)]
    dest = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            dest.close()


if __name__ == "__main__":
    main()
