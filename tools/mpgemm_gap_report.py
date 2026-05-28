#!/usr/bin/env python3
import argparse
import csv
import re
import sys


MPGEMM_ROW_RE = re.compile(
    r"^\s*(?P<impl>[A-Za-z_]+):\s*"
    r"(?P<m>\d+),\s*(?P<n>\d+),\s*(?P<k>\d+),\s*"
    r"(?P<gflops>[0-9]+(?:\.[0-9]+)?)\s+GFLOPS\s*$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare COB one-shot CSV rows against MpGEMM singlePerformance output.")
    parser.add_argument("cob_csv", help="COB CSV from tools/mpgemm_calibration.sh")
    parser.add_argument("mpgemm_output", help="MpGEMM singlePerformance.txt output")
    parser.add_argument("--baseline", default="row_sgemm", help="MpGEMM row to compare against")
    parser.add_argument("--target", default="cob one-shot", help="COB implementation to compare")
    parser.add_argument("--all", action="store_true", help="print every comparable row, not only COB gaps")
    return parser.parse_args()


def shape_key(m, n, k):
    return (str(m), str(n), str(k))


def shape_text(key):
    return "x".join(key)


def read_cob_rows(path, target):
    rows = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") != "gemm" or row.get("implementation") != target:
                continue
            try:
                median = float(row["median_throughput"])
            except (KeyError, ValueError):
                continue
            rows[shape_key(row.get("m", ""), row.get("n", ""), row.get("k", ""))] = (row, median)
    return rows


def read_mpgemm_rows(path, baseline):
    rows = {}
    with open(path) as handle:
        for line in handle:
            match = MPGEMM_ROW_RE.match(line)
            if match is None or match.group("impl") != baseline:
                continue
            key = shape_key(match.group("m"), match.group("n"), match.group("k"))
            rows[key] = float(match.group("gflops"))
    return rows


def main():
    args = parse_args()
    cob_rows = read_cob_rows(args.cob_csv, args.target)
    mpgemm_rows = read_mpgemm_rows(args.mpgemm_output, args.baseline)

    reports = []
    for key, baseline_median in mpgemm_rows.items():
        target = cob_rows.get(key)
        if target is None:
            continue
        target_row, target_median = target
        if target_median <= 0.0 or baseline_median <= 0.0:
            continue
        ratio = target_median / baseline_median
        if ratio >= 1.0 and not args.all:
            continue
        reports.append((ratio, key, target_row, target_median, baseline_median))

    reports.sort(key=lambda item: item[0])
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "shape",
        "target",
        "target_median",
        "target_route",
        "baseline",
        "baseline_median",
        "ratio",
        "gap_percent",
    ])
    for ratio, key, target_row, target_median, baseline_median in reports:
        gap_percent = (baseline_median / target_median - 1.0) * 100.0
        writer.writerow([
            shape_text(key),
            target_row.get("implementation", ""),
            f"{target_median:.6f}",
            target_row.get("route", ""),
            args.baseline,
            f"{baseline_median:.6f}",
            f"{ratio:.6f}",
            f"{gap_percent:.2f}",
        ])


if __name__ == "__main__":
    sys.exit(main())
