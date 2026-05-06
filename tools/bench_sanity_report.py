#!/usr/bin/env python3
import argparse
import csv
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flag benchmark rows whose median is much slower than their best sample.")
    parser.add_argument("csv_file", nargs="?", help="benchmark CSV; defaults to stdin")
    parser.add_argument(
        "--max-drop-percent",
        type=float,
        default=15.0,
        help="flag rows where best/median - 1 exceeds this percent")
    parser.add_argument(
        "--all",
        action="store_true",
        help="print every row, not only flagged rows")
    return parser.parse_args()


def read_rows(path):
    if path:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    return list(csv.DictReader(sys.stdin))


def f64(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    args = parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "shape",
        "implementation",
        "route",
        "best_throughput",
        "median_throughput",
        "best_median_drop_percent",
    ])

    reports = []
    for row in read_rows(args.csv_file):
        if row.get("kind") != "gemm":
            continue
        best = f64(row, "best_throughput")
        median = f64(row, "median_throughput")
        if best is None or median is None or median <= 0.0:
            continue
        drop_percent = (best / median - 1.0) * 100.0
        if args.all or drop_percent > args.max_drop_percent:
            shape = "x".join([row.get("m", ""), row.get("n", ""), row.get("k", "")])
            reports.append((drop_percent, shape, row, best, median))

    reports.sort(key=lambda item: item[0], reverse=True)
    for drop_percent, shape, row, best, median in reports:
        writer.writerow([
            shape,
            row.get("implementation", ""),
            row.get("route", ""),
            f"{best:.6f}",
            f"{median:.6f}",
            f"{drop_percent:.2f}",
        ])


if __name__ == "__main__":
    main()
