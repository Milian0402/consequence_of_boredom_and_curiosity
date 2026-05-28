#!/usr/bin/env python3
import argparse
import csv
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank benchmark CSV shapes where a target implementation trails a baseline.")
    parser.add_argument("csv_file", nargs="?", help="CSV file from cob_gemm_bench; defaults to stdin")
    parser.add_argument("--target", default="cob one-shot", help="implementation to compare")
    parser.add_argument(
        "--include-cob-baselines",
        action="store_true",
        help="allow other cob implementations, such as cob packed-B, as baselines")
    parser.add_argument(
        "--all",
        action="store_true",
        help="print every comparable shape, not only target regressions")
    parser.add_argument(
        "--max-drop-percent",
        type=float,
        default=None,
        help="ignore rows where best/median - 1 exceeds this percent")
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


def shape_key(row):
    return (row.get("m", ""), row.get("n", ""), row.get("k", ""))


def shape_text(key):
    return "x".join(key)


def stable_enough(row, median, max_drop_percent):
    if max_drop_percent is None:
        return True
    best = f64(row, "best_throughput")
    if best is None or median <= 0.0:
        return False
    drop_percent = (best / median - 1.0) * 100.0
    return drop_percent <= max_drop_percent


def main():
    args = parse_args()
    rows = [row for row in read_rows(args.csv_file) if row.get("kind") == "gemm"]
    by_shape = {}
    for row in rows:
        median = f64(row, "median_throughput")
        if median is None:
            continue
        if not stable_enough(row, median, args.max_drop_percent):
            continue
        by_shape.setdefault(shape_key(row), []).append((row, median))

    reports = []
    for key, entries in by_shape.items():
        target = None
        baselines = []
        for row, median in entries:
            implementation = row.get("implementation", "")
            if implementation == args.target:
                target = (row, median)
            elif args.include_cob_baselines or not implementation.startswith("cob "):
                baselines.append((row, median))

        if target is None or not baselines:
            continue

        baseline = max(baselines, key=lambda item: item[1])
        target_row, target_median = target
        baseline_row, baseline_median = baseline
        ratio = target_median / baseline_median if baseline_median > 0.0 else 0.0
        if ratio >= 1.0 and not args.all:
            continue
        reports.append((ratio, key, target_row, target_median, baseline_row, baseline_median))

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
    for ratio, key, target_row, target_median, baseline_row, baseline_median in reports:
        gap_percent = (baseline_median / target_median - 1.0) * 100.0 if target_median > 0.0 else 0.0
        writer.writerow([
            shape_text(key),
            target_row.get("implementation", ""),
            f"{target_median:.6f}",
            target_row.get("route", ""),
            baseline_row.get("implementation", ""),
            f"{baseline_median:.6f}",
            f"{ratio:.6f}",
            f"{gap_percent:.2f}",
        ])


if __name__ == "__main__":
    main()
