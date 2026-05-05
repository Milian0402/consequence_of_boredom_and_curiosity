#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot an (M,N) route heatmap from cob_gemm_bench CSV output.")
    parser.add_argument("csv_file", nargs="?", help="CSV file from cob_gemm_bench; defaults to stdin")
    parser.add_argument("--implementation", default="cob one-shot", help="implementation to plot")
    parser.add_argument("--k", type=int, help="fixed K slice to plot; required if the CSV has multiple K values")
    parser.add_argument("--output", default="bench-heatmap.png", help="PNG path to write")
    parser.add_argument("--dpi", type=int, default=160, help="output DPI")
    parser.add_argument("--title", help="plot title")
    return parser.parse_args()


def read_rows(path):
    if path:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    return list(csv.DictReader(sys.stdin))


def int_field(row, key):
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def float_field(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    args = parse_args()
    rows = [
        row for row in read_rows(args.csv_file)
        if row.get("kind") == "gemm" and row.get("implementation") == args.implementation
    ]
    if not rows:
        raise SystemExit(f"no rows found for implementation: {args.implementation}")

    available_k = sorted({k for row in rows for k in [int_field(row, "k")] if k is not None})
    if args.k is None:
        if len(available_k) != 1:
            values = ", ".join(str(k) for k in available_k)
            raise SystemExit(f"CSV contains multiple K values; pass --k. Available K: {values}")
        args.k = available_k[0]

    cells = {}
    routes = []
    for row in rows:
        m = int_field(row, "m")
        n = int_field(row, "n")
        k = int_field(row, "k")
        throughput = float_field(row, "median_throughput")
        if m is None or n is None or k != args.k or throughput is None:
            continue
        route = row.get("route") or "(none)"
        if route not in routes:
            routes.append(route)
        key = (m, n)
        if key not in cells or throughput > cells[key][1]:
            cells[key] = (route, throughput)

    if not cells:
        raise SystemExit(f"no rows found for K={args.k}")

    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), "cob-matplotlib"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("bench_heatmap.py requires matplotlib and numpy") from exc

    ms = sorted({m for m, _ in cells})
    ns = sorted({n for _, n in cells})
    route_index = {route: i for i, route in enumerate(routes)}
    grid = np.full((len(ms), len(ns)), np.nan)
    labels = [["" for _ in ns] for _ in ms]

    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            cell = cells.get((m, n))
            if cell is None:
                continue
            route, throughput = cell
            grid[i, j] = route_index[route]
            labels[i][j] = f"{throughput:.0f}"

    colors = list(plt.get_cmap("tab20").colors)
    cmap = mcolors.ListedColormap(colors[:max(1, len(routes))])
    cmap.set_bad("#f2f2f2")
    norm = mcolors.BoundaryNorm(range(len(routes) + 1), cmap.N)

    width = max(7.0, 0.65 * len(ns))
    height = max(5.0, 0.55 * len(ms))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(ns)), [str(n) for n in ns], rotation=45, ha="right")
    ax.set_yticks(range(len(ms)), [str(m) for m in ms])
    ax.set_xlabel("N")
    ax.set_ylabel("M")
    ax.set_title(args.title or f"{args.implementation} routes at K={args.k}")
    ax.set_xticks([x - 0.5 for x in range(1, len(ns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(ms))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(ms)):
        for j in range(len(ns)):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color="black")

    handles = [
        mpatches.Patch(color=colors[i % len(colors)], label=route)
        for route, i in route_index.items()
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
