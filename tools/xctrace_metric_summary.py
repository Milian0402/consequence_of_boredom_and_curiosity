#!/usr/bin/env python3
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize xctrace MetricTable XML exported from CPU Counters.")
    parser.add_argument("xml_file", nargs="?", help="MetricTable XML file; defaults to stdin")
    parser.add_argument("--process", help="case-insensitive process substring filter")
    parser.add_argument("--thread", help="case-insensitive thread substring filter")
    parser.add_argument(
        "--metric",
        action="append",
        help="case-insensitive metric substring filter; can be passed more than once")
    parser.add_argument(
        "--min-duration-ns",
        type=float,
        default=0.0,
        help="drop samples shorter than this many nanoseconds")
    parser.add_argument("--by-core", action="store_true", help="split aggregates by core")
    return parser.parse_args()


def read_xml(path):
    if path:
        return ET.parse(path).getroot()
    return ET.fromstring(sys.stdin.read())


def build_id_map(root):
    ids = {}
    for elem in root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            ids[elem_id] = elem
    return ids


def resolve(elem, ids):
    seen = set()
    while elem is not None and "ref" in elem.attrib:
        ref = elem.attrib["ref"]
        if ref in seen:
            return elem
        seen.add(ref)
        elem = ids.get(ref)
    return elem


def cell_text(elem, ids):
    elem = resolve(elem, ids)
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def cell_fmt(elem, ids):
    elem = resolve(elem, ids)
    if elem is None:
        return ""
    return elem.attrib.get("fmt") or (elem.text or "").strip()


def f64(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def boolean_value(text):
    return text.strip().lower() in ("1", "true", "yes")


def includes(needle, haystack):
    return needle.lower() in haystack.lower()


def schema_columns(node):
    schema = node.find("schema")
    if schema is None:
        return {}
    columns = []
    for col in schema.findall("col"):
        mnemonic = col.findtext("mnemonic")
        if mnemonic:
            columns.append(mnemonic)
    return {name: idx for idx, name in enumerate(columns)}


def get_cell(cells, columns, name):
    idx = columns.get(name)
    if idx is None or idx >= len(cells):
        return None
    return cells[idx]


def iter_metric_rows(root, ids):
    nodes = root.findall(".//node")
    if not nodes:
        nodes = [root]
    for node in nodes:
        columns = schema_columns(node)
        required = {
            "duration",
            "pmi-event",
            "metric-display-name",
            "thread",
            "process",
            "metric-value",
            "core",
            "is-ratio",
        }
        if not required.issubset(columns):
            continue
        for row in node.findall("row"):
            cells = list(row)
            duration = f64(cell_text(get_cell(cells, columns, "duration"), ids))
            value = f64(cell_text(get_cell(cells, columns, "metric-value"), ids))
            if duration is None or value is None:
                continue
            yield {
                "duration": duration,
                "event": cell_fmt(get_cell(cells, columns, "pmi-event"), ids),
                "metric": cell_fmt(get_cell(cells, columns, "metric-display-name"), ids),
                "thread": cell_fmt(get_cell(cells, columns, "thread"), ids),
                "process": cell_fmt(get_cell(cells, columns, "process"), ids),
                "core": cell_fmt(get_cell(cells, columns, "core"), ids),
                "is_ratio": boolean_value(
                    cell_text(get_cell(cells, columns, "is-ratio"), ids)),
                "value": value,
            }


def new_stat():
    return {
        "samples": 0,
        "duration": 0.0,
        "weighted": 0.0,
        "total": 0.0,
        "min": None,
        "max": None,
        "cores": set(),
    }


def add_sample(stat, row):
    value = row["value"]
    stat["samples"] += 1
    stat["duration"] += row["duration"]
    stat["weighted"] += value * row["duration"]
    stat["total"] += value
    stat["cores"].add(row["core"])
    stat["min"] = value if stat["min"] is None else min(stat["min"], value)
    stat["max"] = value if stat["max"] is None else max(stat["max"], value)


def fmt_float(value):
    if value is None:
        return ""
    return f"{value:.9f}"


def main():
    args = parse_args()
    root = read_xml(args.xml_file)
    ids = build_id_map(root)
    stats = defaultdict(new_stat)

    for row in iter_metric_rows(root, ids):
        if row["duration"] < args.min_duration_ns:
            continue
        if args.process and not includes(args.process, row["process"]):
            continue
        if args.thread and not includes(args.thread, row["thread"]):
            continue
        if args.metric and not any(includes(metric, row["metric"]) for metric in args.metric):
            continue
        core = row["core"] if args.by_core else ""
        kind = "ratio" if row["is_ratio"] else "count"
        key = (row["metric"], row["event"], kind, core)
        add_sample(stats[key], row)

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "metric",
        "event",
        "kind",
        "core",
        "samples",
        "total_duration_ns",
        "weighted_average",
        "total_value",
        "value_per_second",
        "min_value",
        "max_value",
        "core_count",
        "cores",
    ])
    for (metric, event, kind, core), stat in sorted(stats.items()):
        duration = stat["duration"]
        weighted = stat["weighted"] / duration if duration > 0.0 else None
        per_second = stat["total"] / (duration / 1_000_000_000.0) if kind == "count" and duration > 0.0 else None
        writer.writerow([
            metric,
            event,
            kind,
            core,
            stat["samples"],
            f"{duration:.0f}",
            fmt_float(weighted),
            fmt_float(stat["total"]),
            fmt_float(per_second),
            fmt_float(stat["min"]),
            fmt_float(stat["max"]),
            len(stat["cores"]),
            ";".join(sorted(stat["cores"])),
        ])


if __name__ == "__main__":
    sys.exit(main())
