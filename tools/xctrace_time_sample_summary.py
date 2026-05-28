#!/usr/bin/env python3
import argparse
import bisect
import csv
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize xctrace Time Profiler XML exports.")
    parser.add_argument(
        "xml_file",
        nargs="?",
        help="time-sample or time-profile XML file; defaults to stdin")
    parser.add_argument("--process", help="case-insensitive process substring filter")
    parser.add_argument("--thread", help="case-insensitive thread substring filter")
    parser.add_argument("--state", default="Running", help="thread-state filter")
    parser.add_argument("--sample-type", default="Timer Fired", help="sample-type filter")
    parser.add_argument(
        "--mode",
        choices=("pc", "callstack", "callstack-with-pc"),
        default="pc",
        help="count the sampled PC, callstack frames, or both")
    parser.add_argument(
        "--group",
        choices=("address", "symbol"),
        default="address",
        help="aggregate rows by raw address or by symbol name")
    parser.add_argument(
        "--top",
        "--limit",
        dest="top",
        type=int,
        default=30,
        help="maximum rows to print")
    parser.add_argument("--by-core", action="store_true", help="split samples by core")
    parser.add_argument("--binary", help="binary to symbolize addresses against")
    parser.add_argument(
        "--load-address",
        help="runtime __TEXT load address, for example 0x100640000")
    parser.add_argument("--arch", default="arm64", help="architecture passed to atos")
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


def parse_int(text):
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def bt_addresses(bt, ids):
    bt = resolve(bt, ids)
    if bt is None or bt.tag == "sentinel":
        return (None, [])

    pc = parse_int(cell_text(bt.find("text-address"), ids))
    frames = []
    text_addresses = resolve(bt.find("text-addresses"), ids)
    if text_addresses is not None and text_addresses.text:
        for item in text_addresses.text.split():
            value = parse_int(item)
            if value:
                frames.append(value)
    return (pc, frames)


def tagged_backtrace_frames(stack, ids):
    stack = resolve(stack, ids)
    if stack is None or stack.tag == "sentinel":
        return (None, [], {})

    backtrace = resolve(stack.find("backtrace"), ids)
    if backtrace is None:
        return (None, [], {})

    frames = []
    symbols = {}
    for frame in backtrace.findall("frame"):
        frame = resolve(frame, ids)
        if frame is None:
            continue
        addr = parse_int(frame.attrib.get("addr"))
        if addr is None:
            continue
        frames.append(addr)
        name = frame.attrib.get("name")
        if name:
            symbols[addr] = name
    pc = frames[0] if frames else None
    return (pc, frames, symbols)


def iter_time_samples(root, ids):
    nodes = root.findall(".//node")
    if not nodes:
        nodes = [root]
    for node in nodes:
        columns = schema_columns(node)
        required = {
            "time",
            "thread",
            "core-index",
            "thread-state",
            "cp-user-callstack",
            "sample-type",
        }
        if not required.issubset(columns):
            continue
        for row in node.findall("row"):
            cells = list(row)
            pc, frames = bt_addresses(get_cell(cells, columns, "cp-user-callstack"), ids)
            yield {
                "time": cell_text(get_cell(cells, columns, "time"), ids),
                "thread": cell_fmt(get_cell(cells, columns, "thread"), ids),
                "process": cell_fmt(get_cell(cells, columns, "thread"), ids),
                "core": cell_fmt(get_cell(cells, columns, "core-index"), ids),
                "state": cell_fmt(get_cell(cells, columns, "thread-state"), ids),
                "sample_type": cell_fmt(get_cell(cells, columns, "sample-type"), ids),
                "pc": pc,
                "frames": frames,
                "symbols": {},
            }


def iter_time_profile_samples(root, ids):
    nodes = root.findall(".//node")
    if not nodes:
        nodes = [root]
    for node in nodes:
        columns = schema_columns(node)
        required = {
            "time",
            "thread",
            "process",
            "core",
            "thread-state",
            "stack",
        }
        if not required.issubset(columns):
            continue
        for row in node.findall("row"):
            cells = list(row)
            pc, frames, symbols = tagged_backtrace_frames(
                get_cell(cells, columns, "stack"), ids)
            yield {
                "time": cell_text(get_cell(cells, columns, "time"), ids),
                "thread": cell_fmt(get_cell(cells, columns, "thread"), ids),
                "process": cell_fmt(get_cell(cells, columns, "process"), ids),
                "core": cell_fmt(get_cell(cells, columns, "core"), ids),
                "state": cell_fmt(get_cell(cells, columns, "thread-state"), ids),
                "sample_type": "Timer Fired",
                "pc": pc,
                "frames": frames,
                "symbols": symbols,
            }


def load_nm_symbols(binary):
    try:
        result = subprocess.run(
            ["nm", "-n", binary],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True)
    except (OSError, subprocess.CalledProcessError):
        return (None, [], [])

    symbols = []
    text_base = None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        addr = parse_int("0x" + parts[0])
        name = parts[2]
        if addr is None:
            continue
        if name == "__mh_execute_header":
            text_base = addr
        if parts[1].lower() in ("t", "w"):
            symbols.append((addr, name))
    if text_base is None and symbols:
        text_base = symbols[0][0]
    addrs = [addr for addr, _ in symbols]
    names = [name for _, name in symbols]
    return (text_base, addrs, names)


def symbolize_with_nm(addresses, binary, load_address):
    text_base, symbol_addrs, symbol_names = load_nm_symbols(binary)
    if text_base is None or not symbol_addrs:
        return {}
    load = parse_int(load_address)
    if load is None:
        return {}

    symbols = {}
    max_symbol = symbol_addrs[-1]
    for addr in addresses:
        linked = text_base + (addr - load)
        if linked < text_base or linked > max_symbol:
            continue
        idx = bisect.bisect_right(symbol_addrs, linked) - 1
        if idx < 0:
            continue
        offset = linked - symbol_addrs[idx]
        symbols[addr] = f"{symbol_names[idx]} + {offset}"
    return symbols


def symbolize_with_atos(addresses, binary, load_address, arch):
    if not binary or not load_address or not addresses:
        return {}
    cmd = ["atos", "-o", binary, "-arch", arch, "-l", load_address]
    cmd.extend(hex(addr) for addr in addresses)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True)
    except (OSError, subprocess.CalledProcessError):
        return {}

    symbols = {}
    lines = result.stdout.splitlines()
    for addr, line in zip(addresses, lines):
        if line and line != hex(addr):
            symbols[addr] = line
    return symbols


def symbolize(addresses, binary, load_address, arch):
    if not binary or not load_address:
        return {}
    symbols = symbolize_with_atos(addresses, binary, load_address, arch)
    if len(symbols) < len(addresses):
        fallback = symbolize_with_nm(addresses, binary, load_address)
        fallback.update(symbols)
        symbols = fallback
    return symbols


def add_count(counter, sample, addr):
    if addr:
        key = (addr, sample["core"]) if counter.by_core else (addr, "")
        counter.counts[key] += 1


class AddressCounter:
    def __init__(self, by_core):
        self.by_core = by_core
        self.counts = Counter()


def main():
    args = parse_args()
    root = read_xml(args.xml_file)
    ids = build_id_map(root)
    counter = AddressCounter(args.by_core)
    known_symbols = {}
    total = 0

    samples = list(iter_time_samples(root, ids))
    if not samples:
        samples = list(iter_time_profile_samples(root, ids))

    for sample in samples:
        if args.state and sample["state"] != args.state:
            continue
        if args.sample_type and sample["sample_type"] != args.sample_type:
            continue
        if args.process and not includes(args.process, sample["process"]):
            continue
        if args.thread and not includes(args.thread, sample["thread"]):
            continue
        known_symbols.update(sample["symbols"])
        total += 1
        if args.mode in ("pc", "callstack-with-pc"):
            add_count(counter, sample, sample["pc"])
        if args.mode in ("callstack", "callstack-with-pc"):
            for addr in sample["frames"]:
                add_count(counter, sample, addr)

    top_addresses = sorted({addr for (addr, _) in counter.counts})
    symbols = {addr: known_symbols[addr] for addr in top_addresses if addr in known_symbols}
    missing_addresses = [addr for addr in top_addresses if addr not in symbols]
    symbols.update(symbolize(missing_addresses, args.binary, args.load_address, args.arch))

    writer = csv.writer(sys.stdout)
    writer.writerow(["rank", "address", "core", "samples", "percent", "symbol"])
    if args.group == "symbol":
        grouped = Counter()
        for (addr, core), count in counter.counts.items():
            label = symbols.get(addr) or hex(addr)
            grouped[(label, core)] += count
        for rank, ((label, core), count) in enumerate(grouped.most_common(args.top), start=1):
            percent = (count / total * 100.0) if total else 0.0
            writer.writerow([rank, "", core, count, f"{percent:.6f}", label])
    else:
        for rank, ((addr, core), count) in enumerate(
                counter.counts.most_common(args.top), start=1):
            percent = (count / total * 100.0) if total else 0.0
            writer.writerow([
                rank,
                hex(addr),
                core,
                count,
                f"{percent:.6f}",
                symbols.get(addr, ""),
            ])


if __name__ == "__main__":
    sys.exit(main())
