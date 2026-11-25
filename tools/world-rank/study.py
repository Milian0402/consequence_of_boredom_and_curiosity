"""Run isolated one-thread workers in counterbalanced order on a frozen shape panel."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import select
from statistics import geometric_mean, median
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager

SEEDS = {"discovery": (19473017, 37292229), "confirm": (6191231, 89743091),
         "small": (40762813, 92862107), "replicate": (59147207, 81729353), "short": (61734017, 39142873),
         "pilot": (72413951, 86537761), "heldout": (15741487, 84927113),
         "tile-screen": (43891307, 96537493), "tile-confirm": (53719481, 94173107),
         "tile-replay": (61003441, 74271079), "tile-child": (89132871, 31570843),
         "tile-parent": (45682013, 79065127), "leaf-wave": (63714293, 91375811),
         "aligned-core": (48236471, 85109381), "hk-forms": (62490137, 87035629),
         "medium-transfer": (68539041, 93461257), "signed-leaves": (31048579, 196173023), "release-replay": (73194653, 28417069)}
QUALIFIED_PHASES = ("aligned-core", "hk-forms", "release-replay", "medium-transfer", "signed-leaves")
TILE_PHASES = ("tile-screen", "tile-confirm", "tile-replay", "tile-child", "tile-parent", "leaf-wave", *QUALIFIED_PHASES)
SHAPE_KEYS = ("family", "m", "n", "k", "pad", "mode")
SMALL_VENDOR_MAX_MNK = 64 ** 3


def read_panel(path):
    cases = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(SHAPE_KEYS) - {"mode"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"panel needs columns {sorted(required)}")
        for row in reader:
            case = {k: int(row[k]) for k in ("m", "n", "k", "pad")}
            case.update(family=row["family"].strip(), mode=row.get("mode") or "random")
            if not case["family"] or min(case[k] for k in ("m", "n", "k")) < 1 or case["pad"] < 0:
                raise ValueError(f"invalid panel row {row}")
            if case["mode"] not in ("random", "cancel", "scaled"):
                raise ValueError(f"unknown input distribution {case['mode']}")
            cases.append(case)
    if not cases or len({tuple(c[k] for k in SHAPE_KEYS[1:]) for c in cases}) != len(cases):
        raise ValueError("empty panel or repeated layout/distribution")
    return cases


def implementations(args):
    entries = {} if args.phase in TILE_PHASES else {"cob-native": ("native", "-")}
    peers = [("openblas", str(args.openblas.resolve()))]
    for spec in args.vendor:
        name, sep, lib = spec.partition("=")
        if not sep:
            raise ValueError(f"expected NAME=SO, got {spec}")
        peers.append((name, lib))
    for name, lib in peers:
        variants = [("direct", name)]
        if args.phase not in ("small", "replicate", "short", "pilot", "heldout", *TILE_PHASES):
            variants.append(("hybrid", f"cob-{name}"))
        for variant, label in variants:
            if label in entries:
                raise ValueError(f"duplicate implementation {label}")
            entries[label] = (variant, str(Path(lib).resolve()))
    small_peers = set()
    for spec in args.small_vendor:
        name, sep, lib = spec.partition("=")
        if not sep or name in entries:
            raise ValueError(f"invalid or duplicate small vendor {spec}")
        entries[name] = ("direct", str(Path(lib).resolve()))
        small_peers.add(name)
    for spec in args.candidate:
        name, sep, value = spec.partition("=")
        variant, colon, lib = value.partition(":")
        if not sep or not colon or name in entries:
            raise ValueError(f"invalid candidate {spec}")
        entries[name] = (variant, "-" if lib == "-" else str(Path(lib).resolve()))
    for name, (variant, lib) in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not variant:
            raise ValueError(f"invalid implementation {name}")
        if lib != "-" and not Path(lib).is_file():
            raise FileNotFoundError(lib)
    return entries, small_peers


def eligible_entries(case, entries, small_peers):
    if case["m"] * case["n"] * case["k"] <= SMALL_VENDOR_MAX_MNK:
        return entries
    return {name: spec for name, spec in entries.items() if name not in small_peers}


def case_args(case, seed):
    return [str(case[k]) for k in ("m", "n", "k", "pad")] + [str(seed), case["mode"]]


def observed_affinity(pid):
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return "", ""
    fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
    cpu = fields["Cpus_allowed_list"].strip()
    threads = int(fields["Threads"])
    if not cpu.isdigit() or threads != 1:
        raise ValueError(f"expected one CPU and one process thread, got {cpu}, {threads}")
    return cpu, threads


def reply(proc, prefix, timeout):
    if not select.select([proc.stdout], [], [], timeout)[0]:
        raise TimeoutError(f"worker timed out waiting for {prefix}")
    line = proc.stdout.readline().strip()
    parts = line.split(",")
    if parts[0] != prefix:
        raise RuntimeError(f"expected {prefix}, got {line!r} (exit {proc.poll()})")
    return parts[1:]


@contextmanager
def worker_session(command, stderr, timeout):
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                            text=True, bufsize=1)
    try:
        yield proc
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        if proc.wait(timeout=timeout) != 0:
            raise RuntimeError(f"worker exited {proc.returncode} at shutdown")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdin.close()
        proc.stdout.close()


class Artifact:
    def __init__(self, name, keys, stack):
        self.file = stack.enter_context(name.open("w", newline=""))
        self.writer = csv.DictWriter(self.file, fieldnames=keys)
        self.writer.writeheader()

    def append(self, row):
        self.writer.writerow(row)
        self.file.flush()


def rotated(names, shift, reverse):
    i = shift % len(names)
    order = names[i:] + names[:i]
    return list(reversed(order)) if reverse else order


def run_block(args, trace, case, index, block, seed, entries, accuracy, timings, results, errors, stress):
    names = list(entries)
    shape_args = case_args(case, seed)
    common = dict(phase=args.phase, **case, block=block, seed=seed, stress=int(stress))
    local_cpus = set()
    with tempfile.TemporaryDirectory(prefix="rank-reference-", dir=args.out) as temp, ExitStack() as processes:
        reference = str(Path(temp) / "reference.bin")
        subprocess.run([args.worker, "oracle", str(args.openblas), reference, *shape_args],
                       stdout=trace, stderr=trace, check=True, timeout=args.timeout)
        expected = case["m"] * case["n"]
        if os.path.getsize(reference) != 8 * expected:
            raise RuntimeError("FP64 oracle produced an incomplete reference")
        # Workers retain their own input and output until this one shape and seed finish.
        # Bound resident matrices for unexpected panels; oversized layouts use equally
        # warmed, short-lived worker processes for every implementation instead.
        m, n, k, pad = (case[x] for x in ("m", "n", "k", "pad"))
        per_worker = 4 * (m * (k + pad) + k * (n + pad) + m * (n + pad))
        if os.getenv("RANK_VERIFY_INPUTS"):
            per_worker += 4 * (m * (k + pad) + k * (n + pad))
        resident = per_worker * len(names) + 8 * expected <= args.matrix_budget_mib * 1024**2
        cache = {}

        def acquire(name, stack, record):
            variant, so = entries[name]
            if variant == "linked-public":
                command = [so, "measure", variant, "-", reference, *shape_args]
            else:
                command = [args.worker, "measure", variant, so, reference, *shape_args]
            print(f"worker {name} block={block} seed={seed} {shape_args}", file=trace, flush=True)
            proc = stack.enter_context(worker_session(command, trace, args.timeout))
            fields = reply(proc, "READY", args.timeout)
            if len(fields) < 4:
                raise ValueError(f"malformed READY for {name}: {fields}")
            error, nonfinite, guards, checked = float(fields[0]), *map(int, fields[1:4])
            cpu, threads = observed_affinity(proc.pid)
            if cpu:
                local_cpus.add(cpu)
                if len(local_cpus) != 1:
                    raise ValueError(f"workers pinned to different CPUs: {local_cpus}")
            valid = math.isfinite(error) and 0 <= error <= args.tolerance and not nonfinite and not guards and checked == expected
            if record:
                accuracy.append(dict(common, implementation=name, checked=checked, max_abs_error=error,
                                     nonfinite=nonfinite, guard_changes=guards, cpu=cpu, threads=threads,
                                     verified=int(valid), detail=",".join(fields[4:])))
            if not valid:
                raise ValueError(f"{name} failed full-output validation, fields={fields}")
            return proc

        failed = set()
        def record_failure(name, exc):
            failed.add(name)
            errors.append(dict(common, implementation=name, error=str(exc)))
            print(f"worker failed: {errors[-1]}", file=trace, flush=True)

        for name in rotated(names, index, block == 2):
            try:
                with ExitStack() as independent:
                    proc = acquire(name, independent, True)
                    if resident:
                        owned = independent.pop_all()
                        processes.enter_context(owned)
                        cache[name] = (proc, owned)
            except (RuntimeError, TimeoutError, OSError, ValueError, subprocess.SubprocessError) as exc:
                record_failure(name, exc)
        if stress:
            return
        for r in range(args.rounds):
            order = rotated(names, index + r, bool((r + block - 1) % 2))
            round_calls = set()
            for position, name in enumerate(order):
                if name in failed:
                    continue
                try:
                    with ExitStack() as independent:
                        proc = cache[name][0] if resident else acquire(name, independent, False)
                        proc.stdin.write("time\n")
                        proc.stdin.flush()
                        seconds, calls_text = reply(proc, "TIME", args.timeout)
                        seconds, calls = float(seconds), int(calls_text)
                        if not math.isfinite(seconds) or seconds <= 0 or calls < 1:
                            raise ValueError(f"{name} produced invalid time or batch count")
                        round_calls.add(calls)
                        cpu, threads = observed_affinity(proc.pid)
                        timings.append(dict(common, round=r+1, position=position+1, implementation=name,
                                            calls=calls, seconds=seconds, cpu=cpu, threads=threads,
                                            resident=int(resident)))
                        results.setdefault((index, name), {})[(block, r)] = seconds
                except (RuntimeError, TimeoutError, OSError, ValueError, subprocess.SubprocessError) as exc:
                    record_failure(name, exc)
                    if resident:
                        cache[name][1].__exit__(type(exc), exc, exc.__traceback__)
            if len(round_calls) != 1:
                raise ValueError("implementations used different time batches for the same shape")


def summarize(args, cases, entries, small_peers, results, errors, comparisons):
    ranking = []
    group_ratios = {}
    strongest_ratios = {}
    full_peers = ["openblas", *(spec.partition("=")[0] for spec in args.vendor)]
    for i, case in enumerate(cases):
        names = list(eligible_entries(case, entries, small_peers))
        if args.phase in QUALIFIED_PHASES:
            names = [name for name in names if (i, name) in args.qualified_layouts]
        independents = [name for name in names if name in full_peers or name in small_peers]
        if not names or not independents:
            errors.append({**case, "error": "no eligible independent comparison"})
            continue
        labels = {name: results.get((i, name), {}) for name in names}
        required = {(block, r) for block in (1, 2) for r in range(args.rounds)}
        if any(set(v) != required for v in labels.values()):
            errors.append({**case, "error": "incomplete paired timing rounds"})
            continue
        elapsed = {name: median(obs.values()) for name, obs in labels.items()}
        one_shot = [name for name in independents if args.phase not in QUALIFIED_PHASES or name != "onednn-cached-plan"]
        if not one_shot:
            errors.append({**case, "error": "no eligible one-shot independent comparison"})
            continue
        fastest_peer = min(one_shot, key=elapsed.__getitem__)
        vs_fastest = {}
        for name in names:
            if name not in independents:
                ratio = median(labels[fastest_peer][j] / labels[name][j] for j in sorted(required))
                vs_fastest[name] = ratio
                strongest_ratios.setdefault(name, []).append(ratio)
        ranking.append({**case, "eligible_independents": independents, "median_seconds": elapsed,
                        "fastest_median": min(names, key=elapsed.__getitem__),
                        "fastest_independent_median": fastest_peer, "speedups_vs_fastest_independent": vs_fastest})
        if args.phase in QUALIFIED_PHASES:
            ranking[-1]["fastest_independent_including_cached"] = min(independents, key=elapsed.__getitem__)
            ranking[-1]["fastest_one_shot_median"] = min((n for n in names if n != "onednn-cached-plan"), key=elapsed.__getitem__)
        for name, peer in itertools.permutations(names, 2):
            values = [labels[peer][key] / labels[name][key] for key in sorted(required)]
            ratio = median(values)
            comparisons.append(dict(phase=args.phase, **case, implementation=name, peer=peer,
                                    pairs=len(values), paired_wins=sum(v > 1.0 for v in values),
                                    median_speedup=ratio, minimum_speedup=min(values),
                                    maximum_speedup=max(values)))
            group_ratios.setdefault((case["family"], name, peer), []).append(ratio)
    family = [{"family": group, "implementation": name, "peer": peer, "scored_cases": len(ratios),
               "wins_on_median_ratio": sum(x > 1 for x in ratios), "geometric_mean_speedup": geometric_mean(ratios)}
              for (group, name, peer), ratios in group_ratios.items()]
    data = dict(phase=args.phase, panel_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                implementations=entries, seeds=SEEDS[args.phase], rounds_per_seed=args.rounds,
                small_vendor_max_mnk=SMALL_VENDOR_MAX_MNK, small_vendors=sorted(small_peers),
                full_output_tolerance=args.tolerance, completed_cases=len(ranking),
                expected_cases=len(cases), errors=errors, shape_rankings=ranking, families=family,
                portfolios_vs_fastest_independent={name: dict(geometric_mean_speedup=geometric_mean(values),
                     wins=sum(v > 1 for v in values), cases=len(values))
                     for name, values in strongest_ratios.items()})
    (args.out / "ranking.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "panel", "out", "openblas"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--phase", choices=tuple(SEEDS), required=True)
    cli.add_argument("--vendor", action="append", default=[], metavar="NAME=SO")
    cli.add_argument("--small-vendor", action="append", default=[], metavar="NAME=SO",
                     help="independent direct peer only on shapes with m*n*k <= 64**3")
    cli.add_argument("--candidate", action="append", default=[], metavar="NAME=VARIANT:SO")
    cli.add_argument("--rounds", type=int, help="rounds per seed")
    stress_args = cli.add_mutually_exclusive_group()
    stress_args.add_argument("--stress", action="store_true", help="validate cancellation and scaled input on one representative rectangle")
    stress_args.add_argument("--stress-all", action="store_true", help="validate cancellation and scaled input on every panel layout")
    cli.add_argument("--tolerance", type=float, default=0.002)
    cli.add_argument("--timeout", type=float, default=600)
    cli.add_argument("--matrix-budget-mib", type=int, default=6144)
    args = cli.parse_args()
    args.rounds = args.rounds if args.rounds is not None else (2 if args.phase == "discovery" else 3)
    minimum = 1 if args.phase == "discovery" else 3
    if args.rounds < minimum or not 0 < args.tolerance <= 0.002 or args.timeout <= 0 or args.matrix_budget_mib <= 0:
        cli.error("invalid timing or validation configuration")
    args.out.mkdir(parents=True, exist_ok=True)
    args.worker, args.openblas = args.worker.resolve(), args.openblas.resolve()
    entries, small_peers = implementations(args)
    cases = read_panel(args.panel)
    seeds, results, errors = SEEDS[args.phase], {}, []
    keys = ["phase", *SHAPE_KEYS, "block", "seed", "stress"]
    with ExitStack() as files:
        trace = files.enter_context((args.out / "worker.log").open("w"))
        accuracy = Artifact(args.out / "accuracy.csv", keys + ["implementation", "checked", "max_abs_error",
                            "nonfinite", "guard_changes", "cpu", "threads", "verified", "detail"], files)
        timings = Artifact(args.out / "timings.csv", keys + ["round", "position", "implementation", "calls",
                           "seconds", "cpu", "threads", "resident"], files)
        comps = Artifact(args.out / "comparisons.csv", ["phase", *SHAPE_KEYS, "implementation", "peer", "pairs",
                         "paired_wins", "median_speedup", "minimum_speedup", "maximum_speedup"], files)
        jobs = [(case, i, block, seed, False) for block, seed in enumerate(seeds, 1)
                for i, case in (enumerate(cases) if block == 1 else reversed(list(enumerate(cases))))]
        if args.stress or args.stress_all:
            indices = range(len(cases))
            if args.stress:
                index = next((i for i,c in enumerate(cases) if min(c[k] for k in ("m","n","k")) >= 1536 and
                              c["m"] != c["n"]), 0)
                indices = [index]
            jobs.extend((dict(cases[i], mode=mode), i, 0, seeds[0], True)
                        for i in indices for mode in ("cancel", "scaled"))
        for case, index, block, seed, stress in jobs:
            try:
                methods = eligible_entries(case, entries, small_peers)
                run_block(args, trace, case, index, block, seed, methods, accuracy, timings, results, errors, stress)
            except (RuntimeError, TimeoutError, OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(dict(**case, block=block, seed=seed, error=str(exc)))
                print(f"measurement failed: {errors[-1]}", file=trace, flush=True)
        if args.phase in QUALIFIED_PHASES:
            from aligned_admission import qualified_layouts
            args.qualified_layouts = qualified_layouts(args, cases, entries, results, SEEDS[args.phase])
        summarize(args, cases, entries, small_peers, results, errors, comps)
    if errors:
        raise SystemExit(f"{len(errors)} incomplete/failed checks, see ranking.json and worker.log")
    print(f"{len(cases)} cases, {len(entries)} methods, {args.rounds * len(seeds)} paired rounds per case")


if __name__ == "__main__":
    main()
