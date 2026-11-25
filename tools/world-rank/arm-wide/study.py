"""Run the registered Linux Arm panel. Every record comes from a new process."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys

METHODS = ("incumbent", "candidate", "openblas", "armpl", "blasfeo", "libxsmm-dispatch",
           "libxsmm-retained", "onednn-oneshot", "onednn-cached-plan")
SEEDS = (73194513, 38641729)
FIXTURES = (("random", SEEDS[0]), ("random", SEEDS[1]), ("cancel", SEEDS[0]), ("scaled", SEEDS[1]))
GEOMETRY = "m n k lda ldb ldc ar ac br bc cr cc".split()
METRICS = "checked guards a_mod64 b_mod64 c_mod64 cpu threads error first_s per_call_s".split()
EXTRA = "output_hash code_bytes plans implementation clone_blocked m n k lda ldb ldc".split()


def write_table(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def run(build, vendor, out):
    with (out / "panel.csv").open() as f:
        panel = list(csv.DictReader(f))
    fields = "family method mode seed round place calls".split() + METRICS + EXTRA
    paths = {m: vendor / (m + ".so") for m in METHODS[2:]}
    for policy in ("oneshot", "cached-plan"):
        paths["onednn-" + policy] = vendor / "onednn" / ("onednn-f32-" + policy + ".so")
    failures, parity = {}, []
    with (out / "raw.csv").open("w", newline="") as raw, (out / "workers.log").open("w") as log:
        writer = csv.DictWriter(raw, fields)
        writer.writeheader()

        def record(shape, method, mode, seed, round_id, place):
            family = shape["family"]
            if (family, method) in failures:
                return
            local = method in METHODS[:2]
            worker = build / ((method if local else "incumbent") + ".worker")
            product = "public" if local else str(paths[method])
            control = local and family.startswith("control-") and round_id < 0
            env = os.environ.copy()
            dump = build / f"{family}-{mode}-{seed}-{method}.bin"
            if control:
                env["ARM_CONTROL_OUTPUT"] = str(dump)
            calls = 0 if round_id < 0 else 16
            args = [worker, product, mode, str(seed), str(calls), *(shape[k] for k in GEOMETRY)]
            cmd = subprocess.run(args, capture_output=True, text=True, env=env)
            log.write(f"{family} {method} {mode} {seed} round={round_id} rc={cmd.returncode}\n")
            log.write(cmd.stdout + cmd.stderr)
            log.flush()
            if cmd.returncode:
                failures[family, method] = f"worker rc={cmd.returncode}: " + cmd.stderr[-400:].strip()
                return
            def extract(prefix, names):
                (line,) = [line for line in cmd.stdout.splitlines() if line.startswith(prefix + ",")]
                return dict(zip(names, line.split(",")[1:], strict=True))
            metrics = extract("VIEW", METRICS) | extract("AUDIT", EXTRA)
            writer.writerow(dict(family=family, method=method, mode=mode, seed=seed, round=round_id,
                                 place=place, calls=calls, **metrics))
            raw.flush()
            if control and method == "candidate":
                reference = build / f"{family}-{mode}-{seed}-incumbent.bin"
                bits = dump.read_bytes()
                same = reference.exists() and reference.read_bytes() == bits
                parity.append(dict(family=family, mode=mode, seed=seed, bytes=len(bits),
                                   same=int(same), sha256=hashlib.sha256(bits).hexdigest()))

        for shape in panel:
            for method in METHODS:
                for mode, seed in FIXTURES:
                    record(shape, method, mode, seed, -1, -1)
        for r in range(6):
            shift = r % 3
            ordered = panel[shift:] + panel[:shift]
            for s in (ordered if r < 3 else reversed(ordered)):
                p = (panel.index(s) + 2*shift) % len(METHODS)
                roster = METHODS[p:] + METHODS[:p]
                for place, method in enumerate(roster if r < 3 else reversed(roster)):
                    record(s, method, "random", SEEDS[r//3], r, place)
    write_table(out / "parity.csv", ["family", "mode", "seed", "bytes", "same", "sha256"], parity)
    write_table(out / "failures.csv", ["family", "method", "reason"],
                [dict(family=f, method=m, reason=reason) for (f,m), reason in failures.items()])


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:4]))
