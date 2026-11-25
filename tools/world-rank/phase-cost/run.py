"""Time the three cache-panel phases of the frozen x86 SGEMM without changing the checkout."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time

BASE = "b83033d199137b370c03a5a3d63365f1025162b7"
TREES = {
    "src": "6097adf143c81c70dd5d47606cd3794d7a503692",
    "include": "bf1837a7c027b81b86cd3c4140fc79937f698dcd",
    "tools/world-rank/challenger-flags.txt": "a78f6a1e2549373a8cc86b2eefa9e71faf62afa7",
}
HERE = Path(__file__).resolve().parent
CLOCK_SOURCE = """
#include <time.h>
static uint64_t phase_data[8];
static uint64_t phase_now(void)
{
    struct timespec t;
    __asm__ __volatile__("" ::: "memory");
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) abort();
    __asm__ __volatile__("" ::: "memory");
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
void phase_reset(void) { memset(phase_data, 0, sizeof(phase_data)); }
void phase_get(uint64_t out[8]) { memcpy(out, phase_data, sizeof(phase_data)); }
"""


def instrument(text):
    start = text.index("static int cob_sgemm_cache_panel(")
    end = text.index("\n#endif", start)
    source = text[start:end]
    calls = [
        "cob_blocked_pack_a(packed_a, a + p, lda, m, kb, mr);",
        "cob_blocked_pack_b(packed_b, b + (size_t)p * ldb + j, ldb, kb, nb, nr);",
        "cob_blocked_apply_panel(kernel, m, nb, kb, packed_a, packed_b, c + j, ldc, p != 0);",
    ]
    for i, call in enumerate(calls):
        assert source.count(call) == 1
        source = source.replace(call, f"""{{ uint64_t t = phase_now();
            {call}
            phase_data[{i}] += phase_now() - t;
            phase_data[{i+3}]++;
        }}""")
    assert source.count("free(packed_b);") == 1
    source = source.replace("free(packed_b);", "free(packed_b);\n    phase_data[6]++;\n    phase_data[7] = (uint64_t)nr;")
    return text[:start] + CLOCK_SOURCE + source + text[end:]


def frozen_source(repo, dest):
    git = ["git", "-C", str(repo)]
    present = subprocess.run(git + ["cat-file", "-e", BASE+"^{commit}"], capture_output=True)
    source = BASE if present.returncode == 0 else "HEAD"
    for path, expected in TREES.items():
        got = subprocess.check_output(git + ["rev-parse", f"{source}:{path}"], text=True).strip()
        if got != expected:
            raise SystemExit(f"Missing exact source {BASE} for {path}. Use actions/checkout with fetch-depth: 0.")
    archive = subprocess.check_output(git + ["archive", source, *TREES])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return subprocess.check_output(git + ["rev-parse", source], text=True).strip()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("repo", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This diagnostic measures the Linux x86-64 route.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cpu = min(os.sched_getaffinity(0))
    cc = os.environ.get("CC", "gcc")
    with tempfile.TemporaryDirectory(prefix="cob-phase-") as tmp:
        build = Path(tmp)
        baseline = build/"control"
        source = frozen_source(args.repo.resolve(), baseline)
        profile = build/"instrumented"
        shutil.copytree(baseline, profile)
        target = profile/"src/cob_blocked_leaf.inc"
        target.write_text(instrument(target.read_text()))
        flags = ["-std=c11", "-O3", "-march=native", "-fPIC", "-shared",
                 "-DCOB_ENABLE_PORTABLE_STRASSEN=1", "-DCOB_ENABLE_RECURSIVE_STRASSEN=1",
                 "-DCOB_ENABLE_X86_MEDIUM_ROUTE=1"]
        for line in (baseline/"tools/world-rank/challenger-flags.txt").read_text().splitlines():
            if line:
                flags.append("-D"+line)
        for tree in (baseline, profile):
            subprocess.run([cc, *flags, "-I"+str(tree/"include"), "-I"+str(tree/"src"),
                            str(tree/"src/gemm.c"), "-o", str(tree.with_suffix(".so"))], check=True)
        exe = build/"probe"
        subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", str(HERE/"worker.c"),
                        "-ldl", "-lm", "-o", str(exe)], check=True)
        os.sched_setaffinity(0, {cpu})
        if os.sched_getaffinity(0) != {cpu}:
            raise RuntimeError("Failed to pin timing process")
        command = [str(exe), str(baseline.with_suffix(".so")), str(profile.with_suffix(".so"))]
        with (out/"raw.csv").open("w") as raw, (out/"worker.log").open("w") as log:
            subprocess.run(command, check=True, stdout=raw, stderr=log)
    subprocess.run([os.sys.executable, str(HERE/"summarize.py"), str(out/"raw.csv"),
                    str(out/"summary.csv")], check=True)
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown")
    metadata = dict(frozen_revision=BASE, extracted_from=source, source_trees=TREES, cpu_model=model,
                    pinned_cpu=cpu, threads=1, flags=flags, kernel=platform.release(),
                    cc=subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
                    clock="CLOCK_MONOTONIC", clock_resolution_ns=round(time.clock_getres(time.CLOCK_MONOTONIC)*1e9),
                    metric="Inclusive warm public SGEMM and three exclusive phase intervals; timing probes incur overhead.")
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print((out/"summary.csv").read_text(), end="")


if __name__ == "__main__":
    main()
