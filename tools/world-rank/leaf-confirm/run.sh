#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
rank=$(cd "$here/.." && pwd)
work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/cob-leaf-confirm.XXXXXXXX")
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE GOTO_NUM_THREADS=1

mkdir -p "$out/vendor-builds" "$work/goldens"
cp "$here"/{panel.csv,protocol.md,geometry.csv,novelty.json} "$out/"
(cd "$out" && sha256sum -c <<'DIGESTS'
204a37e5fd61c9530f01ec7a95e01ea2d09acf719784f198af6cdd6221609eff  panel.csv
81d8142ae0feb265c092e9ed3ed26b523b777deb8a4b9ca9fd70a6cb563114ee  protocol.md
DIGESTS
)
python3 -B "$here/check-geometry.py" | cmp - "$out/geometry.csv"
python3 -B "$rank/ragged-hk/check-fringe-identity.py" "$base" > "$out/algebra.txt"
git -C "$rank" rev-parse HEAD > "$out/benchmark-sha.txt"
git -C "$base" rev-parse HEAD HEAD^{tree} > "$out/production-sha.txt"
lscpu > "$out/lscpu.txt"
getconf PAGE_SIZE > "$out/page-bytes.txt"
gcc --version > "$out/compiler.txt"
printf '%s\n' "$cpu" > "$out/pinned-cpu.txt"
git -C "$rank" ls-files -z 'leaf-confirm/*' 'ragged-hk/*' 'views/worker.c' 'worker.c' 'vendor-public/trace.c' 'signed/trace.c' 'openblas/*' 'vendors/*' 'handles/source/*' 'fastmmw/*' 'release-replay/restore-peers.sh' |
    (cd "$rank" && xargs -0 sha256sum) > "$out/study-source.sha256"
for file in "$vendor"/*-build.txt "$vendor"/tblis-status.txt "$vendor"/onednn/qualified-*.log; do
    if [[ -f $file ]]; then cp "$file" "$out/vendor-builds/"; fi
done
bash "$here/standalone.sh" "$base" "$work" "$out" "$cpu"
bash "$here/vendor-release.sh" "$base" "$vendor" "$work" "$out" "$cpu"
# A candidate's failure is recorded independently of the required released field.
if ! bash "$here/hk-leaf-build.sh" "$base" "$vendor" "$work" "$out" "$cpu" > "$out/hk-leaf-build.log" 2>&1; then
    tail -40 "$out/hk-leaf-build.log" >&2
fi
if [[ $(uname -m) == x86_64 ]]; then
    if bash "$rank/handles/source/build.sh" "$vendor" "$work/handles" > "$out/handle-build.log" 2>&1; then
        touch "$work/handles/qualified"
    else
        tail -40 "$out/handle-build.log" >&2
    fi
    for file in "$work/handles"/{handle-builds.csv,handle-provider.sha256}; do
        if [[ -f $file ]]; then cp "$file" "$out/"; fi
    done
elif [[ -f $here/optional-arm.sh ]]; then
    if ! bash "$here/optional-arm.sh" "$vendor" "$work" "$out" "$cpu" > "$out/optional-arm-build.log" 2>&1; then
        tail -40 "$out/optional-arm-build.log" >&2
    fi
fi
taskset -c "$cpu" python3 -B "$here/study.py" "$vendor" "$work" "$out" "$cpu"
python3 -B "$here/score.py" "$out"

# These diagnostic inputs are outside every registered performance cohort.
# Execute them after the last timed sample. Include all numerical failures.
if bash "$here/stable-build.sh" "$base" "$vendor" "$work" "$out" "$cpu" > "$out/guard-build.log" 2>&1; then
    taskset -c "$cpu" python3 -B "$here/guard-pairs.py" "$work" "$out" "$cpu"
else
    echo 'Guard failed to build; see guard-build.log' >&2
    tail -40 "$out/guard-build.log" >&2
fi
bash "$here/post-dyad.sh" "$base" "$vendor" "$work" "$out" "$cpu"
