#!/usr/bin/env bash
set -euo pipefail
source=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$dir" rev-parse --show-toplevel)
release=a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/grouped.XXXXXXXX")
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
[[ $(git -C "$source" rev-parse HEAD) == "$release" ]]
git -C "$source" diff --quiet HEAD --
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT PYTHONDONTWRITEBYTECODE=1
cp "$dir"/{panel.csv,protocol.md,novelty.json,candidate.sha256} "$out/"
getconf PAGE_SIZE > "$out/page-size.txt"
{ uname -a; lscpu; cc --version; cmake --version; git -C "$repo" rev-parse HEAD; } > "$out/host.txt"
for name in openblas-build.txt aocl-configuration.txt blasfeo-build.txt mkl-build.txt mkl-jit-build.txt libxsmm-build.txt; do
    if [[ -f $vendor/$name ]]; then cp "$vendor/$name" "$out/"; fi
done
for name in qualified-oneshot.log qualified-cached-plan.log; do
    if [[ -f $vendor/onednn/$name ]]; then cp "$vendor/onednn/$name" "$out/onednn-$name"; fi
done
bash "$dir/build.sh" "$source" "$build" "$out" "$cpu" > "$out/grouped-build.log" 2>&1 || { tail -60 "$out/grouped-build.log"; exit 1; }
taskset -c "$cpu" python3 "$dir/routes.py" "$out/panel.csv" "$build" "$out"
taskset -c "$cpu" python3 "$dir/study.py" "$build" "$vendor" "$out" "$cpu"
python3 "$dir/score.py" "$out"
