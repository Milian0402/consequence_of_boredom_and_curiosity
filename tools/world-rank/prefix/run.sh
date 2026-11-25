#!/usr/bin/env bash
set -euo pipefail
source=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$dir")
repo=$(git -C "$dir" rev-parse --show-toplevel)
[[ $(git -C "$source" rev-parse HEAD) == a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6 ]]
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/prefix.XXXXXXXX")
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT PYTHONDONTWRITEBYTECODE=1
export RANK_PREFIX_MKL="$vendor/mkl.so"
cp "$dir"/{panel.csv,protocol.md,novelty.json} "$out/"
getconf PAGE_SIZE > "$out/page-size.txt"
{ uname -a; lscpu; cc --version; cmake --version; git -C "$repo" rev-parse HEAD; } > "$out/host.txt"
(cd "$dir" && find source -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/prefix-source.sha256")
for name in openblas-build.txt aocl-configuration.txt blasfeo-build.txt mkl-build.txt mkl-jit-build.txt libxsmm-build.txt mkl-identity.sha256 mkl-jit-identity.sha256 mkl-backend-imports.txt mkl-jit-backend-imports.txt; do
    if [[ -f $vendor/$name ]]; then cp "$vendor/$name" "$out/"; fi
done
for name in qualified-oneshot.log qualified-cached-plan.log; do
    if [[ -f $vendor/onednn/$name ]]; then cp "$vendor/onednn/$name" "$out/onednn-$name"; fi
done
bash "$shared/grouped/build.sh" "$source" "$build" "$out" "$cpu" --incumbent-only > "$out/incumbent-build.log" 2>&1 || { tail -70 "$out/incumbent-build.log"; exit 1; }
bash "$dir/source/build.sh" "$vendor" "$build/prefix" > "$out/prefix-build.log" 2>&1 || { tail -70 "$out/prefix-build.log"; exit 1; }
cp "$build/prefix"/{prefix-builds.csv,prefix-zero-copy.csv} "$out/"
taskset -c "$cpu" python3 "$dir/incumbent.py" "$build" "$out"
taskset -c "$cpu" python3 "$dir/source/routes.py" "$out/panel.csv" "$vendor" "$build/prefix" "$out"
taskset -c "$cpu" python3 "$dir/parity.py" "$build" "$vendor" "$out"
if [[ -d $dir/explicit-jit ]]; then
    if taskset -c "$cpu" bash "$dir/explicit-jit/run.sh" "$out/panel.csv" "$vendor" "$out" > "$out/explicit-jit.log" 2>&1; then
        echo pass > "$out/explicit-jit-probe-status.txt"
    else
        echo unavailable > "$out/explicit-jit-probe-status.txt"
        tail -20 "$out/explicit-jit.log"
    fi
fi
taskset -c "$cpu" python3 "$dir/study.py" "$build" "$vendor" "$out" "$cpu"
python3 "$dir/score.py" "$out"
