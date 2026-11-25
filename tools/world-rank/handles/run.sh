#!/usr/bin/env bash
set -euo pipefail
source_root=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$dir")
repo=$(git -C "$dir" rev-parse --show-toplevel)
[[ $(git -C "$source_root" rev-parse HEAD) == a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6 ]]
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/explicit.XXXXXXXX")
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT PYTHONDONTWRITEBYTECODE=1 RANK_VIEW_HANDLE=1
cp "$dir"/{panel.csv,protocol.md} "$out/"
getconf PAGE_SIZE > "$out/page-size.txt"
{ uname -a; lscpu; cc --version; cmake --version; git -C "$repo" rev-parse HEAD; } > "$out/host.txt"
(cd "$dir" && find source -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/handle-source.sha256")
for name in openblas-build.txt aocl-configuration.txt blasfeo-build.txt mkl-build.txt mkl-jit-build.txt libxsmm-build.txt mkl-identity.sha256 mkl-jit-identity.sha256 mkl-backend-imports.txt mkl-jit-backend-imports.txt; do
    if [[ -f $vendor/$name ]]; then cp "$vendor/$name" "$out/"; fi
done
for name in qualified-oneshot.log qualified-cached-plan.log; do
    if [[ -f $vendor/onednn/$name ]]; then cp "$vendor/onednn/$name" "$out/onednn-$name"; fi
done
bash "$shared/grouped/build.sh" "$source_root" "$build" "$out" "$cpu" --incumbent-only > "$out/incumbent-build.log" 2>&1 || { tail -70 "$out/incumbent-build.log"; exit 1; }
taskset -c "$cpu" python3 "$shared/prefix/incumbent.py" "$build" "$out"
bash "$dir/source/build.sh" "$vendor" "$build/handles" > "$out/handle-build.log" 2>&1 || { tail -70 "$out/handle-build.log"; exit 1; }
cp "$build/handles"/{handle-builds.csv,handle-provider.sha256} "$out/"
for file in "$build/handles"/*.imports "$build/handles"/*.dependencies "$build/handles"/*.symbols "$build/handles"/*.dynamic; do
    if [[ -f $file ]]; then cp "$file" "$out/"; fi
done
taskset -c "$cpu" python3 "$dir/source/routes.py" "$out/panel.csv" "$vendor" "$build/handles" "$out"
for method in mkl-handle-oneshot mkl-handle-prepared; do
    cp "$build/handles/$method.so" "$vendor/$method.so"
    cmp "$build/handles/$method.so" "$vendor/$method.so"
done
taskset -c "$cpu" python3 "$dir/study.py" "$build" "$vendor" "$out" "$cpu"
python3 "$dir/score.py" "$out"
