#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$dir")
root=$(git -C "$dir" rev-parse --show-toplevel)
release=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
scratch=$(mktemp -d)
test "$(uname -m)" = aarch64
rev=a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6
test "$(git -C "$release" rev-parse HEAD)" = "$rev"
patch=71e1c7a3ff3c3b807c0d76b89f2c067fdbd9d96261309b7e390110dcfa3f42be
panel=5e470c40de6e63b6debbb96afd146eda58de83e50ef3da1e2f19186a8990b4c4
printf '%s  %s\n' "$patch" "$dir/candidate.patch" "$panel" "$dir/panel.csv" | sha256sum --check --status
cp "$dir"/{protocol.md,panel.csv,score.py} "$out/"
jq -n --arg release "$rev" --arg patch "$patch" --arg panel "$panel" '{release:$release,patch_sha256:$patch,panel_sha256:$panel}' > "$out/source.json"
git -C "$root" rev-parse HEAD > "$out/benchmark-revision.txt"
getconf PAGE_SIZE > "$out/page-size.txt"
{ uname -a; lscpu; gcc --version; } > "$out/host.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT
cpu=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
export PYTHONDONTWRITEBYTECODE=1
digest(){ sha256sum "$1" | cut -d' ' -f1; }
printf 'method,archive_sha256,worker_sha256,trace_sha256\n' > "$out/binaries.csv"
tuned="-mcpu=native -DCOB_ENABLE_N2_SMALL=1"
while read -r setting; do
    if [[ -n $setting ]]; then tuned+=" -D$setting"; fi
done < "$shared/challenger-flags.txt"
for method in incumbent candidate; do
    src="$scratch/$method-source"; build="$scratch/$method-build"; sdk="$scratch/$method-sdk"
    mkdir -p "$src"
    git -C "$release" archive "$rev" | tar xf - -C "$src"
    opts=(); wraps=(-Wl,--wrap=cob_sgemm_rowmajor)
    if [[ $method == candidate ]]; then
        (cd "$src" && git apply "$dir/candidate.patch")
        opts+=(-DCOB_ENABLE_ARM_STREAM_B=ON)
        wraps+=(-DARM_STREAM_ENABLED=1 -Wl,--wrap=cob_arm_stream_b_product)
    fi
    cmake -S "$src" -B "$build" -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="$tuned" \
        -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON \
        -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON "${opts[@]}" > "$out/$method-configure.log"
    cmake --build "$build" --parallel 3 > "$out/$method-build.log"
    taskset -c "$cpu" ctest --test-dir "$build" --output-on-failure > "$out/$method-tests.log" || { cat "$out/$method-tests.log"; exit 1; }
    cmake --install "$build" --prefix "$sdk" > "$out/$method-install.log"
    cp "$build/compile_commands.json" "$out/$method-flags.json"
    archive="$sdk/lib/libcob_matmul.a"
    cmp "$archive" "$build/libcob_matmul.a"
    cmp "$sdk/include/cob_gemm.h" "$src/include/cob_gemm.h"
    flags=(-std=c11 -O3 -ffunction-sections -fdata-sections -Wl,--gc-sections "-I$sdk/include" "-I$dir"
      '-DVIEW_AUDIT_HEADER="audit.h"' -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
    gcc "${flags[@]}" "$shared/views/worker.c" "$archive" -ldl -lm -o "$scratch/$method.worker"
    gcc "${flags[@]}" "$shared/views/worker.c" "$dir/witness.c" "${wraps[@]}" "$archive" -ldl -lm -o "$scratch/$method.trace"
    nm "$scratch/$method.worker" > "$scratch/$method.nm"
    grep -Eq ' T cob_sgemm_rowmajor$' "$scratch/$method.nm"
    if grep -Eq ' [Tt] __wrap_' "$scratch/$method.nm"; then exit 1; fi
    printf '%s,%s,%s,%s\n' "$method" "$(digest "$archive")" "$(digest "$scratch/$method.worker")" "$(digest "$scratch/$method.trace")" >> "$out/binaries.csv"
done
python3 - "$scratch" > "$out/clone-probe.txt" <<'PYPROBE'
from pathlib import Path
import os, signal, subprocess, sys
for method in ("incumbent","candidate"):
    ret=subprocess.run([Path(sys.argv[1])/(method+".worker")],env=dict(os.environ,ARM_ASSERT_NO_CLONE="1"))
    if ret.returncode != -signal.SIGSYS:
        raise RuntimeError(f"{method} clone probe returned {ret.returncode}")
    print(method, "SIGSYS")
PYPROBE
taskset -c "$cpu" python3 "$dir/routes.py" "$scratch" "$out"
taskset -c "$cpu" python3 "$dir/study.py" "$scratch" "$vendor" "$out"
python3 "$dir/score.py" "$out"
