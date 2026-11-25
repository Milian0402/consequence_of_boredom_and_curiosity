#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
base=$(cd "$1" && pwd);work=$2;out=$3;cpu=$4
here=$(cd "$(dirname "$0")" && pwd);cc=${CC:-gcc}
shared="$here/../ragged-hk"
sha=f322a9ab58454762563b97323fdc5f29016cc39f
[[ $(git -C "$base" rev-parse HEAD) == "$sha" && $(git -C "$base" rev-parse HEAD^{tree}) == db8d3b50584af7632355314888857ddafa0a41d7 ]]
git -C "$base" diff --quiet HEAD --
case $(uname -m) in x86_64) tuning=-march=native;; aarch64) tuning=-mcpu=native;; *) exit 1;; esac
export ZERO_AR_DATE=1
mkdir -p "$work" "$out/standalone"
for method in released-hk15 ragged-hk15; do
    home=$work/$method; proof=$out/standalone/$method; src=$home/source
    mkdir -p "$src" "$proof"
    git -C "$base" archive HEAD CMakeLists.txt Makefile src include tests bench third_party | tar -x -C "$src"
    cmopts=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$cc" "-DCMAKE_C_FLAGS=$tuning" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCOB_BUILD_BENCH=OFF -DCOB_BLAS_LEAF=NONE -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON
        -DCOB_ENABLE_HK15=ON -DCMAKE_INSTALL_LIBDIR=lib)
    traceflags=(-Wl,--wrap=cob_sgemm_rowmajor)
    if [[ $method == ragged-hk15 ]]; then
        taskset -c "$cpu" python3 -B "$src/tests/check_hk15_ragged.py" > "$proof/remainder-fixtures.txt"
        cmopts+=(-DCOB_ENABLE_HK15_RAGGED=ON);traceflags+=(-DHK_RAGGED_TRACE=1 -Wl,--wrap=cob_hk15_ragged_finish)
    fi
    (cd "$src" && sha256sum CMakeLists.txt Makefile include/*.h src/*.c src/*.h src/*.inc) > "$proof/source.sha256"
    cmake -S "$src" -B "$home/build" "${cmopts[@]}" -DCOB_BUILD_TESTS=ON > "$proof/configure.log"
    cmake --build "$home/build" --parallel 3 > "$proof/build.log" 2>&1 || { tail -50 "$proof/build.log";exit 1; }
    taskset -c "$cpu" ctest --test-dir "$home/build" --output-on-failure > "$proof/tests.log" || { cat "$proof/tests.log";exit 1; }
    cmake --install "$home/build" --prefix "$home/sdk" > "$proof/install.log"
    archive=$home/sdk/lib/libcob_matmul.a
    cmp "$archive" "$home/build/libcob_matmul.a";cmp "$home/sdk/include/cob_gemm.h" "$base/include/cob_gemm.h"
    cf=(-std=c11 -O3 "$tuning" -ffunction-sections -fdata-sections -Wl,--gc-sections -I"$home/sdk/include"
        -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
    "$cc" "${cf[@]}" "$shared/worker.c" "$archive" -lcrypto -lm -ldl -o "$home/ordinary.worker"
    if [[ $method == released-hk15 && $(uname -m) == x86_64 ]]; then
        # The common timer emits native create status and preparation cost outside SGEMM.
        "$cc" "${cf[@]}" -DRANK_VIEW_HANDLE=1 "-DVIEW_AUDIT_HEADER=\"$here/handle-proof.h\"" \
            "$shared/worker.c" "$archive" -lcrypto -lm -ldl -o "$work/handle.worker"
    fi
    "$cc" -std=gnu11 -O2 "$tuning" -I"$home/sdk/include" "$here/dyad.c" "$archive" -lm -ldl -pthread -o "$work/$method.dyad"
    cp -a "$src" "$home/diagnostic-source"
    python3 -B "$shared/instrument-hk.py" "$home/diagnostic-source" > "$proof/instrument.log"
    cmake -S "$home/diagnostic-source" -B "$home/trace-build" "${cmopts[@]}" -DCOB_BUILD_TESTS=OFF >> "$proof/instrument.log"
    cmake --build "$home/trace-build" --parallel 3 --target cob_matmul >> "$proof/instrument.log" 2>&1 || { tail -50 "$proof/instrument.log";exit 1; }
    "$cc" "${cf[@]}" "${traceflags[@]}" "$shared/worker.c" "$shared/hk-witness.c" "$home/trace-build/libcob_matmul.a" -lcrypto -lm -ldl -o "$home/trace.worker"
    nm --defined-only "$home/ordinary.worker" > "$home/symbols.txt"
    grep -E ' T cob_sgemm_rowmajor$' "$home/symbols.txt" > "$proof/symbols.txt"
    if grep -E ' T (__wrap_|hk_trace_)' "$home/symbols.txt";then exit 1;fi
    ldd "$home/ordinary.worker" > "$proof/linked.txt"
    if grep -E 'lib(mkl|blas|openblas|armpl|gomp|iomp)|not found' "$proof/linked.txt";then exit 1;fi
    cp "$home/build/compile_commands.json" "$proof/compile-commands.json"
    sha256sum "$archive" "$home/ordinary.worker" "$home/trace.worker" "$home/trace-build/libcob_matmul.a" "$home/sdk/include/cob_gemm.h" > "$proof/products.sha256"
done
