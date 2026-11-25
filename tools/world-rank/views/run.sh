#!/usr/bin/env bash
set -euo pipefail
release=$(cd "$1" && pwd); vendor=$(cd "$2" && pwd); out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd); shared=$(dirname "$dir")
root=$(git -C "$shared" rev-parse --show-toplevel); builds=$(mktemp -d)
getconf PAGE_SIZE > "$out/page-size.txt"
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$out"
cp "$dir"/{panel.csv,protocol.md,novelty.json,score.py} "$out/"
sha256sum "$dir"/{panel.csv,protocol.md,novelty.json,worker.c,run.sh,study.py,score.py} > "$out/benchmark-sha256.txt"
{ uname -a; lscpu; gcc --version; getconf PAGE_SIZE; git -C "$root" rev-parse HEAD; } > "$out/host.txt"
python3 "$shared/frontier/sources.py" "$release" "$builds" "$out"
gcc -std=c11 -O2 "$shared/stream/isa.c" -o "$builds/isa"
taskset -c "$cpu" "$builds/isa" > "$out/isa.txt"
printf 'implementation,header,archive,worker,trace\n' > "$out/binaries.csv"
digest(){ sha256sum "$1" | cut -d' ' -f1; }
for method in default stream packed; do
  src=$builds/frontier-incumbent/source; mode=OFF; cflags='-march=x86-64 -masm=att'
  stamp=$builds/$method; sdk=$stamp/sdk; objects=$stamp/build; wrappers=(); defs=(); flags_name=cmake-default
  if [[ $method != default ]]; then
    mode=ON; flags_name=cmake-att
    wrappers=(-Wl,--wrap=cob_sgemm_rowmajor,--wrap=posix_memalign,--wrap=cob_stream_b_kernel_avx2,--wrap=cob_stream_b_kernel_avx512)
  fi
  if [[ $method == packed ]]; then
    src=$builds/stream-packed-a/source; cflags+=' -DCOB_ENABLE_X86_PACKED_A_FRONTIER=1'; defs=(-DFRONTIER_PACKED=1)
    wrappers+=(-Wl,--wrap=cob_stream_b_packed_avx2,--wrap=cob_stream_b_packed_avx512)
  fi
  cmake -S "$src" -B "$objects" -DCMAKE_C_COMPILER=gcc -DCMAKE_C_FLAGS="$cflags" -DCMAKE_BUILD_TYPE=Release -DCOB_BUILD_TESTS=ON -DCOB_BUILD_BENCH=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_LIBDIR=lib -DCOB_ENABLE_X86_STREAM_B="$mode" > "$out/$method-configure.log"
  cmake --build "$objects" --parallel 2 > "$out/$method-build.log"
  taskset -c "$cpu" ctest --test-dir "$objects" --output-on-failure > "$out/$method-tests.log" || { cat "$out/$method-tests.log"; exit 1; }
  cmake --install "$objects" --prefix "$sdk" > "$out/$method-install.log"
  python3 "$shared/stream-release/flags.py" "$objects/compile_commands.json" "$flags_name" "$out/$method-flags.json"
  archive=$sdk/lib/libcob_matmul.a; header=$sdk/include/cob_gemm.h
  cmp "$archive" "$objects/libcob_matmul.a"; cmp "$header" "$src/include/cob_gemm.h"
  flags=(-std=c11 -O3 -march=x86-64 -ffunction-sections -fdata-sections -Wl,--gc-sections -I"$sdk/include" -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
  gcc "${flags[@]}" "$dir/worker.c" "$archive" -ldl -lm -o "$builds/$method.worker"
  nm --defined-only "$builds/$method.worker" > "$stamp/ordinary.nm"
  grep -Eq ' T cob_sgemm_rowmajor$' "$stamp/ordinary.nm"
  if grep -E ' (T|t) __wrap_' "$stamp/ordinary.nm"; then exit 1; fi
  trace=none
  if [[ $method != default ]]; then
    gcc "${flags[@]}" "${defs[@]}" "$dir/worker.c" "$shared/frontier/witness.c" "${wrappers[@]}" "$archive" -ldl -lm -o "$builds/$method.trace"
    trace=$(digest "$builds/$method.trace")
  else
    ! grep -E ' T cob_stream_b_(kernel|packed)' "$stamp/ordinary.nm" || exit 1
  fi
  printf '%s,%s,%s,%s,%s\n' "$method" "$(digest "$header")" "$(digest "$archive")" "$(digest "$builds/$method.worker")" "$trace" >> "$out/binaries.csv"
done
gcc -O3 -std=c11 -fPIC -shared $(pkg-config --cflags openblas) "$shared/openblas.c" $(pkg-config --libs openblas) -o "$vendor/openblas.so"
cp "$vendor"/{mkl-build.txt,mkl-jit-build.txt,aocl-configuration.txt} "$out/"
pkg-config --modversion openblas > "$out/openblas-version.txt"
for name in openblas mkl mkl-jit aocl; do
  ldd "$vendor/$name.so" > "$out/$name-dependencies.txt"
  printf '%s,,,%.64s,\n' "$name" "$(digest "$vendor/$name.so")" >> "$out/binaries.csv"
done
taskset -c "$cpu" python3 "$dir/study.py" "$builds" "$vendor" "$out" "$cpu"
python3 "$dir/score.py" "$out"
