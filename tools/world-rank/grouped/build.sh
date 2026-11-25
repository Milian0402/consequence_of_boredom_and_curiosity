#!/usr/bin/env bash
# Compile the pinned release and its one-source-delta challenger from fresh trees.
set -euo pipefail
release=$(cd "$1" && pwd)
work=$(mkdir -p "$2" && cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
cpu=$4
methods=(stream-incumbent grouped-b)
source_options=()
if [[ $# == 5 && $5 == --incumbent-only ]]; then
    methods=(stream-incumbent)
    source_options+=(--incumbent-only)
elif [[ $# != 4 ]]; then
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$here")
cc=${CC:-gcc}
sha(){ sha256sum "$1" | cut -d' ' -f1; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
python3 -B "$here/sources.py" "$release" "$work" "$out" "${source_options[@]}" > "$out/grouped-prepare.txt"
repo=$(git -C "$shared" rev-parse --show-toplevel)
git -C "$repo" show b83033d199137b370c03a5a3d63365f1025162b7:tools/world-rank/challenger-flags.txt > "$out/grouped-dispatch-flags.txt"
"$cc" -std=c11 -O2 "$shared/stream/isa.c" -o "$work/grouped-isa"
taskset -c "$cpu" "$work/grouped-isa" > "$out/grouped-native-features.txt"
pinned=$(taskset -c "$cpu" python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
[[ $pinned == "$cpu" ]]
read -r has2 has512 < "$out/grouped-native-features.txt"
[[ $has2 == 1 && ( $has512 == 0 || $has512 == 1 ) ]]
printf 'implementation,source_revision,source_manifest_sha256,patch_sha256,archive_sha256,installed_archive_sha256,header_sha256,worker_sha256,trace_sha256,flags_sha256,public_symbol\n' > "$out/grouped-builds.csv"
: > "$out/grouped-guard-tests.txt"
for method in "${methods[@]}"; do
    root=$work/$method; src=$root/source; sdk=$root/sdk; obj=$root/objects
    cflags='-march=native -masm=att'
    while IFS= read -r spec; do [[ -z $spec ]] || cflags+=" -D$spec"; done < "$out/grouped-dispatch-flags.txt"
    wrappers=(-Wl,--wrap=cob_sgemm_rowmajor,--wrap=posix_memalign,--wrap=cob_stream_b_kernel_avx2,--wrap=cob_stream_b_kernel_avx512)
    witness_defs=()
    if [[ $method == grouped-b ]]; then
        cflags+=' -DCOB_ENABLE_X86_GROUPED_B=1'
        wrappers+=(-Wl,--wrap=cob_stream_b_grouped_avx2,--wrap=cob_stream_b_grouped_avx512)
        witness_defs+=(-DGROUPED_ENABLED=1)
    fi
    cmake -S "$src" -B "$obj" -DCMAKE_C_COMPILER="$cc" -DCMAKE_C_FLAGS="$cflags" \
        -DCMAKE_BUILD_TYPE=Release -DCOB_BUILD_TESTS=ON -DCOB_BUILD_BENCH=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_LIBDIR=lib -DCOB_ENABLE_X86_STREAM_B=ON \
        -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_ENABLE_X86_MEDIUM_ROUTE=ON \
        > "$out/$method-configure.txt"
    cmake --build "$obj" --parallel 2 > "$out/$method-build.txt" 2>&1 || { tail -60 "$out/$method-build.txt"; exit 1; }
    taskset -c "$cpu" ctest --test-dir "$obj" --output-on-failure > "$out/$method-tests.txt" ||
        { cat "$out/$method-tests.txt"; exit 1; }
    test_line=$(taskset -c "$cpu" "$obj/cob_stream_b_test")
    [[ $test_line =~ ^PASS\ native_width=(32|64)\ AVX2_width=32\ checked=[0-9]+\ max_abs_error= ]]
    printf '%s %s\n' "$method" "$test_line" >> "$out/grouped-guard-tests.txt"
    cmake --install "$obj" --prefix "$sdk" > "$out/$method-install.txt"
    lib=$sdk/lib/libcob_matmul.a; header=$sdk/include/cob_gemm.h
    cmp "$src/include/cob_gemm.h" "$header"
    cmp "$obj/libcob_matmul.a" "$lib"
    python3 -B "$shared/stream-release/flags.py" "$obj/compile_commands.json" cmake-native-on "$out/$method-flags.json"
    worker_flags=(-DRANK_VIEW_RECORD_LD=1 -std=c11 -O3 -march=x86-64 -I"$shared" -I"$sdk/include" -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
    if [[ ${RANK_VIEW_HANDLE:-0} == 1 ]]; then worker_flags+=(-DRANK_VIEW_HANDLE=1); fi
    "$cc" "${worker_flags[@]}" "$shared/views/worker.c" "$lib" -lm -ldl -o "$work/$method.worker"
    "$cc" "${worker_flags[@]}" "${witness_defs[@]}" "$shared/views/worker.c" "$here/witness.c" "${wrappers[@]}" "$lib" -lm -ldl -o "$work/$method.trace.worker"
    for type in archive ordinary trace; do
        file=$lib; [[ $type != ordinary ]] || file=$work/$method.worker; [[ $type != trace ]] || file=$work/$method.trace.worker
        nm --defined-only "$file" > "$root/$type.nm"
        grep -Eq ' T cob_sgemm_rowmajor$' "$root/$type.nm"
        for isa in avx2 avx512; do
            grep -Eq " T cob_stream_b_kernel_$isa$" "$root/$type.nm"
            if [[ $method == grouped-b ]]; then grep -Eq " T cob_stream_b_grouped_$isa$" "$root/$type.nm"
            elif grep -Eq " T cob_stream_b_grouped_$isa$" "$root/$type.nm"; then exit 1; fi
        done
    done
    if grep -Eq ' (T|t) __wrap_|cob_stream_b_count|grouped_trace' "$root/ordinary.nm"; then exit 1; fi
    for symbol in cob_sgemm_rowmajor posix_memalign cob_stream_b_kernel_avx2 cob_stream_b_kernel_avx512; do
        grep -Eq " T __wrap_$symbol$" "$root/trace.nm"
    done
    if [[ $method == grouped-b ]]; then
        for isa in avx2 avx512; do grep -Eq " T __wrap_cob_stream_b_grouped_$isa$" "$root/trace.nm"; done
    fi
    ldd "$work/$method.worker" > "$out/$method-libraries.txt"
    if grep -Ei 'lib(openblas|blas|mkl|gomp|iomp)|not found' "$out/$method-libraries.txt"; then exit 1; fi
    manifest=$(python3 -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["methods"][sys.argv[2]]["sha256"])' "$out/grouped-sources.json" "$method")
    patch_id=none; [[ $method != grouped-b ]] || patch_id=$(sha "$here/candidate.patch")
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,cob_sgemm_rowmajor\n' "$method" \
        "$(git -C "$release" rev-parse HEAD)" "$manifest" "$patch_id" "$(sha "$obj/libcob_matmul.a")" "$(sha "$lib")" \
        "$(sha "$header")" "$(sha "$work/$method.worker")" "$(sha "$work/$method.trace.worker")" "$(sha "$out/$method-flags.json")" >> "$out/grouped-builds.csv"
done
if [[ ${#methods[@]} == 1 ]]; then exit 0; fi
python3 -B - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
before=json.loads((root/"stream-incumbent-flags.json").read_text())
after=json.loads((root/"grouped-b-flags.json").read_text())
opt="-DCOB_ENABLE_X86_GROUPED_B=1"
for name,flags in after.items():
    if opt in before[name] or flags.count(opt)!=1 or [f for f in flags if f!=opt]!=before[name]:
        raise ValueError("Unexpected compiler difference for "+name)
PY
