#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$1" && pwd)
build=$(mkdir -p "$2" && cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$repo/tools/world-rank/stream
study=$repo/tools/world-rank
cc=${CC:-cc}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
python3 -B "$dir/prepare.py" "$repo" "$build" "$out" > "$out/source-check.txt"
printf '%s\n' 'implementation,isa,variant,status,reason,flags,source_sha256,worker_sha256,trace_worker_sha256,object_sha256,trace_source_sha256' > "$out/builds.csv"
sha(){ sha256sum "$1" | cut -d' ' -f1; }
common=(-O3 -std=c11 -fPIC -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_X86_MEDIUM_ROUTE=1)
# Preserve the same frozen release dispatch switches used by the preceding medium study.
git -C "$repo" show b83033d199137b370c03a5a3d63365f1025162b7:tools/world-rank/challenger-flags.txt > "$out/dispatch-flags.txt"
while IFS= read -r f; do [[ -z $f ]] || common+=("-D$f"); done < "$out/dispatch-flags.txt"
"$cc" -std=c11 -O2 "$dir/isa.c" -o "$build/cpu-isa"
taskset -c "$cpu" "$build/cpu-isa" > "$out/native-features.txt"
read -r has_avx2 has_avx512 < "$out/native-features.txt"
for target in native avx2 avx512; do
    flags=(-march=native)
    case $target in
        native) names=(stream-incumbent stream-native) ;;
        avx2) flags+=(-mno-avx512f -mprefer-vector-width=256 -DCOB_DISABLE_X86_AVX512=1); names=(stream-avx2-incumbent stream-avx2) ;;
        avx512) flags+=(-mavx512f -mprefer-vector-width=512); names=(stream-avx512-incumbent stream-avx512) ;;
    esac
    if [[ $has_avx2 != 1 || ( $target == avx512 && $has_avx512 != 1 ) ]]; then
        for n in "${names[@]}"; do printf '%s,%s,,unsupported,required_cpu_feature_absent,,,,,\n' "$n" "$target" >> "$out/builds.csv"; done
        [[ $target != native ]] || exit 1
        continue
    fi
    "$cc" "${flags[@]}" -dM -E -x c /dev/null > "$out/$target-macros.txt"
    grep -q '^#define __AVX2__ 1$' "$out/$target-macros.txt"
    grep -q '^#define __FMA__ 1$' "$out/$target-macros.txt"
    isa=avx2
    if grep -q '^#define __AVX512F__ 1$' "$out/$target-macros.txt" && [[ $target != avx2 ]]; then isa=avx512; fi
    if [[ $target != native && $isa != "$target" ]]; then exit 1; fi
    for mode in 0 1; do
        name=${names[mode]}
        source=$build/release
        variant=incumbent
        extra=()
        if ((mode)); then source=$build/candidate; variant=candidate; extra+=(-DCOB_ENABLE_X86_STREAM_B=1); fi
        cflags=("${common[@]}" "${flags[@]}" "${extra[@]}" -I"$source/include")
        "$cc" "${cflags[@]}" -c "$source/src/gemm.c" -o "$build/$name.o"
        traceflags=()
        if ((mode)); then traceflags+=(-DCOB_STREAM_B_TRACE=1); fi
        "$cc" "${cflags[@]}" "${traceflags[@]}" -c "$source.trace/src/gemm.c" -o "$build/$name.trace.o"
        wflags=(-std=c11 -O3 "${flags[@]}" -I"$source/include" -I"$study" -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
        "$cc" "${wflags[@]}" "$study/worker.c" "$build/$name.o" -lm -ldl -o "$build/$name.worker"
        "$cc" "${wflags[@]}" -DSTREAM_CANDIDATE=$mode "$study/worker.c" "$dir/witness.c" "$build/$name.trace.o" -Wl,--wrap=cob_sgemm_rowmajor -lm -ldl -o "$build/$name.trace.worker"
        nm --defined-only "$build/$name.worker" > "$build/$name.nm"
        grep -E '[[:space:]]T cob_sgemm_rowmajor$' "$build/$name.nm"
        if grep -Eq 'stream_trace_|cob_stream_b_count|__wrap_cob_sgemm_rowmajor' "$build/$name.nm"; then exit 1; fi
        if ((mode)); then
            "$cc" "${cflags[@]}" -DCOB_STREAM_B_TRACE=1 "-DMEDIUM_GEMM_SOURCE=\"$source/src/gemm.c\"" "$dir/check.c" -lm -o "$build/$name.check"
            taskset -c "$cpu" "$build/$name.check" | tee "$out/$name-fixtures.txt"
        fi
        printf '%s,%s,%s,qualified,public_entry_built_and_fixture_passed,%s,%s,%s,%s,%s,%s\n' "$name" "$isa" "$variant" "${common[*]} ${flags[*]} ${extra[*]}" \
            "$(sha "$source/src/gemm.c")" "$(sha "$build/$name.worker")" "$(sha "$build/$name.trace.worker")" "$(sha "$build/$name.o")" "$(sha "$source.trace/src/gemm.c")" >> "$out/builds.csv"
    done
done
