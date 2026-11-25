#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
work=$(cd "$3" && pwd)
out=$(cd "$4" && pwd)
cpu=$5
here=$(cd "$(dirname "$0")/.." && pwd)
rank=$(dirname "$here")
core=$here/core
cc=${CC:-cc}
candidates=$work/candidates
mkdir -p "$candidates" "$out/build"
[[ $(git -C "$base" rev-parse HEAD) == f322a9ab58454762563b97323fdc5f29016cc39f ]]
case $(uname -m) in
    x86_64) tuning=-march=native; scoped=-DCOB_ENABLE_X86_MEDIUM_ROUTE=ON; stream=-DCOB_ENABLE_X86_STREAM_B=ON ;;
    aarch64) tuning=-mcpu=native; scoped=-DCOB_ENABLE_N2_SMALL=ON; stream=-DCOB_ENABLE_X86_STREAM_B=OFF ;;
    *) exit 2;;
esac
flags=(-std=c11 -O3 -fno-fast-math -fPIC -ffunction-sections -fdata-sections)
worker_flags=(-Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor -I"$base/include" -I"$rank")
for method in cob-released cob-released-hk15; do
    root=$work/$method
    enabled=OFF
    [[ $method != cob-released-hk15 ]] || enabled=ON
    cmake -S "$base" -B "$root" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_COMPILER="$cc" "-DCMAKE_C_FLAGS=$tuning" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON -DCOB_BLAS_LEAF=NONE -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_ENABLE_HK15="$enabled" "$scoped" "$stream" > "$out/build/$method-configure.log"
    cmake --build "$root" --parallel 3 > "$out/build/$method-build.log"
    taskset -c "$cpu" ctest --test-dir "$root" --output-on-failure > "$out/build/$method-tests.log"
    cp "$root/compile_commands.json" "$out/build/$method-compile.json"
    "$cc" "${flags[@]}" "$tuning" "${worker_flags[@]}" -Wl,--gc-sections "$here/worker.c" "$root/libcob_matmul.a" -lcrypto -ldl -lm -o "$work/$method.worker"
done
cp "$work/cob-released.worker" "$work/ordinary.worker"
archive=$work/cob-released/libcob_matmul.a
"$cc" "${flags[@]}" -c "$core/fp32.c" -o "$work/fp32.o"
"$cc" "${flags[@]}" -c "$core/precision.c" -o "$work/precision.o"
"$cc" -O2 -std=c11 "$core/fused_probe.c" -ldl -lm -o "$work/fused-probe"
# Macroscopic decisions and registration are external to the ordinary measured wrappers.
build_family(){
    local prefix=$1 lib=$2 backend=$3
    local tag=${prefix:-onednn-}
    "$cc" -shared -Wl,-z,defs "$work/precision.o" "$work/fp32.o" "$archive" -L"$backend" "-Wl,-rpath,$backend" -l:"$lib" -lm -o "$candidates/${tag}core.so" || return 1
    local name mode
    for name in mixed1 mixed3 mixed6 mixed9 mixed6-bands; do
        case "$name" in mixed6-bands) mode=16;; *) mode=${name#mixed};; esac
        "$cc" "${flags[@]}" -shared -Wl,-z,defs -DPRECISION_MODE="$mode" "$core/plugin.c" -L"$candidates" '-Wl,-rpath,$ORIGIN' -l:"${tag}core.so" -o "$candidates/$prefix$name.so" || return 1
    done
}
if bash "$here/backend/build.sh" "$vendor/onednn/prefix" "$work/onednn-bf16" > "$out/build/onednn-bf16.log" 2>&1;then
    cp "$work/onednn-bf16/bf16-build.txt" "$out/build/"
    build_family '' libbf16_backend.so "$work/onednn-bf16"
else tail -35 "$out/build/onednn-bf16.log" >&2; fi
if [[ $(uname -m) == aarch64 && -f $here/acl-backend/build.sh ]]; then
    if bash "$here/acl-backend/build.sh" "$vendor" "$work/arm-bf16" > "$out/build/arm-bf16.log" 2>&1;then
        build_family arm- libacl_bf16_backend.so "$work/arm-bf16"
        find "$work/arm-bf16" -maxdepth 1 -type f \( -name '*.json' -o -name '*.sha256' -o -name '*.txt' \) -exec cp {} "$out/build/" \;
    else tail -35 "$out/build/arm-bf16.log" >&2;fi
fi
if [[ $(uname -m) == x86_64 ]];then
    for kind in ordinary trace; do
        opt=();dest=$candidates/fused6-amx.so
        if [[ $kind == trace ]];then opt=(-DCOB_AMX_TRACE=1);dest=$candidates/fused6-amx.trace.so;fi
        "$cc" "${flags[@]}" -shared -Wl,-z,defs -DPRECISION_MODE=6 -DPRECISION_FUSED=1 -DAMX_FUSED6_BENCH_ABI=1 "${opt[@]}" "$core/plugin.c" "$here/fused-amx/amx_fused6.c" "$here/fused-amx/amx_compute.c" "$work/fp32.o" "$archive" -lm -o "$dest"
    done
    bash "$here/fused-amx/tests/check.sh" > "$out/build/fused-smoke.txt" 2>&1
    objdump -d "$candidates/fused6-amx.so" | sed -n '/tdpbf16ps/p' > "$out/build/fused-instructions.txt"
    test "$(wc -l < "$out/build/fused-instructions.txt")" -ge 6
fi
sha256sum "$archive" "$work/ordinary.worker" "$work/cob-released-hk15.worker" "$candidates"/*.so > "$out/build/products.sha256"
