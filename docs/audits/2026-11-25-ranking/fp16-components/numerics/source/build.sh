#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
archive=${1:?pass the built release libcob_matmul.a}
output=${2:?pass a scratch build directory}
apple=${3:-"$here/../../apple"}
repo=${4:-$here}
while [[ ! -f "$repo/tools/world-rank/precision/core/fp32.c" ]]; do
    parent=$(dirname "$repo")
    [[ "$parent" != "$repo" ]] || { echo "Pass the repository root as argument four" >&2; exit 1; }
    repo=$parent
done
mkdir -p "$output"
cc=${CC:-clang}
flags=(-O3 -std=c11 -fPIC -fno-fast-math -ffp-contract=off)
"$cc" "${flags[@]}" -c "$repo/tools/world-rank/precision/core/fp32.c" -o "$output/fp32.o"
"$cc" "${flags[@]}" -c "$here/primitive.c" -o "$output/scalar.o"
"$cc" "${flags[@]}" -march=armv8.4-a+fp16fml -DWIDEN -c "$here/primitive.c" -o "$output/fmlal.o"
"$cc" "${flags[@]}" -march=armv8.4-a+fp16fml -c "$here/pairwise.c" -o "$output/pairwise.o"
"$cc" "${flags[@]}" -c "$here/sme-bridge.c" -o "$output/sme.o"
"$cc" "${flags[@]}" -mcpu=apple-m4 -march=armv8.7-a+sme+sme2 -c "$apple/tiles.c" -o "$output/tiles.o"
"$cc" "${flags[@]}" -c "$apple/plane.c" -o "$output/plane.o"
for e in scalar fmlal sme pairwise;do
    extra=(); tag=$e
    if [[ $e == sme ]]; then extra=("$output/tiles.o" "$output/plane.o");tag=apple-sme-f16f32;fi
    if [[ $e == pairwise ]];then tag=pairwise-fmlal;fi
    for balance in 0 1;do
        "$cc" "${flags[@]}" -DDEPTH_BALANCE="$balance" "-DENGINE=\"$tag\"" -dynamiclib "$here/fp16.c" "$output/$e.o" "${extra[@]}" "$output/fp32.o" "$archive" -o "$output/$e-$balance.dylib"
    done
done
