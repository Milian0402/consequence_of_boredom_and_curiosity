#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?pass an output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cc=${CC:-cc}
common=(-O3 -std=c11 -ffp-contract=off)
"$cc" "${common[@]}" -mcpu=apple-m4 -march=armv8.7-a+sme+sme2 -c "$here/tiles.c" -o "$out/tiles.o"
"$cc" "${common[@]}" -c "$here/plane.c" -o "$out/plane.o"
"$cc" "${common[@]}" -c "$here/pack.c" -o "$out/pack.o"
"$cc" -dynamiclib "$out/tiles.o" "$out/plane.o" "$out/pack.o" -o "$out/apple-fp16.dylib"
"$cc" -O2 "$here/probe-env.c" -o "$out/probe-env"
"$out/probe-env" > "$out/platform.txt"
otool -tvV "$out/tiles.o" | awk '/smstart|smstop|fmopa/{print}' > "$out/sme-instructions.txt"
