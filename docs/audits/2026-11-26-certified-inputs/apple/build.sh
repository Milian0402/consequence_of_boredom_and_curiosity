#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?pass the output directory}; mkdir -p "$out";out=$(cd "$out" && pwd)
cc=${CC:-cc}
"$cc" -D_DARWIN_C_SOURCE -std=c11 -mcpu=apple-m4 -march=armv8.7-a+sme+sme2 -O3 \
    -fno-fast-math -ffp-contract=off -dynamiclib -framework Accelerate \
    "$here/exact.c" "$here/bitset.c" -Wl,-install_name,@rpath/exact.dylib -o "$out/exact.dylib"
otool -tvV "$out/exact.dylib" | awk '/smopa|smstart|smstop/{print}' > "$out/instructions.txt"
