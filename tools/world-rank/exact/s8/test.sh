#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
sdk=$(cd "${1:?oneDNN prefix}" && pwd)
out=$(mkdir -p "${2:?local diagnostic output}" && cd "$2" && pwd)
cxx=${CXX:-c++}
opts=(-std=c++17 -O3 -fno-fast-math -Wall -Wextra -Werror -fPIC -I"$sdk/include" "-DEXACT_INT8_DNNL_PATH=\"$sdk/lib/libdnnl.so.3\"")
"$cxx" "${opts[@]}" -shared "$src/checked-int8.cpp" -ldl -pthread -o "$out/ordinary.so"
"$cxx" "${opts[@]}" -DEXACT_INT8_TEST_FAULTS=1 -shared "$src/checked-int8.cpp" -ldl -pthread -o "$out/diagnostic.so"
if readelf -d "$out/ordinary.so" | grep NEEDED | grep 'libdnnl';then exit 1;fi
"$cxx" -O2 -std=c++17 -Wall -Wextra -Werror "$src/smoke.cpp" -ldl -o "$out/check"
"$out/check" "$out/ordinary.so" auto > "$out/host-transfer.txt"
if [[ ${3:-} == --qemu-n2 ]];then
 [[ $(uname -m) == aarch64 ]]
 qemu-aarch64 -cpu neoverse-n2 "$out/check" "$out/ordinary.so" positive > "$out/emulated-native-operation.txt"
 qemu-aarch64 -cpu neoverse-n2 "$out/check" "$out/diagnostic.so" faults > "$out/emulated-failures.txt"
 EXACT_INT8_DNNL_PATH=/absent/no-library.so qemu-aarch64 -cpu neoverse-n2 "$out/check" "$out/ordinary.so" missing > "$out/emulated-missing-provider.txt"
fi
