#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "${1:?pass a local COB repository with the baseline commit}" && pwd)
out=$(mkdir -p "${2:?pass an output directory}" && cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
base=85a007747ff54241f47c4754216e0a65748f2847
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hk-guard-qa.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
src=$tmp/source
mkdir "$src"
git -C "$repo" archive "$base" CMakeLists.txt Makefile src include bench tests third_party | tar -x -C "$src"
patch -s -d "$src" -p1 -i "$here/certificate.patch"
python3 -B "$here/check.py" "$src" "$out" --base-repo "$repo"
python3 -B "$src/tests/check_hk15.py" "$src" > "$out/check-hk15.log"
python3 -B "$src/tests/check_hk15_ragged.py" > "$out/check-ragged.log"
cc=${CC:-cc}
opts=(-DCMAKE_BUILD_TYPE=Release -DCOB_ENABLE_HK15_RAGGED=ON -DCOB_ENABLE_PORTABLE_STRASSEN=ON
      -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=OFF)
if [[ $(uname -s) == Linux ]]; then opts+=(-DCMAKE_POSITION_INDEPENDENT_CODE=ON); fi
cmake -S "$src" -B "$tmp/build" "${opts[@]}"
cmake --build "$tmp/build" --parallel 2 --target cob_matmul
archive=$tmp/build/libcob_matmul.a
case $(uname -s) in
  Darwin) "$cc" -dynamiclib -Wl,-all_load "$archive" -o "$tmp/worker.dylib"
          lib=$tmp/worker.dylib ;;
  Linux)  "$cc" -shared -Wl,--whole-archive "$archive" -Wl,--no-whole-archive -lm -o "$tmp/worker.so"
          lib=$tmp/worker.so ;;
  *) echo "This stack check requires macOS or Linux." >&2; exit 2 ;;
esac
"$cc" -O2 "$here/small-stack.c" -ldl -pthread -o "$tmp/small-stack"
"$tmp/small-stack" "$lib" > "$out/small-stack.csv"
"$cc" -O3 -std=c11 -DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_RAGGED=1 \
  -DCOB_DISABLE_APPLE_AMX=1 -DCOB_DISABLE_APPLE_SME=1 -I"$src/include" -I"$src/src" \
  "$here/check-timing-inputs.c" "$src/src/cob_hk15_ragged.c" -lm -o "$tmp/timing-inputs"
"$tmp/timing-inputs" > "$out/timing-inputs.csv"
python3 -B "$here/timing/score.py" > "$out/timing-summary.json"
