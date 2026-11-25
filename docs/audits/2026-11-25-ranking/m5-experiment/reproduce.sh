#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=${1:-.}
base=b218beff1eaf8fd606d0934405cdc1e28ecdfe15
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo 'This reproduction uses Apple Silicon SME2.' >&2
    exit 2
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/base" "$tmp/candidate"
git -C "$repo" archive "$base" CMakeLists.txt src include tests third_party | tar -x -C "$tmp/base"
cp -R "$tmp/base/." "$tmp/candidate/"
patch -s --batch -d "$tmp/candidate" -p1 -i "$here/candidate.patch"
for rev in base candidate; do
    cmake -S "$tmp/$rev" -B "$tmp/$rev-build" -DCMAKE_BUILD_TYPE=Release \
        -DCOB_STRASSEN_SME=ON -DCOB_ENABLE_PORTABLE_STRASSEN=ON \
        -DCOB_BUILD_TESTS=OFF -DCOB_BUILD_BENCH=OFF
    cmake --build "$tmp/$rev-build" --parallel 2
    cc -O2 -std=c11 -I"$tmp/$rev/include" "$here/fixture.c" \
        "$tmp/$rev-build/libcob_matmul.a" -o "$tmp/$rev-fixture"
    printf '%s ' "$rev"
    result=0
    "$tmp/$rev-fixture" "$tmp/$rev.bin" || result=$?
    if [[ $result != 1 ]]; then
        echo "Fixture differed from the recorded threshold failure (exit $result)" >&2
        exit 2
    fi
done
cmp "$tmp/base.bin" "$tmp/candidate.bin"
echo 'All 1,600 baseline and candidate FP32 values are bit-identical. Accuracy gate failed.'
exit 1
