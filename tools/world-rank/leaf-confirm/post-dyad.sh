#!/usr/bin/env bash
set -euo pipefail
src=$1; vendor=$2; work=$3; out=$4; cpu=$5
here=$(cd "$(dirname "$0")" && pwd)
receipts=$out/dyadic-stress
mkdir -p "$receipts"
printf 'method,status,executable_sha256\n' > "$receipts/status.csv"
arch=$(uname -m)
native=mkl
if [[ $arch == aarch64 ]]; then native=armpl; fi
prefix=$(cat "$vendor/openblas-prefix.txt")
"${CC:-gcc}" -std=gnu11 -O2 -DWITH_OPENBLAS -I"$src/include" -I"$prefix/include" "$here/dyad.c" \
    "$prefix/lib/libopenblas.a" -lpthread -lm -ldl -o "$work/direct-openblas.dyad"
methods=(direct-openblas released-hk15 ragged-hk15 release-openblas-two release-openblas-aligned-two
    "release-$native-two" "release-$native-aligned-two" hk15-openblas-leaf "hk15-$native-leaf")
if [[ -x $work/guarded-hk15.dyad ]]; then methods+=(guarded-hk15); fi
for name in "${methods[@]}"; do
    exe=$work/$name.dyad
    if [[ ! -x $exe ]]; then
        printf '%s,not_built,\n' "$name" >> "$receipts/status.csv"
        continue
    fi
    digest=$(sha256sum "$exe" | cut -d' ' -f1)
    rc=0
    taskset -c "$cpu" "$exe" 1931 2903 2881 "$name" > "$receipts/$name.csv" 2> "$receipts/$name.log" || rc=$?
    if ((rc==0)) && [[ $(wc -l < "$receipts/$name.csv") == 20 ]]; then result=complete; else result=failed-$rc; fi
    printf '%s,%s,%s\n' "$name" "$result" "$digest" >> "$receipts/status.csv"
done
