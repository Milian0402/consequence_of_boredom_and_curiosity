#!/usr/bin/env bash
# Reuse the exact serial OpenBLAS archive and FP64 oracle from the ranking job.
set -euo pipefail
[[ $# == 1 && $(uname -s) == Linux ]]
vendor=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
runner=${RUNNER_TEMP:-/tmp}
openblas_sha=e0166008be8e466242aa76b2ff75ce3f0fbf574a
grep -F "Upstream OpenBLAS v0.3.34 $openblas_sha" "$vendor/openblas-build.txt"
grep -F 'SINGLE_THREADED' "$vendor/openblas-build.txt"
test -s "$vendor/openblas.so"
read -r OPENBLAS_PREFIX < "$vendor/openblas-prefix.txt"
test -s "$OPENBLAS_PREFIX/lib/libopenblas.a"
[[ $(git -C "$OPENBLAS_PREFIX/../source" rev-parse HEAD) == "$openblas_sha" ]]
export OPENBLAS_PREFIX FASTMMW_ORACLE="$vendor/openblas.so"
tmp=$(mktemp -d "$runner/fastmmw-rank.XXXXXX")
revision=96949ce923a9ac10a13f98f76e88c76b1e756f52
archive="${FASTMMW_ARCHIVE:-$tmp/FastMMW.tar.gz}"
if [[ -z ${FASTMMW_ARCHIVE:-} ]]; then
    curl --proto '=https' --fail --silent --show-error --location --retry 3 --connect-timeout 30 --max-time 180 \
        "https://codeload.github.com/paolodalberto/FastMMW/tar.gz/$revision" -o "$archive"
fi
for cutoff in 1536 768; do
    build="$tmp/build-$cutoff"
    FASTMMW_LEAF="$cutoff" bash "$here/build.sh" "$build" "$archive"
    grep -m1 -F 'FastMMW s_wm serial, OpenBLAS OpenBLAS 0.3.34' "$build/smoke.log"
    cp "$build/fastmmw-padded.so" "$vendor/fastmmw-$cutoff.so"
    cp "$build/fastmmw-padded-trace.so" "$vendor/fastmmw-$cutoff.trace.so"
    {
        cat "$build/build-info.txt" "$vendor/openblas-build.txt" "$build/smoke.log"
        printf 'Public adapter rounds to the recursion-depth multiple; all staging happens inside each SGEMM call.\n'
    } > "$vendor/fastmmw-$cutoff-build.txt"
done
