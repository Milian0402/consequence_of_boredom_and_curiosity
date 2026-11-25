#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]]
here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
status="$out/tblis-status.txt"
flags=$(awk -F: '/^flags[[:space:]]*:/ {print $2; exit}' /proc/cpuinfo 2>/dev/null || true)
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]] ||
    ! grep -Eq '^vendor_id[[:space:]]*:[[:space:]]*GenuineIntel$' /proc/cpuinfo ||
    [[ " $flags " != *' avx2 '* || " $flags " != *' fma '* ]]; then
    echo 'skipped: this comparator requires native Intel Linux x86-64 with AVX2 and FMA' | tee "$status"
    exit 0
fi
revision=6e929ab34c366c4ec6804ad2bf7cae4b84ee81ab
echo 'building: native Intel Haswell FP32 AB' > "$status"
archive="$out/tblis-source.tar.gz"
curl --proto '=https' --fail --silent --show-error --location --retry 3 --connect-timeout 15 --max-time 150 \
    "https://codeload.github.com/flame/tblis-strassen/tar.gz/$revision" -o "$archive"
timeout 780 bash "$here/compile.sh" "$out" "$archive"
