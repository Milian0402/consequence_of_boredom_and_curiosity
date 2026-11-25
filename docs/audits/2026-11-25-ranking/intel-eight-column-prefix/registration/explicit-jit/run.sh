#!/usr/bin/env bash
# Run this evidence-only API probe outside all benchmark clocks.
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
panel=$(realpath "$1"); vendor=$(realpath "$2"); out=$(mkdir -p "$3" && cd "$3" && pwd)
tmp=$(mktemp -d)
readarray -t sdk < "$vendor/mkl-prefix.txt"
test "${#sdk[@]}" = 2
inc=${sdk[0]}; lib=${sdk[1]}
gcc -std=c11 -O2 -Wall -Wextra -Werror -isystem "$inc" "$dir/probe.c" -o "$tmp/probe" \
  -L"$lib" "-Wl,--disable-new-dtags,-rpath,$lib" -Wl,--no-as-needed -l:libmkl_intel_lp64.so.3 \
  -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 -Wl,--as-needed -lpthread -lm -ldl
ldd "$tmp/probe" > "$out/explicit-jit-dependencies.txt"
sha256sum "$dir/probe.c" "$tmp/probe" "$inc/mkl_cblas.h" "$lib/libmkl_intel_lp64.so.3" > "$out/explicit-jit.sha256"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
python3 -B "$dir/record.py" "$tmp/probe" "$panel" "$out/explicit-jit.csv"
