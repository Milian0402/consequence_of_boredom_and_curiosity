#!/usr/bin/env bash
set -euo pipefail
test_dir=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$test_dir/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
case $(uname -s) in
    Darwin) linkage=(-dynamiclib) ;;
    Linux) linkage=(-shared) ;;
    *) echo "Unsupported platform for the real ABI tests" >&2; exit 2 ;;
esac
common=(-O2 -std=c11 -fPIC -DAMX_FUSED6_BENCH_ABI=1)
for variant in ordinary trace; do
    diagnostic=()
    if [[ $variant == trace ]]; then diagnostic=(-DCOB_AMX_TRACE=1); fi
    "${CC:-cc}" "${common[@]}" "${diagnostic[@]}" "${linkage[@]}" "$src/amx_fused6.c" "$src/amx_compute.c" \
        "$test_dir/control.c" -lm -o "$tmp/$variant.so"
done
python3 -B "$test_dir/check_fused.py" "$tmp/ordinary.so" --trace "$tmp/trace.so" "$@"
