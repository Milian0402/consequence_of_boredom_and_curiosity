#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
vendors=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
panel=$(realpath "$3")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/proxy"
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
case "$(awk -F: '/^vendor_id/{gsub(/ /,"",$2);print $2;exit}' /proc/cpuinfo)" in
    GenuineIntel) provider=mkl ;;
    AuthenticAMD) provider=aocl ;;
    *) echo "unsupported x86 vendor" >&2; exit 2 ;;
esac
test -r "$vendors/$provider.so"
cc=${CC:-cc}
"$cc" -O2 -std=c11 "$here/check.c" -ldl -o "$work/witness"
"$cc" -O2 -std=c11 -fPIC -shared -DHOST_PROVIDER="\"$provider\"" "$here/proxy.c" -ldl -o "$work/proxy/$provider.so"
export RANK_WITNESS_BLAS_DIR="$vendors" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1
echo "implementation,family,m,n,k,pad,provider,traced,padded,provider_m,provider_lda,provider_ldb,provider_ldc,checked,cpu,passed" > "$out/host-pad-witness.csv"
for mode in 0 1; do
    if [[ $mode == 0 ]]; then name=host-direct; else name=host-pad8; fi
    "$cc" -O3 -std=c11 -fPIC -shared -DHOST_PAD="$mode" "$here/adapter.c" -ldl -o "$out/$name.so"
    "$cc" -O3 -std=c11 -fPIC -shared -DHOST_PAD="$mode" -DHOST_PAD_TRACE "$here/adapter.c" -ldl -o "$work/$name.trace.so"
    RANK_HOST_BLAS_DIR="$vendors" taskset -c "$cpu" "$work/witness" "$out/$name.so" "$panel" "$mode" 0 "$provider" >> "$out/host-pad-witness.csv"
    RANK_HOST_BLAS_DIR="$work/proxy" taskset -c "$cpu" "$work/witness" "$work/$name.trace.so" "$panel" "$mode" 1 "$provider" >> "$out/host-pad-witness.csv"
done
awk -F, 'NR==1 || $8==1' "$out/host-pad-witness.csv" > "$out/host-pad-routes.csv"
{
    echo "provider=$provider"
    sha256sum "$vendors/$provider.so" "$out/host-direct.so" "$out/host-pad8.so" "$here/adapter.c" "$here/check.c" "$here/proxy.c" "$here/protocol.md" "$panel" | awk '{n=split($2,a,"/");print $1"  "a[n]}'
} > "$out/host-pad-provenance.txt"
