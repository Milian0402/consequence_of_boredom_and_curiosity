#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=${1:-$here/checkout}
output=${2:?supply an output directory}
mkdir -p "$output"
cc=${CC:-cc}
arch=${CHECK_ARCH:-$(uname -m)}
flags="-std=c11 -O3 -DCOB_DISABLE_APPLE_AMX=1 -DCOB_DISABLE_APPLE_SME=1 -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_FUSED_STRASSEN=1 -DCOB_ENABLE_BLOCKED_LEAF=1 -DCOB_FUSED_STRASSEN_MIN_HARMONIC_DIM=16 -DCOB_FUSED_MC=40 -DCOB_FUSED_NC=48 ${EXTRA_FLAGS:-}"
case "$arch" in arm64|aarch64) flags="$flags -DCOB_USE_NEON=1";; esac
renames=""
for fn in sgemm_rowmajor sgemm_ref_rowmajor sgemm_pack_a sgemm_free_packed_a sgemm_pack_b sgemm_free_packed_b sgemm_rowmajor_packed_b sgemm_rowmajor_packed_ab; do renames="$renames -Dcob_$fn=prior_$fn"; done
for mode in prior final; do
    prefix=
    if [ "$mode" = prior ]; then prefix="$renames -DCOB_ENABLE_FUSED_FINAL_PANEL=0"; else prefix="-DCOB_ENABLE_FUSED_FINAL_PANEL=1"; fi
    cat > "$output/$mode.c" <<SRC
#include "gemm.c"
int ${mode}_panel_run(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc){
    return cob_sgemm_strassen_fused(m,n,k,a,lda,b,ldb,c,ldc);
}
SRC
    "$cc" $flags $prefix -I"$repo/include" -I"$repo/src" -c "$output/$mode.c" -o "$output/$mode.o"
done
"$cc" -O2 -std=c11 ${EXTRA_FLAGS:-} "$here/check.c" "$output/prior.o" "$output/final.o" -lm -o "$output/check"
"$output/check" > "$output/accuracy.csv"
