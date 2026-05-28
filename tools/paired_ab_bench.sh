#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: sh tools/paired_ab_bench.sh BASELINE_GEMM_C CANDIDATE_GEMM_C [SHAPE...]" >&2
    echo "       COB_AB_MODE=packed compares the packed-B API; packed-AB compares the fully prepacked API." >&2
    echo "       COB_AB_MAX_REPEATS and COB_AB_CV_TARGET enable automatic repeat extension." >&2
    echo "       COB_AB_A_FLAGS and COB_AB_B_FLAGS append side-specific compiler flags." >&2
    echo "       COB_AB_ITERS forces multiple GEMM calls per timed sample for noisy shapes." >&2
    echo "       COB_AB_COOLDOWN_US sleeps after each paired sample for thermally noisy shapes." >&2
    echo "       COB_AB_HOLDOUT=0 disables split-half holdout reporting." >&2
    exit 2
fi

BASELINE_SRC=$1
CANDIDATE_SRC=$2
shift 2

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR=${COB_AB_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cob-ab.XXXXXX")}
CC_BIN=${CC:-cc}
OPT_FLAGS=${COB_AB_OPT_FLAGS:--O3}

COMMON_CFLAGS="$OPT_FLAGS -std=c11 -Wall -Wextra -Wpedantic -I$ROOT_DIR/include"
A_FLAGS=${COB_AB_A_FLAGS:-}
B_FLAGS=${COB_AB_B_FLAGS:-}
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

if [ "$UNAME_S" = "Darwin" ]; then
    COMMON_CFLAGS="$COMMON_CFLAGS -D_DARWIN_C_SOURCE"
fi

if [ "$UNAME_M" = "arm64" ] || [ "$UNAME_M" = "aarch64" ]; then
    COMMON_CFLAGS="$COMMON_CFLAGS -DCOB_USE_NEON=1"
fi

RENAME_A="-Dcob_sgemm_ref_rowmajor=cob_a_sgemm_ref_rowmajor -Dcob_sgemm_pack_b=cob_a_sgemm_pack_b -Dcob_sgemm_free_packed_b=cob_a_sgemm_free_packed_b -Dcob_sgemm_rowmajor_packed_b=cob_a_sgemm_rowmajor_packed_b -Dcob_sgemm_pack_a=cob_a_sgemm_pack_a -Dcob_sgemm_free_packed_a=cob_a_sgemm_free_packed_a -Dcob_sgemm_rowmajor_packed_ab=cob_a_sgemm_rowmajor_packed_ab -Dcob_sgemm_rowmajor=cob_a_sgemm_rowmajor"
RENAME_B="-Dcob_sgemm_ref_rowmajor=cob_b_sgemm_ref_rowmajor -Dcob_sgemm_pack_b=cob_b_sgemm_pack_b -Dcob_sgemm_free_packed_b=cob_b_sgemm_free_packed_b -Dcob_sgemm_rowmajor_packed_b=cob_b_sgemm_rowmajor_packed_b -Dcob_sgemm_pack_a=cob_b_sgemm_pack_a -Dcob_sgemm_free_packed_a=cob_b_sgemm_free_packed_a -Dcob_sgemm_rowmajor_packed_ab=cob_b_sgemm_rowmajor_packed_ab -Dcob_sgemm_rowmajor=cob_b_sgemm_rowmajor"

echo "paired A/B build dir: $BUILD_DIR" >&2
"$CC_BIN" $COMMON_CFLAGS $A_FLAGS $RENAME_A -c "$BASELINE_SRC" -o "$BUILD_DIR/gemm_a.o"
"$CC_BIN" $COMMON_CFLAGS $B_FLAGS $RENAME_B -c "$CANDIDATE_SRC" -o "$BUILD_DIR/gemm_b.o"
"$CC_BIN" $COMMON_CFLAGS "$ROOT_DIR/bench/bench_ab.c" "$BUILD_DIR/gemm_a.o" "$BUILD_DIR/gemm_b.o" -lm -o "$BUILD_DIR/cob_ab_bench"

exec "$BUILD_DIR/cob_ab_bench" "$@"
