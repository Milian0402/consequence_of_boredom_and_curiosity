#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: sh tools/paired_accelerate_bench.sh COB_GEMM_C [SHAPE...]" >&2
    echo "       Compares COB as A against Apple Accelerate as B in the paired A/B harness." >&2
    echo "       Direct one-shot mode only; COB_AB_MODE=packed is not supported." >&2
    exit 2
fi

COB_SRC=$1
shift

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR=${COB_AB_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cob-accel-ab.XXXXXX")}
CC_BIN=${CC:-cc}
OPT_FLAGS=${COB_AB_OPT_FLAGS:--O3}
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

if [ "$UNAME_S" != "Darwin" ]; then
    echo "paired Accelerate benchmark requires macOS" >&2
    exit 2
fi
if [ "${COB_AB_MODE:-}" != "" ] && [ "${COB_AB_MODE:-}" != "direct" ]; then
    echo "paired Accelerate benchmark supports only direct one-shot mode" >&2
    exit 2
fi

COMMON_CFLAGS="$OPT_FLAGS -std=c11 -Wall -Wextra -Wpedantic -I$ROOT_DIR/include -D_DARWIN_C_SOURCE -DACCELERATE_NEW_LAPACK=1"
if [ "$UNAME_M" = "arm64" ] || [ "$UNAME_M" = "aarch64" ]; then
    COMMON_CFLAGS="$COMMON_CFLAGS -DCOB_USE_NEON=1"
fi

RENAME_A="-Dcob_sgemm_ref_rowmajor=cob_a_sgemm_ref_rowmajor -Dcob_sgemm_pack_b=cob_a_sgemm_pack_b -Dcob_sgemm_free_packed_b=cob_a_sgemm_free_packed_b -Dcob_sgemm_rowmajor_packed_b=cob_a_sgemm_rowmajor_packed_b -Dcob_sgemm_pack_a=cob_a_sgemm_pack_a -Dcob_sgemm_free_packed_a=cob_a_sgemm_free_packed_a -Dcob_sgemm_rowmajor_packed_ab=cob_a_sgemm_rowmajor_packed_ab -Dcob_sgemm_rowmajor=cob_a_sgemm_rowmajor"

echo "paired Accelerate A/B build dir: $BUILD_DIR" >&2
"$CC_BIN" $COMMON_CFLAGS $RENAME_A -c "$COB_SRC" -o "$BUILD_DIR/gemm_a.o"
"$CC_BIN" $COMMON_CFLAGS -DCOB_AB_B_ACCELERATE=1 "$ROOT_DIR/bench/bench_ab.c" "$BUILD_DIR/gemm_a.o" -framework Accelerate -lm -o "$BUILD_DIR/cob_accel_ab_bench"

export VECLIB_MAXIMUM_THREADS=1
export ACCELERATE_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

exec "$BUILD_DIR/cob_accel_ab_bench" "$@"
