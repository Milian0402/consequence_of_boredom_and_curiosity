#!/bin/sh
set -eu

MPGEMM_DIR=${COB_MPGEMM_DIR:-/private/tmp/mpgemm_latest}
OUT_DIR=${COB_MPGEMM_OUT_DIR:-/private/tmp/cob-mpgemm-calibration-$(date +%Y%m%d-%H%M%S)}
COB_BENCH=${COB_MPGEMM_COB_BENCH:-./build/cob_gemm_bench}
COB_REPEATS=${COB_MPGEMM_COB_REPEATS:-31}
RUN_MPGEMM=${COB_MPGEMM_RUN_MPGEMM:-1}
SHAPES=${COB_MPGEMM_SHAPES:-"64x2112x7168 64x24576x1536 64x32768x512 64x7168x16384 64x4096x7168 64x7168x2048"}

if [ ! -d "$MPGEMM_DIR" ]; then
    echo "MpGEMM checkout not found: $MPGEMM_DIR" >&2
    echo "clone https://github.com/MpGEMM/mpgemm or set COB_MPGEMM_DIR" >&2
    exit 1
fi

if [ ! -x "$COB_BENCH" ]; then
    echo "COB benchmark not found: $COB_BENCH" >&2
    echo "run 'make all' first, or set COB_MPGEMM_COB_BENCH" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

{
    echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mpgemm_dir: $MPGEMM_DIR"
    git -C "$MPGEMM_DIR" rev-parse --short HEAD 2>/dev/null | sed 's/^/mpgemm_commit: /' || true
    echo "cob_bench: $COB_BENCH"
    git rev-parse --short HEAD 2>/dev/null | sed 's/^/cob_commit: /' || true
    echo "cob_repeats: $COB_REPEATS"
    echo "cob_bench_iters: ${COB_BENCH_ITERS:-auto}"
    echo "shapes: $SHAPES"
    license_files=$(find "$MPGEMM_DIR" -maxdepth 2 \( -iname 'license*' -o -iname 'copying*' \) -print)
    if [ "$license_files" ]; then
        echo "mpgemm_license_files:"
        printf '%s\n' "$license_files"
    else
        echo "mpgemm_license_files: none found within depth 2"
    fi
} > "$OUT_DIR/context.txt"

if [ "$RUN_MPGEMM" != "0" ]; then
    make -C "$MPGEMM_DIR/src"
    make -C "$MPGEMM_DIR/benchmark" correct.x singlePerformance.x
    "$MPGEMM_DIR/benchmark/correct.x" > "$OUT_DIR/mpgemm-correct.txt"
    "$MPGEMM_DIR/benchmark/singlePerformance.x" > "$OUT_DIR/mpgemm-singlePerformance.txt"
fi

COB_BENCH_ONLY=one-shot \
COB_BENCH_ROUTE=1 \
COB_BENCH_REPEATS="$COB_REPEATS" \
COB_GRID_BENCH="$COB_BENCH" \
    sh tools/bench_grid.sh $SHAPES > "$OUT_DIR/cob-one-shot.csv"

python3 tools/bench_sanity_report.py "$OUT_DIR/cob-one-shot.csv" \
    > "$OUT_DIR/cob-one-shot.sanity.csv"

if [ -f "$OUT_DIR/mpgemm-singlePerformance.txt" ]; then
    python3 tools/mpgemm_gap_report.py \
        "$OUT_DIR/cob-one-shot.csv" \
        "$OUT_DIR/mpgemm-singlePerformance.txt" \
        > "$OUT_DIR/mpgemm-row-sgemm.gaps.csv"
    python3 tools/mpgemm_gap_report.py --all \
        "$OUT_DIR/cob-one-shot.csv" \
        "$OUT_DIR/mpgemm-singlePerformance.txt" \
        > "$OUT_DIR/mpgemm-row-sgemm.all.csv"
fi

echo "$OUT_DIR"
