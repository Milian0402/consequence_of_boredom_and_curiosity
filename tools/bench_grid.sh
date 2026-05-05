#!/bin/sh
set -eu

BENCH=${COB_GRID_BENCH:-./build/cob_gemm_bench}
BATCH_SIZE=${COB_GRID_BATCH:-16}

if [ ! -x "$BENCH" ]; then
    echo "benchmark binary not found: $BENCH" >&2
    echo "run 'make bench' first, or set COB_GRID_BENCH" >&2
    exit 1
fi

if [ "$BATCH_SIZE" -lt 1 ]; then
    echo "COB_GRID_BATCH must be positive" >&2
    exit 1
fi

header_printed=0
batch=""
batch_count=0

run_batch() {
    if [ "$batch_count" -eq 0 ]; then
        return
    fi

    if [ "$header_printed" -eq 0 ]; then
        COB_BENCH_CSV=1 "$BENCH" $batch
        header_printed=1
    else
        COB_BENCH_CSV=1 "$BENCH" $batch | sed '1d'
    fi

    batch=""
    batch_count=0
}

add_shape() {
    batch="$batch $1"
    batch_count=$((batch_count + 1))
    if [ "$batch_count" -ge "$BATCH_SIZE" ]; then
        run_batch
    fi
}

if [ "$#" -gt 0 ]; then
    for shape in "$@"; do
        add_shape "$shape"
    done
    run_batch
    exit 0
fi

if [ "${COB_GRID_M:-}" ] && [ "${COB_GRID_N:-}" ] && [ "${COB_GRID_K:-}" ]; then
    for m in $COB_GRID_M; do
        for n in $COB_GRID_N; do
            for k in $COB_GRID_K; do
                add_shape "${m}x${n}x${k}"
            done
        done
    done
else
    for size in ${COB_GRID_SIZES:-64 128 192 256 384 512 768 1024 1280 1536 2048}; do
        add_shape "$size"
    done
fi

run_batch
