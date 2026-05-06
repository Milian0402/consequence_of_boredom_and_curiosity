#!/bin/sh
set -eu

BENCH=${COB_COUNTER_BENCH:-./build/cob_gemm_bench}
MPERF=${COB_COUNTER_MPERF:-/private/tmp/mperf/mperf-stat}
KPEP_DB=${COB_COUNTER_KPEP_DB:-as5}
ONLY=${COB_COUNTER_ONLY:-one-shot}
REPEATS=${COB_COUNTER_REPEATS:-3}
EVENTS=${COB_COUNTER_EVENTS:-"cycles instructions l1d-cache-misses l1d-tlb-misses l2-tlb-misses-data map-stalls dispatch-stalls CORE_WAITING_SME_ENGINE_CYCLE LDST_UNIT_WAITING_SME_ENGINE_MEM_DATA INST_SME_ENGINE_ALU"}

if [ ! -x "$BENCH" ]; then
    echo "benchmark binary not found: $BENCH" >&2
    echo "run 'make all' first, or set COB_COUNTER_BENCH" >&2
    exit 1
fi

if [ ! -x "$MPERF" ]; then
    echo "mperf-stat not found: $MPERF" >&2
    echo "build https://github.com/tmcgilchrist/mperf, or set COB_COUNTER_MPERF" >&2
    exit 1
fi

if ! MPERF_KPEP_DB="$KPEP_DB" "$MPERF" -l >/dev/null 2>&1; then
    echo "mperf could not load the '$KPEP_DB' PMC database" >&2
    echo "on this M5 Max, /usr/share/kpep/as5.plist is the expected database" >&2
    echo "set COB_COUNTER_KPEP_DB to override" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- \
        64x2112x7168 \
        64x4096x7168 \
        64x7168x2048 \
        64x8192x1024 \
        512x1216x2048 \
        512x1280x1536 \
        1024x1024x1536
fi

event_args=""
for event in $EVENTS; do
    event_args="$event_args -e $event"
done

for shape in "$@"; do
    echo "== $shape ($ONLY) ==" >&2
    # shellcheck disable=SC2086
    MPERF_KPEP_DB="$KPEP_DB" "$MPERF" $event_args -- \
        env \
        COB_BENCH_ONLY="$ONLY" \
        COB_BENCH_PACK_SETUP=1 \
        COB_BENCH_REPEATS="$REPEATS" \
        COB_BENCH_ROUTE=1 \
        "$BENCH" "$shape"
done
