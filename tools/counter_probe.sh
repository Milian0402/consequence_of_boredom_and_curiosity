#!/bin/sh
set -eu

BENCH=${COB_COUNTER_BENCH:-./build/cob_gemm_bench}
MPERF=${COB_COUNTER_MPERF:-/private/tmp/mperf/mperf-stat}
KPEP_DB=${COB_COUNTER_KPEP_DB:-as5}
ONLY=${COB_COUNTER_ONLY:-one-shot}
REPEATS=${COB_COUNTER_REPEATS:-3}
PROFILE=${COB_COUNTER_PROFILE:-pipeline}
if [ "${COB_COUNTER_EVENTS+x}" ]; then
    EVENTS=$COB_COUNTER_EVENTS
else
    case "$PROFILE" in
        pipeline)
            EVENTS="cycles instructions l1d-cache-misses l1d-tlb-misses l2-tlb-misses-data map-stalls dispatch-stalls CORE_WAITING_SME_ENGINE_CYCLE LDST_UNIT_WAITING_SME_ENGINE_MEM_DATA"
            ;;
        sme)
            EVENTS="cycles instructions INST_SME_ENGINE_LD INST_SME_ENGINE_ST INST_SME_ENGINE_ALU INST_SME_ENGINE_PACKING_FUSED LD_SME_NORMAL_UOP ST_SME_NORMAL_UOP LDST_SME_XPG_UOP LDST_X64_UOP"
            ;;
        memory)
            EVENTS="cycles instructions L1D_CACHE_MISS_LD_NONSPEC L1D_TLB_MISS_NONSPEC LD_SRC_LL_CACHE_NONSPEC LD_SRC_MEMSYS_NONSPEC PL2_CACHE_ACCESS PL2_CACHE_MISS PL2_CACHE_MISS_LD L2_TLB_MISS_DATA"
            ;;
        *)
            echo "unknown COB_COUNTER_PROFILE: $PROFILE" >&2
            echo "expected one of: pipeline, sme, memory" >&2
            exit 1
            ;;
    esac
fi

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
    echo "== $shape ($ONLY, $PROFILE counters) ==" >&2
    # shellcheck disable=SC2086
    MPERF_KPEP_DB="$KPEP_DB" "$MPERF" $event_args -- \
        env \
        COB_BENCH_ONLY="$ONLY" \
        COB_BENCH_PACK_SETUP=1 \
        COB_BENCH_REPEATS="$REPEATS" \
        COB_BENCH_ROUTE=1 \
        "$BENCH" "$shape"
done
