#!/bin/sh
set -eu

OUT_DIR=${COB_AUDIT_OUT_DIR:-"/tmp/cob-claim-audit-$(date -u +%Y%m%dT%H%M%SZ)"}
BENCH=${COB_AUDIT_BENCH:-./build/cob_gemm_bench}
REPEATS=${COB_AUDIT_REPEATS:-11}
BATCH=${COB_AUDIT_BATCH:-8}
COOLDOWN_SEC=${COB_AUDIT_COOLDOWN_SEC:-0}
SUITES=${COB_AUDIT_SUITES:-"square medium skinny"}

mkdir -p "$OUT_DIR"

require_bench() {
    if [ ! -x "$BENCH" ]; then
        echo "benchmark binary not found: $BENCH" >&2
        echo "run 'make all' first, or set COB_AUDIT_BENCH" >&2
        exit 1
    fi
}

write_context() {
    {
        echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "repo=$(pwd)"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "git_status_short_begin"
        git status --short
        echo "git_status_short_end"
        echo "uname=$(uname -a)"
        for key in \
            hw.model \
            hw.machine \
            hw.ncpu \
            hw.activecpu \
            hw.l1dcachesize \
            hw.l2cachesize \
            hw.cachelinesize \
            hw.pagesize \
            hw.optional.arm.FEAT_SME \
            hw.optional.arm.FEAT_SME2 \
            hw.optional.arm.FEAT_SME_F32F32; do
            value=$(sysctl -n "$key" 2>/dev/null || true)
            if [ "$value" ]; then
                echo "$key=$value"
            fi
        done
        echo "bench=$BENCH"
        echo "repeats=$REPEATS"
        echo "batch=$BATCH"
        echo "cooldown_sec=$COOLDOWN_SEC"
        echo "suites=$SUITES"
    } > "$OUT_DIR/context.txt"
}

cooldown() {
    if [ "$COOLDOWN_SEC" -gt 0 ]; then
        sleep "$COOLDOWN_SEC"
    fi
}

run_grid_env() {
    name=$1
    m_values=$2
    n_values=$3
    k_values=$4
    csv="$OUT_DIR/$name.csv"

    echo "running suite: $name" >&2
    COB_BENCH_ROUTE=1 \
        COB_BENCH_REPEATS="$REPEATS" \
        COB_GRID_BATCH="$BATCH" \
        COB_GRID_BENCH="$BENCH" \
        COB_GRID_M="$m_values" \
        COB_GRID_N="$n_values" \
        COB_GRID_K="$k_values" \
        sh tools/bench_grid.sh > "$csv"
    python3 tools/bench_gap_report.py --target "cob one-shot" "$csv" > "$OUT_DIR/$name.one-shot.gaps.csv"
    python3 tools/bench_gap_report.py --target "cob one-shot" --include-cob-baselines "$csv" \
        > "$OUT_DIR/$name.one-shot-vs-best.gaps.csv"
    python3 tools/bench_gap_report.py --target "cob packed-B" "$csv" > "$OUT_DIR/$name.packed-b.gaps.csv"
}

run_grid_shapes() {
    name=$1
    shift
    csv="$OUT_DIR/$name.csv"

    echo "running suite: $name" >&2
    COB_BENCH_ROUTE=1 \
        COB_BENCH_REPEATS="$REPEATS" \
        COB_GRID_BATCH="$BATCH" \
        COB_GRID_BENCH="$BENCH" \
        sh tools/bench_grid.sh "$@" > "$csv"
    python3 tools/bench_gap_report.py --target "cob one-shot" "$csv" > "$OUT_DIR/$name.one-shot.gaps.csv"
    python3 tools/bench_gap_report.py --target "cob one-shot" --include-cob-baselines "$csv" \
        > "$OUT_DIR/$name.one-shot-vs-best.gaps.csv"
    python3 tools/bench_gap_report.py --target "cob packed-B" "$csv" > "$OUT_DIR/$name.packed-b.gaps.csv"
}

write_summary() {
    summary="$OUT_DIR/summary.md"
    {
        echo "# COB Claim Audit"
        echo
        echo "- Commit: \`$(git rev-parse --short HEAD)\`"
        echo "- Repeats: \`$REPEATS\`"
        echo "- Cooldown seconds between suites: \`$COOLDOWN_SEC\`"
        echo "- Output directory: \`$OUT_DIR\`"
        echo
        for gaps in "$OUT_DIR"/*.gaps.csv; do
            [ -f "$gaps" ] || continue
            title=$(basename "$gaps" .gaps.csv)
            echo "## $title"
            echo
            python3 - "$gaps" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    print("No target gaps against non-COB baselines.")
else:
    print("| shape | target | route | baseline | ratio | gap |")
    print("| --- | --- | --- | --- | ---: | ---: |")
    for row in rows[:10]:
        print(
            f"| {row['shape']} | {row['target']} | {row['target_route']} | "
            f"{row['baseline']} | {row['ratio']} | {row['gap_percent']}% |"
        )
PY
            echo
        done
    } > "$summary"
}

require_bench
write_context

for suite in $SUITES; do
    cooldown
    case "$suite" in
        square)
            sizes=${COB_AUDIT_SQUARE_SIZES:-"64 128 192 256 384 512 768 1024 1280 1536 2048"}
            shapes=""
            for size in $sizes; do
                shapes="$shapes $size"
            done
            # shellcheck disable=SC2086
            run_grid_shapes square $shapes
            ;;
        medium)
            run_grid_env \
                medium \
                "${COB_AUDIT_MEDIUM_M:-"512 768 1024"}" \
                "${COB_AUDIT_MEDIUM_N:-"512 768 1024 1152 1216 1280"}" \
                "${COB_AUDIT_MEDIUM_K:-"1536 2048 3072 4096"}"
            ;;
        skinny)
            # Keep this explicit: the skinny claim area is sparse and route-specific.
            # Additional shapes can be appended through COB_AUDIT_SKINNY_EXTRA.
            # shellcheck disable=SC2086
            run_grid_shapes skinny \
                64x1024x512 64x2048x512 64x4096x512 64x8192x512 64x32768x512 \
                64x1024x1536 64x1408x1536 64x2048x2048 64x4096x2048 \
                64x2112x7168 64x4096x7168 64x7168x2048 64x24576x1536 \
                96x4096x512 96x8192x512 96x4096x1024 96x8192x1024 \
                128x4096x512 128x8192x512 128x4096x1024 128x8192x1024 \
                ${COB_AUDIT_SKINNY_EXTRA:-}
            ;;
        *)
            echo "unknown audit suite: $suite" >&2
            exit 1
            ;;
    esac
done

write_summary
echo "$OUT_DIR"
