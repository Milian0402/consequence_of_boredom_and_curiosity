# Sourced by tile-study.sh on a native runner after the incumbent is built.
replay_provenance() {
    local dest=$1
    mkdir -p "$dest" || return 1
    cp "$out/host.txt" "$out/build.txt" "$out/tile-build.txt" "$dest/" || return 1
    local manifest
    for manifest in "$vendor"/*.txt; do
        if [[ -f $manifest ]]; then cp "$manifest" "$dest/" || return 1; fi
    done
}

replay_measure() {
    local dest=$1
    shift
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$dest/panel.csv" "$dest" \
        "$vendor/openblas.so" --phase tile-replay --rounds 3 --stress "$@"
}

replay_hk_entry() {
    local dest=$1
    replay_provenance "$dest" || return 1
    cp "$dir/entry-replay.csv" "$dest/panel.csv" || return 1
    python3 "$dir/finite/check_entry.py" "$repo" > "$dest/hk-public-routes.txt" || return 1
    # Catch a replay accidentally built from a revision preceding integration.
    grep -q 'COB_ENABLE_HK15' "$repo/src/gemm.c" || { echo 'Missing public HK15 entry' >&2; return 1; }
    compile_entry "$repo" "$build/public-hk.so" -DCOB_ENABLE_HK15=1 || return 1
    local schedule raw_dir
    for schedule in immediate stored; do
        raw_dir="$build/hk-$schedule"
        local schedule_flags=()
        if [[ $schedule == stored ]]; then schedule_flags=(stored); fi
        bash "$dir/finite/build.sh" "$raw_dir" "$build/incumbent.so" "${schedule_flags[@]}" || return 1
        cp "$raw_dir/hk233-certificate.txt" "$dest/$schedule-certificate.txt" || return 1
        cp "$raw_dir/hk233-smoke.txt" "$dest/$schedule-smoke.txt" || return 1
    done
    {
        printf 'public_entry_flag=-DCOB_ENABLE_HK15=1\n'
        sha256sum "$build/public-hk.so" "$build/hk-immediate/hk233.so" "$build/hk-stored/hk233.so" \
            "$build/incumbent.so" "$build/worker" "$repo/src/gemm.c" "$repo/src/cob_hk15.inc" \
            "$dir/study.py" "$dir/entry-replay.inc" "$dest/panel.csv"
    } > "$dest/replay-build.txt" || return 1
    replay_measure "$dest" "${replay_peers[@]}" \
        --candidate "cob-incumbent=direct:$build/incumbent.so" \
        --candidate "cob-hk15-public=direct:$build/public-hk.so" \
        --candidate "hk233=hk233:$build/hk-immediate/hk233.so" \
        --candidate "hk233-stored=hk233:$build/hk-stored/hk233.so"
}

replay_stock_failure_only() {
    # The complete attempt may have named worker errors plus a summary row for
    # missing pairs. Never treat an oracle, second worker or timing failure as
    # a stock-only failure, even if some valid worker observations were written.
    python3 - "$1" <<'PY'
import csv
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
rank = json.loads((dest / "ranking.json").read_text())
errors = rank["errors"]
if not any(e.get("implementation") == "blasfeo" for e in errors):
    raise SystemExit("No recorded stock BLASFEO worker error")
for e in errors:
    if e.get("implementation") == "blasfeo":
        continue
    if "implementation" not in e and e.get("error") == "incomplete paired timing rounds":
        continue
    raise SystemExit(f"Failure extends beyond stock BLASFEO: {e}")

expected = {(b, r) for b in (1, 2) for r in range(1, rank["rounds_per_seed"] + 1)}
other = set(rank["implementations"]) - {"blasfeo"}
observed = {name: [] for name in other}
with (dest / "timings.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if row["implementation"] in other:
            observed[row["implementation"]].append((int(row["block"]), int(row["round"])))
if any(len(items) != len(expected) or set(items) != expected for items in observed.values()):
    raise SystemExit("Missing or repeated paired observations for another implementation")
print("Stock BLASFEO failed its replay. All other implementations have complete paired timings.")
PY
}

replay_stock_row() {
    local dest=$1 fallback=$2
    replay_provenance "$dest" || return 1
    [[ -s $vendor/blasfeo-row48.so && -s $vendor/blasfeo.so ]] || {
        echo 'Stock and row-slab BLASFEO adapters are both required' >&2; return 1;
    }
    printf 'family,m,n,k,pad\nwide-stock-replay,255,2048,768,4\n' > "$dest/panel.csv" || return 1
    compile_entry "$repo" "$build/replay-exact-edge.so" -DCOB_ENABLE_X86_EXACT_EDGE=1 || return 1
    sha256sum "$vendor/blasfeo.so" "$vendor/blasfeo-row48.so" "$build/incumbent.so" \
        "$build/replay-exact-edge.so" "$build/worker" "$dest/panel.csv" "$dir/study.py" \
        "$dir/entry-replay.inc" > "$dest/replay-build.txt" || return 1
    local variants=(--vendor "blasfeo-row48=$vendor/blasfeo-row48.so"
        --candidate "cob-incumbent=direct:$build/incumbent.so"
        --candidate "exact-edge=direct:$build/replay-exact-edge.so")
    if replay_measure "$dest" "${replay_peers[@]}" "${variants[@]}"; then return 0; fi

    replay_stock_failure_only "$dest" > "$dest/stock-status.txt" || return 1
    cat "$dest/stock-status.txt"
    replay_provenance "$fallback" || return 1
    cp "$dest/panel.csv" "$dest/stock-status.txt" "$dest/replay-build.txt" "$fallback/" || return 1
    replay_measure "$fallback" "${replay_peers_without_stock[@]}" "${variants[@]}"
}

run_entry_replay() {
    local root
    root=$(dirname "$out") || return 1
    local replay_peers=() replay_peers_without_stock=()
    local name
    for name in "${names[@]}"; do
        case $name in openblas|libxsmm-*) continue ;; esac
        replay_peers+=(--vendor "$name=$vendor/$name.so")
        if [[ $name != blasfeo ]]; then replay_peers_without_stock+=(--vendor "$name=$vendor/$name.so"); fi
    done
    local hk_status=0 stock_status=0
    replay_hk_entry "$root/entry-replay" || hk_status=$?
    if [[ $arch == x86_64 ]]; then
        replay_stock_row "$root/stock-replay" "$root/stock-replay-without-stock" || stock_status=$?
    fi
    if (( hk_status || stock_status )); then
        printf 'Replay failed (HK15=%d, stock-row=%d). See each panel artifact.\n' "$hk_status" "$stock_status" >&2
        return 1
    fi
}
