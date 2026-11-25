# Sourced from tile-study.sh after native flags, vendor paths and worker are ready.
run_hk_child() {
    local dest="${out%/*}/hk-child"
    local baseline=9df006cc27398802d9dc24b73b5208800be5c155
    local snapshot="$build/hk-public-baseline" plugins="$build/hk-child-plugins"
    mkdir -p "$snapshot" "$plugins" "$dest"
    # The public timing uses the unmodified 9df006c source, including its own
    # certified schedule. Only its diagnostic companion uses the new counters.
    git -C "$repo" archive "$baseline" src include tools/world-rank/cob_adapter.c \
        tools/world-rank/vendor.h tools/world-rank/finite/hk233.inc | tar -x -C "$snapshot"
    cp "$dir/hk-child.csv" "$dest/panel.csv"
    cp "$out/host.txt" "$out/build.txt" "$out/tile-build.txt" "$dest/"
    local info
    for info in "$vendor"/*.txt; do if [[ -f $info ]]; then cp "$info" "$dest/"; fi; done
    {
        printf 'baseline_revision=%s\ncandidate_revision=%s\n' "$baseline" "$(git -C "$repo" rev-parse HEAD)"
        printf 'common_flags=%s\ntarget_flags=%s\n' "${common[*]}" "${target[*]}"
        sha256sum "$build/worker" "$dir/study.py" "$dir/hybrid_next.c" "$dir/hk-child-study.inc" \
            "$dir/hk-child-routes.py" "$snapshot/src/gemm.c" "$repo/src/gemm.c" \
            "$repo/src/cob_fused_strassen.inc" "$repo/src/cob_hk15.inc" "$dest/panel.csv"
    } > "$dest/child-build.txt"
    local candidates=() route_args=() threshold label cutoff_flags=()
    for threshold in 1024 960 896; do
        local base="$repo"
        label="cob-hk15-$threshold"
        cutoff_flags=(-DCOB_HK15_CHILD_HARMONIC="$threshold")
        if [[ $threshold == 1024 ]]; then
            base="$snapshot"
            label=cob-hk15-public
            cutoff_flags=()
        fi
        compile_entry "$base" "$plugins/$label.so" -DCOB_ENABLE_HK15=1 "${cutoff_flags[@]}"
        compile_entry "$repo" "$plugins/$label.trace.so" -DCOB_ENABLE_HK15=1 \
            -DCOB_HK15_CHILD_HARMONIC="$threshold" -DCOB_HK15_CHILD_TRACE=1
        candidates+=(--candidate "$label=direct:$plugins/$label.so")
        route_args+=(--method "$label" "$threshold")
        {
            printf 'method=%s timed_flags=-DCOB_ENABLE_HK15=1 %s trace_cutoff=%s\n' \
                "$label" "${cutoff_flags[*]}" "$threshold"
            sha256sum "$plugins/$label.so" "$plugins/$label.trace.so"
        } >> "$dest/child-build.txt"
    done
    local direct=() name
    for name in "${names[@]}"; do
        case $name in openblas|libxsmm-*) continue ;; esac
        direct+=(--vendor "$name=$vendor/$name.so")
    done
    local native=armpl
    if [[ $arch == x86_64 ]]; then native=mkl; fi
    local hybrid=(--candidate "hybrid-openblas=hybrid-next:$vendor/openblas.so"
        --candidate "hybrid-$native=hybrid-next:$vendor/$native.so")

    # Disjoint instrumented calls establish the child route. The timed kernels
    # have no counters. Keep numerical and timing evidence even if a probe fails.
    local route_status=0 study_status=0
    taskset -c "$cpu" python3 "$dir/hk-child-routes.py" "$dest/panel.csv" "$dest/routes.csv" \
        "$plugins" "${route_args[@]}" || route_status=$?
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$dest/panel.csv" "$dest" \
        "$vendor/openblas.so" --phase tile-child --rounds 3 --stress-all \
        "${direct[@]}" "${hybrid[@]}" "${candidates[@]}" || study_status=$?
    if (( route_status || study_status )); then
        printf 'Child-threshold replay incomplete (route probes %d, paired study %d)\n' \
            "$route_status" "$study_status" >&2
        return 1
    fi
}
