# Sourced from tile-study.sh after compile_entry, the fixed flags and vendor DSOs are ready.
run_hk_parent() {
    local dest="${out%/*}/hk-parent" baseline=9df006cc27398802d9dc24b73b5208800be5c155
    local old=cob-hk15-public primary=cob-hk15-parent-960
    local snapshot="$build/hk-parent-baseline" plugins="$build/hk-parent-plugins"
    local info name native=armpl
    [[ $arch != x86_64 ]] || native=mkl
    mkdir -p "$dest" "$snapshot" "$plugins"
    git -C "$repo" archive "$baseline" src include tools/world-rank/cob_adapter.c \
        tools/world-rank/vendor.h tools/world-rank/finite/hk233.inc | tar -x -C "$snapshot"
    cp "$dir/hk-parent.csv" "$dest/panel.csv"
    printf '%s  %s\n' 707584c88806ea83b13cc0b5e63815a967751a099cc49b0b27e8fdba413b386b \
        "$dest/panel.csv" | sha256sum -c --status || return 1
    cp "$out/host.txt" "$out/build.txt" "$out/tile-build.txt" "$dest/"
    for info in "$vendor"/*.txt; do if [[ -f $info ]]; then cp "$info" "$dest/"; fi; done

    compile_entry "$snapshot" "$plugins/$old.so" -DCOB_ENABLE_HK15=1
    compile_entry "$repo" "$plugins/$old.trace.so" -DCOB_ENABLE_HK15=1 \
        -DCOB_HK15_CHILD_HARMONIC=1024 -DCOB_HK15_CHILD_TRACE=1
    compile_entry "$repo" "$plugins/$primary.so" -DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1
    compile_entry "$repo" "$plugins/$primary.trace.so" -DCOB_ENABLE_HK15=1 \
        -DCOB_ENABLE_HK15_PARENT_960=1 -DCOB_HK15_CHILD_TRACE=1
    {
        printf 'baseline_revision=%s\ncandidate_revision=%s\n' "$baseline" "$(git -C "$repo" rev-parse HEAD)"
        printf 'common_flags=%s\ntarget_flags=%s\n' "${common[*]}" "${target[*]}"
        printf '%s\n' \
            "$old timed: snapshot $baseline -DCOB_ENABLE_HK15=1" \
            "$old trace: candidate source -DCOB_ENABLE_HK15=1 -DCOB_HK15_CHILD_HARMONIC=1024 -DCOB_HK15_CHILD_TRACE=1" \
            "$primary timed: candidate source -DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1" \
            "$primary trace: candidate flags plus -DCOB_HK15_CHILD_TRACE=1"
        sha256sum "$plugins"/*.so "$dest/panel.csv" "$build/worker" "$dir/study.py" \
            "$dir/hk-parent-study.inc" "$dir/hk-parent-routes.py" "$dir/hk-child-routes.py" \
            "$dir/hk-parent-vendors.py" "$dir/hybrid_next.c" "$snapshot/src/gemm.c" \
            "$snapshot/src/cob_hk15.inc" "$snapshot/tools/world-rank/finite/hk233.inc" \
            "$repo/src/gemm.c" "$repo/src/cob_hk15.inc" "$repo/src/cob_fused_strassen.inc" \
            "$repo/tools/world-rank/finite/hk233.inc"
    } > "$dest/parent-build.txt"

    local direct=() available=() qualification_status=0 routes_status=0 study_status=0 hybrid_status=0
    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv"
    for name in "${names[@]}"; do
        case "$name" in
            libxsmm-*) printf '%s,small_jit,,ineligible,above_size_domain\n' "$name" >> "$dest/eligibility.csv"; continue ;;
            openblas) ;; # Passed once as study.py's positional reference.
            *) direct+=(--vendor "$name=$vendor/$name.so") ;;
        esac
        printf '%s,direct,,included,dense_fp32\n' "$name" >> "$dest/eligibility.csv"
        sha256sum "$vendor/$name.so" >> "$dest/parent-build.txt"
    done
    for name in fastmmw-768 fastmmw-1536; do
        if [[ -s $vendor/$name.so && -s $vendor/$name.trace.so && -s $vendor/$name-build.txt ]]; then
            available+=("$name")
            sha256sum "$vendor/$name.so" "$vendor/$name.trace.so" "$vendor/$name-build.txt" >> "$dest/parent-build.txt"
        else
            printf '%s,external_winograd,openblas,excluded,missing_binary_trace_or_native_manifest\n' "$name" >> "$dest/eligibility.csv"
            qualification_status=1
        fi
    done
    taskset -c "$cpu" python3 "$dir/hk-parent-routes.py" "$dest/panel.csv" "$dest" "$plugins" "$vendor" \
        --fastmmw "${available[@]}" || routes_status=$?
    for name in "${available[@]}"; do
        if python3 "$dir/hk-parent-vendors.py" qualified "$dest" "$name"; then
            direct+=(--vendor "$name=$vendor/$name.so")
            printf '%s,external_winograd,openblas,included,exact_padded_output_and_native_leaf_calls\n' "$name" >> "$dest/eligibility.csv"
        else
            printf '%s,external_winograd,openblas,excluded,failed_native_route_evidence\n' "$name" >> "$dest/eligibility.csv"
            qualification_status=1
        fi
    done
    local candidates=(--candidate "$old=direct:$plugins/$old.so" --candidate "$primary=direct:$plugins/$primary.so")
    for name in openblas "$native"; do
        candidates+=(--candidate "hybrid-$name=hybrid-next:$vendor/$name.so")
        printf 'hybrid-%s,cob_winograd,%s,included,worker_bridge_counter\n' "$name" "$name" >> "$dest/eligibility.csv"
    done
    for name in "$old" "$primary"; do
        printf '%s,standalone,,included,native_public_entry\n' "$name" >> "$dest/eligibility.csv"
    done
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$dest/panel.csv" "$dest" "$vendor/openblas.so" \
        --phase tile-parent --rounds 3 --stress-all "${direct[@]}" "${candidates[@]}" || study_status=$?
    python3 "$dir/hk-parent-vendors.py" hybrids "$dest" "$native" || hybrid_status=$?
    if (( qualification_status || routes_status || study_status || hybrid_status )); then
        printf 'Incomplete HK parent confirmation (eligibility %d, routes %d, study %d, hybrid proof %d)\n' \
            "$qualification_status" "$routes_status" "$study_status" "$hybrid_status" >&2
        return 1
    fi
}
