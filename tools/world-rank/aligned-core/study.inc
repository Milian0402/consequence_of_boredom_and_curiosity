# Sourced by tile-study.sh on a pinned native host.
run_aligned_core() {
    local dest="${out%/*}/$phase" plugins="$build/aligned-plugins" snapshot="$build/hk-baseline"
    local revision=0cff6ec089a43e65eeb3bfefd12a1fc68f205be4 incumbent=cob-hk15-parent-960
    local native=armpl name file provider mode fail=0 prototype=$repo
    local modes=(one two aligned-two)
    local options=(-DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1)
    local entries=()
    [[ $arch != x86_64 ]] || native=mkl
    mkdir -p "$dest" "$plugins" "$snapshot"
    (cd "$dir/$phase" && sha256sum --check frozen.sha256) || return 1
    cp "$panel" "$dest/panel.csv"
    cp "$dir/$phase/protocol.md" "$out"/{host.txt,build.txt,tile-build.txt} "$dest/"
    for file in "$vendor"/*.txt; do if [[ -f $file ]]; then cp "$file" "$dest/"; fi; done
    git -C "$repo" archive "$revision" src include tools/world-rank/cob_adapter.c tools/world-rank/vendor.h \
        tools/world-rank/finite/hk233.inc | tar -x -C "$snapshot" || return 1
    compile_entry "$snapshot" "$plugins/$incumbent.so" "${options[@]}" || return 1
    compile_entry "$snapshot" "$plugins/$incumbent.trace.so" "${options[@]}" -DCOB_HK15_CHILD_TRACE=1 || return 1
    if [[ $phase == release-replay ]]; then
        stage_release || fail=1
        modes=(two aligned-two)
    fi
    bash "$prototype/tools/world-rank/vendor-public/build-native.sh" "$prototype" "$vendor" "$plugins" || fail=1
    for file in "$plugins"/*-build.txt; do if [[ -f $file ]]; then cp "$file" "$dest/"; fi; done
    {
        printf 'baseline=%s\nsource_revision=%s\ncpu=%s\ncommon_flags=%s\ntarget_flags=%s\nincumbent_flags=%s\n' \
            "$revision" "$(git -C "$repo" rev-parse HEAD)" "$cpu" "${common[*]}" "${target[*]}" "${options[*]}"
        sha256sum "$plugins"/*.so "$build/worker" "$snapshot/src/gemm.c" "$dest/panel.csv"
        sha256sum "$dir"/{worker.c,study.py,aligned_admission.py} "$dir"/aligned-core/*.{csv,md,py,sha256,inc} "$dir"/vendor-public/*.{c,sh,py,patch}
    } > "$dest/aligned-build.txt" || return 1
    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv"
    aligned_entry() {
        local label=$1 role=$2 leaf=$3 path=$4 variant=$5
        if [[ ! -s $path ]]; then
            printf '%s,%s,%s,unavailable,missing_library_or_failed_qualification\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
            return 1
        fi
        printf '%s,%s,%s,included,subject_to_four_full_fp64_and_layout_routes\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
        sha256sum "$path" >> "$dest/aligned-build.txt"
        case $variant in
            oracle) ;;
            vendor) entries+=(--vendor "$label=$path") ;;
            *) entries+=(--candidate "$label=$variant:$path") ;;
        esac
    }
    aligned_entry openblas direct openblas "$vendor/openblas.so" oracle || return 1
    for name in "${names[@]}"; do
        [[ $name == openblas ]] || aligned_entry "$name" direct "$name" "$vendor/$name.so" vendor || fail=1
    done
    for name in fastmmw-768 fastmmw-1536; do
        aligned_entry "$name" external_winograd openblas "$vendor/$name.so" vendor || fail=1
    done
    for mode in oneshot cached-plan; do
        name="onednn-$mode"
        if [[ -f $vendor/onednn/qualified ]]; then
            aligned_entry "$name" direct onednn "$vendor/onednn/onednn-f32-$mode.so" vendor || fail=1
        else
            printf '%s,direct,onednn,unavailable,native_strict_sequential_smoke_missing\n' "$name" >> "$dest/eligibility.csv"
            fail=1
        fi
    done
    if [[ -d $vendor/tblis-evidence ]]; then cp -R "$vendor/tblis-evidence" "$dest/"; fi
    for mode in 1 2; do
        name=tblis-ab-$mode
        if [[ $arch == x86_64 && -f $vendor/$name.so && -f $vendor/tblis-status.txt ]] && grep -q "^qualified:" "$vendor/tblis-status.txt"; then
            aligned_entry "$name" external_strassen tblis "$vendor/$name.so" vendor || fail=1
        else
            printf '%s,external_strassen,tblis,ineligible,native_intel_haswell_qualification_unavailable\n' "$name" >> "$dest/eligibility.csv"
        fi
    done
    aligned_entry "$incumbent" standalone cob "$plugins/$incumbent.so" direct || return 1
    if [[ $phase == release-replay ]]; then register_release || fail=1; fi
    for provider in openblas "$native"; do
        for mode in "${modes[@]}"; do
            name="cob-$provider-$mode"
            aligned_entry "$name" public_winograd "$provider" "$plugins/$name.so" direct || fail=1
        done
        aligned_entry "hybrid-$provider" worker_winograd "$provider" "$vendor/$provider.so" hybrid-next || fail=1
    done
    # Trace the public entry separately from the uninstrumented timing process.
    taskset -c "$cpu" python3 "$prototype/tools/world-rank/vendor-public/check-routes.py" "$panel" "$plugins" "$dest/aligned-routes.csv" || fail=1
    taskset -c "$cpu" python3 "$dir/aligned-core/peer-routes.py" "$panel" "$plugins" "$vendor" "$dest" || fail=1
    if [[ $phase == release-replay ]]; then verify_release || fail=1; fi
    export RANK_VERIFY_INPUTS=1
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$panel" "$dest" "$vendor/openblas.so" \
        --phase "$phase" --rounds 3 --stress-all "${entries[@]}" || fail=1
    python3 "$dir/hk-parent-vendors.py" hybrids "$dest" "$native" || fail=1
    if [[ $phase == release-replay ]]; then python3 "$dir/release-replay/score.py" "$dest" || fail=1; fi
    return "$fail"
}
