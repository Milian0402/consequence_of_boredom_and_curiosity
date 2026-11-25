# Separate complete one-thread products on the frozen HK prepacking panel.
run_hk_forms() {
    local dest="${out%/*}/hk-forms" plugins="$build/hk-forms-plugins" snapshot="$build/hk-forms-old"
    local base=29f5817ed45a728bc012f332fa378dc95c6e7ef5 old=cob-hk15-parent-960 new=hk-packed-forms
    local native=armpl name file mode failure=0
    local options=(-DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1)
    local packed=-DCOB_ENABLE_HK_PACK_B_FORMS=1
    local entries=()
    [[ $arch != x86_64 ]] || native=mkl
    mkdir -p "$dest" "$plugins" "$snapshot" || return 1
    cp "$panel" "$dest/panel.csv" || return 1
    printf '%s  %s\n' af82edd4b9c3200279d0bbea7471e0738359417780efaec73b8bbb60433d9603 "$dest/panel.csv" |
        sha256sum --check --status || return 1
    cp "$out"/{host.txt,build.txt,tile-build.txt} "$dest/" || return 1
    for file in "$vendor"/*.txt; do if [[ -f $file ]]; then cp "$file" "$dest/" || return 1; fi; done
    if [[ -d $vendor/tblis-evidence ]]; then cp -R "$vendor/tblis-evidence" "$dest/" || return 1; fi
    git -C "$repo" archive "$base" src include tools/world-rank/cob_adapter.c tools/world-rank/vendor.h \
        tools/world-rank/finite/hk233.inc | tar -x -C "$snapshot" || return 1
    compile_entry "$snapshot" "$plugins/$old.so" "${options[@]}" || return 1
    compile_entry "$snapshot" "$plugins/$old.trace.so" "${options[@]}" -DCOB_HK15_CHILD_TRACE=1 || return 1
    compile_entry "$repo" "$plugins/$new.so" "${options[@]}" "$packed" || return 1
    compile_entry "$repo" "$plugins/$new.trace.so" "${options[@]}" "$packed" \
        -DCOB_HK15_CHILD_TRACE=1 -DCOB_HK_PACK_B_TRACE=1 || return 1
    {
        printf 'baseline=%s\nsource_revision=%s\ncpu=%s\ncommon_flags=%s\ntarget_flags=%s\nincumbent_flags=%s\ncandidate_flags=%s\n' \
            "$base" "$(git -C "$repo" rev-parse HEAD)" "$cpu" "${common[*]}" "${target[*]}" "${options[*]}" "${options[*]} $packed"
        sha256sum "$plugins"/*.so "$build/worker" "$snapshot/src/gemm.c" "$snapshot/src/cob_hk15.inc" \
            "$repo"/src/{gemm.c,cob_hk_pack_b.inc,cob_hk15.inc,cob_fused_strassen.inc} "$dir"/finite/hk233.inc \
            "$dir"/{hk-forms-routes.py,hk-forms-study.inc,aligned_admission.py,study.py,worker.c} "$dest/panel.csv"
    } > "$dest/hk-forms-build.txt" || return 1
    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv"
    forms_entry() {
        local label=$1 role=$2 leaf=$3 path=$4 variant=$5
        if [[ ! -s $path ]]; then
            printf '%s,%s,%s,unavailable,missing_library\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
            return 1
        fi
        printf '%s,%s,%s,included,subject_to_four_fp64_and_applicable_native_routes\n' \
            "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
        sha256sum "$path" >> "$dest/hk-forms-build.txt" || return 1
        case $variant in
            oracle) ;;
            vendor) entries+=(--vendor "$label=$path") ;;
            *) entries+=(--candidate "$label=$variant:$path") ;;
        esac
    }
    forms_entry openblas direct openblas "$vendor/openblas.so" oracle || return 1
    for name in "${names[@]}"; do
        case $name in
            openblas) continue ;;
            libxsmm-*) printf '%s,small_jit,,ineligible,above_size_domain\n' "$name" >> "$dest/eligibility.csv"; continue ;;
        esac
        forms_entry "$name" direct "$name" "$vendor/$name.so" vendor || failure=1
    done
    for name in fastmmw-768 fastmmw-1536; do
        forms_entry "$name" external_winograd openblas "$vendor/$name.so" vendor || failure=1
        if [[ -s $vendor/$name.trace.so ]]; then sha256sum "$vendor/$name.trace.so" >> "$dest/hk-forms-build.txt"; fi
    done
    for mode in oneshot cached-plan; do
        name=onednn-$mode
        if [[ -f $vendor/onednn/qualified && -s $vendor/onednn/onednn-f32-$mode.so ]]; then
            forms_entry "$name" direct onednn "$vendor/onednn/onednn-f32-$mode.so" vendor || failure=1
        else
            printf '%s,direct,onednn,unavailable,qualified_prebuilt_library_unavailable\n' "$name" >> "$dest/eligibility.csv"
        fi
    done
    for mode in 1 2; do
        name=tblis-ab-$mode
        if [[ $arch == x86_64 && -s $vendor/$name.so && -s $vendor/tblis-status.txt ]] &&
            grep -q '^qualified:' "$vendor/tblis-status.txt"; then
            forms_entry "$name" external_strassen tblis "$vendor/$name.so" vendor || failure=1
        else
            printf '%s,external_strassen,tblis,ineligible,qualified_native_library_unavailable\n' "$name" >> "$dest/eligibility.csv"
        fi
    done
    forms_entry "$old" standalone cob "$plugins/$old.so" direct || return 1
    forms_entry "$new" standalone cob "$plugins/$new.so" direct || return 1
    for name in openblas "$native"; do
        forms_entry "hybrid-$name" worker_winograd "$name" "$vendor/$name.so" hybrid-next || failure=1
    done
    taskset -c "$cpu" python3 "$dir/hk-forms-routes.py" "$panel" "$plugins" "$vendor" "$dest" || failure=1
    export RANK_VERIFY_INPUTS=1
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$panel" "$dest" "$vendor/openblas.so" \
        --phase hk-forms --rounds 3 --stress-all "${entries[@]}" || failure=1
    python3 "$dir/hk-parent-vendors.py" hybrids "$dest" "$native" || failure=1
    return "$failure"
}
