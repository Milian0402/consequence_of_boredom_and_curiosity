# Sourced by tile-study.sh. All methods use the same frozen inputs and core.
run_leaf_wave() {
    local dest="${out%/*}/leaf-wave" plugins="$build/leaf-wave-plugins" snapshot="$build/leaf-wave-baseline"
    local base=0cff6ec089a43e65eeb3bfefd12a1fc68f205be4 old=cob-hk15-parent-960
    local native=armpl name flag info faults=0 use_x86=0
    local trace_args=() standalone=("$old" winograd-cob-1 winograd-cob-2)
    local active=(-DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1)
    local direct=() candidates=()
    if [[ $arch == x86_64 ]]; then
        native=mkl
        if grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then
            use_x86=1
            trace_args+=(--x86)
            standalone+=(hk-register hk-k288)
        fi
    fi
    mkdir -p "$dest" "$plugins" "$snapshot"
    cp "$dir/leaf-wave.csv" "$dest/panel.csv"
    printf '%s  %s\n' e1fb795b4de6b84ad332bdf55bcc8571f3861325009c92ea904ca22ae57cf5e6 "$dest/panel.csv" | sha256sum -c --status || return 1
    cp "$out/host.txt" "$out/build.txt" "$out/tile-build.txt" "$dest/"
    for info in "$vendor"/*.txt; do if [[ -f $info ]]; then cp "$info" "$dest/"; fi; done
    git -C "$repo" archive "$base" src include tools/world-rank/cob_adapter.c tools/world-rank/vendor.h \
        tools/world-rank/finite/hk233.inc | tar -x -C "$snapshot" || return 1
    compile_entry "$snapshot" "$plugins/$old.so" "${active[@]}" || return 1
    compile_entry "$snapshot" "$plugins/$old.trace.so" "${active[@]}" -DCOB_HK15_CHILD_TRACE=1 || return 1
    bash "$dir/standalone-winograd/build.sh" "$repo" "$plugins" || return 1
    cp "$plugins/winograd-cob-build.txt" "$dest/"
    bash "$dir/leaf-public-openblas.sh" "$repo" "$vendor" "$build/leaf-public-openblas.so" || return 1
    cp "$build/leaf-public-openblas.so" "$plugins/cob-openblas-winograd.so" || return 1
    cp "$build/leaf-public-openblas.trace.so" "$plugins/cob-openblas-winograd.trace.so" || return 1
    cp "$build/leaf-public-openblas.previous.so" "$plugins/cob-openblas-previous.so" || return 1
    for info in "$build"/leaf-public-openblas*.txt; do if [[ -f $info ]]; then cp "$info" "$dest/"; fi; done
    if (( use_x86 )); then
        for name in hk-register hk-k288; do
            case "$name" in
                hk-register) flag=-DCOB_ENABLE_HK_X86_REGISTER=1 ;;
                hk-k288) flag=-DCOB_ENABLE_HK_X86_K288=1 ;;
            esac
            compile_entry "$repo" "$plugins/$name.so" "${active[@]}" "$flag" || return 1
            compile_entry "$repo" "$plugins/$name.trace.so" "${active[@]}" "$flag" -DCOB_HK_X86_LEAF_TRACE=1 || return 1
        done
    fi
    {
        printf 'baseline_revision=%s\ncandidate_revision=%s\ncommon_flags=%s\ntarget_flags=%s\nhk_flags=%s\n' \
            "$base" "$(git -C "$repo" rev-parse HEAD)" "${common[*]}" "${target[*]}" "${active[*]}"
        sha256sum "$plugins"/*.so "$build/worker" "$dir"/leaf-wave*.{csv,py,inc} "$dir/leaf-public-openblas.sh" \
            "$dir/standalone-winograd/build.sh" "$dir/standalone-winograd/adapter.c" \
            "$dir/standalone-winograd/one_leaf.c" "$dir/standalone-winograd/schedule.patch" \
            "$dir/study.py" "$dir/hybrid_next.c" \
            "$dir/hk-parent-routes.py" "$dir/hk-child-routes.py" "$dir/hk-parent-vendors.py" \
            "$snapshot/src/gemm.c" "$repo/src/gemm.c" "$dest/panel.csv"
    } > "$dest/leaf-wave-build.txt" || return 1
    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv"
    for name in "${names[@]}"; do
        case "$name" in
            libxsmm-*) printf '%s,small_jit,,ineligible,above_size_domain\n' "$name" >> "$dest/eligibility.csv"; continue ;;
            openblas) ;; # Supplied as the positional reference.
            *) direct+=(--vendor "$name=$vendor/$name.so") ;;
        esac
        printf '%s,direct,,included,dense_fp32\n' "$name" >> "$dest/eligibility.csv"
        sha256sum "$vendor/$name.so" >> "$dest/leaf-wave-build.txt" || return 1
    done
    for name in "${standalone[@]}"; do
        candidates+=(--candidate "$name=direct:$plugins/$name.so")
        printf '%s,standalone,cob,included,native_single_thread_entry\n' "$name" >> "$dest/eligibility.csv"
    done
    if (( ! use_x86 )); then
        for name in hk-register hk-k288; do
            printf '%s,standalone,cob,ineligible,requires_x86_avx2_fma\n' "$name" >> "$dest/eligibility.csv"
        done
    fi
    candidates+=(--candidate "cob-openblas-winograd=direct:$plugins/cob-openblas-winograd.so")
    printf 'cob-openblas-winograd,public_winograd,openblas,included,statically_linked_serial_openblas\n' >> "$dest/eligibility.csv"
    candidates+=(--candidate "cob-openblas-previous=direct:$plugins/cob-openblas-previous.so")
    printf 'cob-openblas-previous,public_winograd_previous,openblas,included,prior_public_entry_with_same_pinned_static_blas\n' >> "$dest/eligibility.csv"
    for name in openblas "$native"; do
        candidates+=(--candidate "hybrid-$name=hybrid-next:$vendor/$name.so")
        printf 'hybrid-%s,cob_winograd,%s,included,worker_bridge_counter\n' "$name" "$name" >> "$dest/eligibility.csv"
    done
    # Route checks run outside timing. Retain the other methods' results after an unqualified rival.
    taskset -c "$cpu" python3 "$dir/leaf-wave-routes.py" "$dest/panel.csv" "$dest" "$plugins" "$vendor" \
        "${trace_args[@]}" || faults=1
    for name in fastmmw-768 fastmmw-1536; do
        if [[ -s $vendor/$name-build.txt ]] && python3 "$dir/leaf-wave-routes.py" --qualify "$dest" "$name"; then
            direct+=(--vendor "$name=$vendor/$name.so")
            printf '%s,external_winograd,openblas,included,qualify_per_layout_using_route_and_four_fp64_checks\n' "$name" >> "$dest/eligibility.csv"
            sha256sum "$vendor/$name.so" "$vendor/$name.trace.so" >> "$dest/leaf-wave-build.txt" || return 1
        else
            faults=1
            printf '%s,external_winograd,openblas,excluded,failed_native_qualification\n' "$name" >> "$dest/eligibility.csv"
        fi
    done
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$dest/panel.csv" "$dest" "$vendor/openblas.so" \
        --phase leaf-wave --rounds 3 --stress-all "${direct[@]}" "${candidates[@]}" || faults=1
    python3 "$dir/hk-parent-vendors.py" hybrids "$dest" "$native" || faults=1
    return "$faults"
}
