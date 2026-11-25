# Measure the pinned release and the optional transposed tile on the same native core.
run_medium() {
    local dest="${out%/*}/medium-transfer" plugins="$build/medium-plugins" baseline="$build/medium-source"
    local parent=b218beff1eaf8fd606d0934405cdc1e28ecdfe15 old=medium-incumbent new=medium-transposed
    local flag=-DCOB_ENABLE_MEDIUM_TRANSPOSED_TILE=1 tracer=-DCOB_MEDIUM_TRANSPOSE_TRACE=1
    local tree name mode file failure=0
    local entries=() extra=()
    mkdir -p "$dest" "$plugins" || return 1
    cp "$panel" "$dest/panel.csv" || return 1
    printf '%s  %s\n' eb886434a55e82f6aa6ececafcd65e3b8dd9c746e1c15b2b05de3dff5f7e0a44 "$dest/panel.csv" |
        sha256sum --check --status || return 1
    cp "$out"/{host.txt,build.txt,tile-build.txt} "$dest/" || return 1
    for file in "$vendor"/*.txt; do if [[ -f $file ]]; then cp "$file" "$dest/" || return 1; fi; done
    python3 "$dir/medium/prepare.py" "$repo" "$baseline" > "$dest/medium-source.txt" || return 1
    for name in "$old" "$new"; do
        if [[ $name == "$old" ]]; then tree=$baseline; extra=(); else tree=$baseline.candidate; extra=("$flag"); fi
        compile_entry "$tree" "$plugins/$name.so" "${extra[@]}" || return 1
        # The timed DSO stays uninstrumented. The companion records whether the previous
        # x86 panel actually completed. A count delta isolates the current public call.
        cp -R "$tree" "$tree.route" || return 1
        python3 - "$tree.route/src/gemm.c" <<'PY' || return 1
from pathlib import Path
import sys
p = Path(sys.argv[1])
source = p.read_text()
marker = "cob_sgemm_x86_medium_panel(m, n, k, a, lda, b, ldb, c, ldc)) return;"
if source.count(marker) != 1:
    raise ValueError("unrecognized incumbent dispatch")
source = source.replace(marker, marker.replace("return;", "{ ++medium_probe_hits; return; }"))
declaration = ("static _Thread_local int medium_probe_hits;\n"
               "int rank_old_medium_successes(void) { return medium_probe_hits; }\n")
p.write_text(declaration + source)
PY
        if [[ $name == "$new" ]]; then extra+=("$tracer"); fi
        compile_entry "$tree.route" "$plugins/$name.trace.so" "${extra[@]}" || return 1
    done
    {
        printf 'baseline=%s\nsource_revision=%s\ncpu=%s\ncommon_flags=%s\ntarget_flags=%s\ncandidate_flag=%s\ntrace_flag=%s\n' \
            "$parent" "$(git -C "$repo" rev-parse HEAD)" "$cpu" "${common[*]}" "${target[*]}" "$flag" "$tracer"
        sha256sum "$plugins"/*.so "$build/worker" "$baseline"/src/gemm.c "$baseline.candidate"/src/gemm.c \
            "$baseline.route"/src/gemm.c "$baseline.candidate.route"/src/gemm.c "$dest/panel.csv" \
            "$dir"/{worker.c,study.py,aligned_admission.py,medium-routes.py,medium-study.inc} "$dir/medium/prepare.py"
    } > "$dest/medium-build.txt" || return 1
    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv"
    medium_entry() {
        local label=$1 role=$2 leaf=$3 library=$4 variant=$5
        if [[ ! -s $library ]]; then
            printf '%s,%s,%s,unavailable,missing_library\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
            return 1
        fi
        printf '%s,%s,%s,included,requires_four_full_fp64_and_applicable_route_checks\n' \
            "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
        sha256sum "$library" >> "$dest/medium-build.txt" || return 1
        case $variant in
            oracle) ;;
            vendor) entries+=(--vendor "$label=$library") ;;
            direct) entries+=(--candidate "$label=direct:$library") ;;
        esac
    }
    medium_entry openblas direct openblas "$vendor/openblas.so" oracle || return 1
    for name in "${names[@]}"; do
        case $name in
            openblas) continue ;;
            libxsmm-*) printf '%s,small_jit,,ineligible,above_size_domain\n' "$name" >> "$dest/eligibility.csv"; continue ;;
        esac
        medium_entry "$name" direct "$name" "$vendor/$name.so" vendor || failure=1
    done
    for mode in oneshot cached-plan; do
        name=onednn-$mode
        if [[ -f $vendor/onednn/qualified && -s $vendor/onednn/onednn-f32-$mode.so ]]; then
            medium_entry "$name" direct onednn "$vendor/onednn/onednn-f32-$mode.so" vendor || failure=1
        else
            printf '%s,direct,onednn,unavailable,qualified_library_unavailable\n' "$name" >> "$dest/eligibility.csv"
            failure=1
        fi
    done
    medium_entry "$old" standalone cob "$plugins/$old.so" direct || return 1
    medium_entry "$new" standalone cob "$plugins/$new.so" direct || return 1
    taskset -c "$cpu" python3 "$dir/medium-routes.py" "$panel" "$plugins" "$dest/routes.csv" || failure=1
    export RANK_MEDIUM_BATCH=1 RANK_VERIFY_INPUTS=1
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$panel" "$dest" "$vendor/openblas.so" \
        --phase medium-transfer --rounds 3 --stress-all "${entries[@]}" || failure=1
    return "$failure"
}
