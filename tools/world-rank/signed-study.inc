# shellcheck shell=bash
# Sourced by tile-study.sh after the common native worker and vendor adapters are available.
run_signed_leaves() {
    local dest="${out%/*}/signed-leaves" plugins="$build/signed-build" native=armpl
    local study_status=0 name role provider file variant library
    local entries=() providers=(openblas armpl)
    if [[ $arch == x86_64 ]]; then native=mkl; providers=(openblas mkl); fi
    mkdir -p "$dest" "$plugins" || return 1
    printf '%s  %s\n' a6252eac41350bc92c87c15476441ab30aec65b6dc9602715ddbb44f98a2d2c9 "$panel" |
        sha256sum --check --status || return 1
    cp "$panel" "$dest/panel.csv" || return 1
    cp "$dir/signed/frozen/protocol.md" "$dest/protocol.md" || return 1
    cp "$dir/signed/protocol.md" "$dest/transfer-protocol.md" || return 1
    cp "$out"/{host.txt,build.txt,tile-build.txt} "$dest/" || return 1
    for file in "$vendor"/*.txt; do
        if [[ -f $file ]]; then cp "$file" "$dest/" || return 1; fi
    done
    export RANK_VERIFY_INPUTS=1
    unset RANK_MEDIUM_BATCH
    bash "$dir/signed/build.sh" "$repo" "$vendor" "$plugins" "$dest" || return 1
    taskset -c "$cpu" python3 "$dir/signed/routes.py" "$panel" "$plugins" "$dest/routes.csv" || return 1
    taskset -c "$cpu" python3 "$dir/signed/external-routes.py" "$panel" "$vendor" "$dest/fastmmw-routes.csv" || return 1

    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv" || return 1
    printf 'implementation,basename,sha256\n' > "$dest/worker-binaries.csv" || return 1
    signed_entry() {
        local name=$1 role=$2 provider=$3 variant=$4 library=$5 digest
        if [[ ! -s $library ]]; then
            printf '%s,%s,%s,unavailable,missing_qualified_binary\n' "$name" "$role" "$provider" >> "$dest/eligibility.csv"
            return 1
        fi
        digest=$(sha256sum "$library") || return 1
        printf '%s,%s,%s\n' "$name" "${library##*/}" "${digest%% *}" >> "$dest/worker-binaries.csv" || return 1
        printf '%s,%s,%s,included,requires_full_fp64_and_applicable_routes\n' "$name" "$role" "$provider" >> "$dest/eligibility.csv" || return 1
        case $variant in
            oracle) ;;
            vendor) entries+=(--vendor "$name=$library") ;;
            *) entries+=(--candidate "$name=$variant:$library") ;;
        esac
    }
    signed_entry openblas direct openblas oracle "$vendor/openblas.so" || return 1
    for name in "${names[@]}"; do
        case $name in openblas|libxsmm-*) continue ;; esac
        if [[ $name == mkl-jit && ! -s $vendor/$name.so ]]; then
            printf '%s,direct,mkl-jit,unavailable,ordinary_sequential_jit_api_unavailable\n' "$name" >> "$dest/eligibility.csv" || return 1
            continue
        fi
        signed_entry "$name" direct "$name" vendor "$vendor/$name.so" || return 1
    done
    for variant in oneshot cached-plan; do
        name=onednn-$variant
        library="$vendor/onednn/onednn-f32-$variant.so"
        if [[ -f $vendor/onednn/qualified && -s $library ]]; then
            signed_entry "$name" direct onednn vendor "$library" || return 1
        else
            printf '%s,direct,onednn,unavailable,strict_sequential_smoke_or_binary_missing\n' "$name" >> "$dest/eligibility.csv" || return 1
        fi
    done
    for name in fastmmw-768 fastmmw-1536; do
        if [[ -s $vendor/$name.so && -s $vendor/$name.trace.so ]]; then
            # An independent Winograd adapter is an external rival in its own class.
            # Keep the coordinator's direct-only denominator composed of actual direct peers.
            signed_entry "$name" external_winograd openblas direct "$vendor/$name.so" || return 1
        else
            printf '%s,external_winograd,openblas,unavailable,plain_or_trace_binary_missing\n' "$name" >> "$dest/eligibility.csv" || return 1
        fi
    done
    for provider in "${providers[@]}"; do
        for role in original signed; do
            name=$role-$provider
            signed_entry "$name" "$role" "$provider" linked-public "$plugins/$name.worker" || return 1
        done
    done
    (
        cd "$dir" || exit 1
        sha256sum signed-study.inc signed/{build.sh,routes.py,external-routes.py,score.py,panel.csv} \
            study.py aligned_admission.py worker.c
    ) > "$dest/driver-checksums.sha256" || return 1
    taskset -c "$cpu" python3 "$dir/study.py" "$build/worker" "$panel" "$dest" "$vendor/openblas.so" \
        --phase signed-leaves --rounds 3 --stress-all "${entries[@]}" || study_status=$?
    python3 "$dir/signed/score.py" "$dest" "$native" "$study_status" "$cpu"
}
