# Execute the released and experimental production entry points independently of vendor adapters.
run_stream_b() {
    local dest="${out%/*}/stream-b" plugins="$build/stream-plugins" status=0 name mode library i
    local entries=()
    mkdir -p "$dest" "$plugins" || return 1
    cp "$panel" "$dest/panel.csv" || return 1
    cp "$dir/stream/"{protocol.md,preregistration.json} "$out/"{host.txt,build.txt,tile-build.txt} "$dest/" || return 1
    for file in "$vendor"/*.txt; do if [[ -f $file ]]; then cp "$file" "$dest/" || return 1; fi; done
    bash "$dir/stream/build.sh" "$repo" "$plugins" "$dest" > "$dest/stream-build.log" 2>&1 || { tail -50 "$dest/stream-build.log"; return 1; }
    taskset -c "$cpu" python3 -B "$dir/stream/routes.py" "$panel" "$plugins" "$dest" || status=1
    printf 'implementation,role,leaf,status,reason\n' > "$dest/eligibility.csv" || return 1
    stream_entry() {
        local label=$1 role=$2 leaf=$3 file=$4 abi=$5
        if [[ ! -s $file ]]; then
            printf '%s,%s,%s,unavailable,missing_or_unqualified_native_method\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
            return 1
        fi
        printf '%s,%s,%s,included,requires_full_fp64_and_applicable_route\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
        sha256sum "$file" >> "$dest/stream-binaries.sha256"
        case $abi in
            oracle) ;;
            vendor) entries+=(--vendor "$label=$file") ;;
            linked-public) entries+=(--candidate "$label=linked-public:$file") ;;
            *) return 2 ;;
        esac
    }
    stream_entry openblas direct openblas "$vendor/openblas.so" oracle || return 1
    for name in mkl mkl-jit aocl blasfeo; do
        stream_entry "$name" direct "$name" "$vendor/$name.so" vendor || status=1
    done
    for mode in oneshot cached-plan; do
        name=onednn-$mode
        library="$vendor/onednn/onednn-f32-$mode.so"
        if [[ -s $library && -f $vendor/onednn/qualified ]]; then stream_entry "$name" direct onednn "$library" vendor || status=1
        else printf '%s,direct,onednn,unavailable,strict_native_runtime_qualification_failed\n' "$name" >> "$dest/eligibility.csv"; status=1; fi
    done
    while IFS=, read -r name isa variant buildstatus remainder; do
        [[ $name != implementation ]] || continue
        if [[ $buildstatus == unsupported ]]; then
            printf '%s,standalone,cob,ineligible,unsupported_native_instruction_set\n' "$name" >> "$dest/eligibility.csv"
        else stream_entry "$name" standalone cob "$plugins/$name.worker" linked-public || return 1; fi
    done < "$dest/builds.csv"
    export RANK_MEDIUM_BATCH=1 RANK_VERIFY_INPUTS=1
    taskset -c "$cpu" python3 -B "$dir/study.py" "$build/worker" "$panel" "$dest" "$vendor/openblas.so" \
        --phase stream-b --rounds 3 --stress-all "${entries[@]}" || status=1
    python3 -B "$dir/stream/score.py" "$dest" || status=1
    return "$status"
}
