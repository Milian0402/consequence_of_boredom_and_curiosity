# These hooks share the established single-core controller and artifact schemas.
stage_release() {
    local source="$build/production" frozen=29f5817ed45a728bc012f332fa378dc95c6e7ef5
    prototype=$build/prototype
    mkdir -p "$prototype"
    git -C "$repo" archive "$frozen" src include tools/world-rank | tar -x -C "$prototype" || return 1
    cp "$dir/release-replay/"{source-identity.json,training-hosts.json,novelty.txt,frozen.sha256,fixture-correction.md} "$dest/"
    cp "$dir/release-public/test-backport.json" "$dest/"
    python3 "$dir/release-replay/prepare-sources.py" "$repo" "$source" | tee "$dest/production-source-check.txt" || return 1
    bash "$dir/release-public/build-native.sh" "$source" "$repo" "$vendor" "$plugins" "$source.previous" || return 1
}

register_release() {
    local status=0 name provider mode
    release_entry() {
        local label=$1 role=$2 leaf=$3
        if [[ ! -f $plugins/$label.qualified ]]; then
            printf '%s,%s,%s,unavailable,production_build_or_tests_failed\n' "$label" "$role" "$leaf" >> "$dest/eligibility.csv"
            return 1
        fi
        aligned_entry "$label" "$role" "$leaf" "$plugins/$label.worker" linked-public
    }
    if [[ -f $plugins/release-build-status.csv ]]; then cp "$plugins/release-build-status.csv" "$dest/"; fi
    release_entry release-previous original_release openblas || status=1
    for provider in openblas "$native"; do
        for mode in two aligned-two; do
            name="release-$provider-$mode"
            release_entry "$name" production_winograd "$provider" || status=1
        done
    done
    for file in "$plugins"/*-tests.txt; do if [[ -f $file ]]; then cp "$file" "$dest/"; fi; done
    return "$status"
}

verify_release() {
    sha256sum "$dir"/{worker.c,study.py,aligned_admission.py} "$dir"/release-public/*.{c,sh,py,json,gz} \
        "$dir"/release-replay/*.{csv,json,md,py,gz,inc,txt,sha256,sh} > "$dest/replay-build.txt" || return 1
    taskset -c "$cpu" python3 "$dir/release-public/check-routes.py" "$panel" "$plugins" "$dest/release-routes.csv"
}
