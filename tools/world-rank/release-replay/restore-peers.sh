#!/usr/bin/env bash
# Retry any missing pinned installation before measuring. Retain all repair logs.
set -euo pipefail
dir=$(cd "$(dirname "$0")/.." && pwd)
vendor=$(cd "${1:?vendor directory}" && pwd)
raw=${2:?artifact directory}
mkdir -p "$raw/peer-restoration"
missing=0
rebuild() {
    local name=$1
    shift
    echo "Retrying $name"
    if ! "$@" > "$raw/peer-restoration/$name.log" 2>&1; then
        tail -16 "$raw/peer-restoration/$name.log" >&2
        missing=1
    fi
}
if [[ ! -s $vendor/openblas.so || ! -s $vendor/openblas-prefix.txt ]]; then rebuild openblas bash "$dir/openblas/build.sh" "$vendor"; fi
if [[ $(uname -m) == x86_64 ]]; then
    name=mkl
    if [[ ! -s $vendor/aocl.so ]]; then rebuild aocl bash "$dir/aocl/build.sh" "$vendor"; fi
else name=armpl; fi
if [[ ! -s $vendor/$name.so || ! -s $vendor/$name-prefix.txt || ( $name == mkl && ! -s $vendor/mkl-jit.so ) ]]; then
    rebuild "$name" bash "$dir/vendors/build.sh" "$vendor"
fi
if [[ ! -s $vendor/blasfeo.so ]]; then rebuild blasfeo bash "$dir/blasfeo/build.sh" "$vendor"; fi
if [[ ! -s $vendor/fastmmw-1536.so || ! -s $vendor/fastmmw-768.so || ! -s $vendor/fastmmw-768.trace.so || ! -s $vendor/fastmmw-1536.trace.so ]]; then
    rebuild fastmmw bash "$dir/fastmmw/native-ci.sh" "$vendor"
fi
onednn_available() {
    [[ -e $vendor/onednn/qualified && -s $vendor/onednn/onednn-f32-oneshot.so &&
        -s $vendor/onednn/onednn-f32-cached-plan.so && -s $vendor/onednn/qualified-oneshot.log &&
        -s $vendor/onednn/qualified-cached-plan.log ]] &&
        grep -q '^PASS ' "$vendor/onednn/qualified-oneshot.log" &&
        grep -q '^PASS ' "$vendor/onednn/qualified-cached-plan.log"
}
if ! onednn_available; then
    build=$(mktemp -d "$RUNNER_TEMP/rebuild-onednn.XXXXXX")
    rebuild onednn env JOBS=4 bash "$dir/onednn/build.sh" "$build"
    if [[ -s $build/qualified-oneshot.log && -s $build/qualified-cached-plan.log ]] &&
        grep -q '^PASS ' "$build/qualified-oneshot.log" && grep -q '^PASS ' "$build/qualified-cached-plan.log"; then
        mkdir -p "$vendor/onednn" "$raw/onednn/build"
        cp -a "$build/prefix" "$build"/onednn-f32-*.so "$vendor/onednn/"
        if bash "$dir/onednn/qualify.sh" "$vendor/onednn" "$vendor/onednn/prefix/include" > "$raw/onednn/restored.txt"; then
            cp "$vendor/onednn"/{qualified-*.log,host.txt} "$raw/onednn/"
            cp "$build"/{build-fingerprint.txt,dependencies.txt,configure.log,build.log,imports.txt,host.txt,qualified-*.log,cmake/CMakeCache.txt,cmake/compile_commands.json} "$raw/onednn/build/"
            touch "$vendor/onednn/qualified"
        else missing=1; fi
    fi
fi
if [[ $(uname -m) == x86_64 ]] && ! grep -Eq '^(qualified:|skipped:)' "$vendor/tblis-status.txt" 2>/dev/null; then
    # Optional only. Its own script writes a native Haswell eligibility record.
    bash "$dir/tblis/build.sh" "$vendor" > "$raw/peer-restoration/tblis.log" 2>&1 || tail -12 "$raw/peer-restoration/tblis.log" >&2
fi
onednn_available || missing=1
return_code=$missing
if [[ $return_code == 0 ]]; then echo 'All required vendor installations available.'; fi
exit "$return_code"
