#!/usr/bin/env bash
# Time the previous promoted route, independent BLAS libraries, and prospective tiles.
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$dir/../.." && pwd)
out=$(mkdir -p "${1:?usage: tile-study.sh OUTPUT VENDORS}/tiles" && cd "$1/tiles" && pwd)
vendor=$(cd "${2:?missing vendor directory}" && pwd)
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/rank-tiles.XXXXXX")
phase=$(cat "$dir/phase.txt")
arch=$(uname -m)
cc=${CC:-cc}
previous=b83033d199137b370c03a5a3d63365f1025162b7
case "$phase" in tile-screen|tile-confirm|tile-replay) ;; *) exit 2 ;; esac
[[ $(uname -s) == Linux ]]
common=(-std=c11 -O3 -fPIC -shared -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1)
case "$arch" in
    x86_64)
        target=(-march=native -DCOB_ENABLE_X86_MEDIUM_ROUTE=1)
        names=(openblas aocl mkl mkl-jit blasfeo)
        panel="$dir/tiles-${phase#tile-}.csv" ;;
    aarch64)
        target=(-mcpu=native -DCOB_USE_NEON=1 -DCOB_ENABLE_N2_SMALL=1)
        names=(openblas armpl blasfeo libxsmm-wide libxsmm-cached)
        panel="$dir/n2-tiles-${phase#tile-}.csv" ;;
    *) echo "Unsupported tile-study architecture $arch" >&2; exit 2 ;;
esac
if [[ $phase == tile-replay ]]; then panel="$dir/entry-replay.csv"; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
peers=()
for name in "${names[@]}"; do
    [[ -r "$vendor/$name.so" ]] || { echo "Required benchmark library $name missing" >&2; exit 1; }
    if [[ $name != openblas ]]; then peers+=(--vendor "$name=$vendor/$name.so"); fi
done
cp "$panel" "$out/panel.csv"
for info in "$vendor"/*.txt; do if [[ -f "$info" ]]; then cp "$info" "$out/"; fi; done
cd "$repo"
bash "$dir/build.sh" "$build" "$repo"
cp "$build/build.txt" "$out/"
mkdir "$build/previous"
git archive "$previous" src include tools/world-rank/cob_adapter.c tools/world-rank/vendor.h tools/world-rank/challenger-flags.txt | tar -x -C "$build/previous"
# Use the same historical base flags on both sides. Each source gets its own headers.
while IFS= read -r flag; do
    if [[ -n $flag ]]; then common+=("-D$flag"); fi
done < "$build/previous/tools/world-rank/challenger-flags.txt"
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
{
    uname -a
    lscpu
    "$cc" --version
    taskset -pc $$
    git rev-parse HEAD
} > "$out/host.txt"
{
    printf 'phase=%s\narchitecture=%s\ncpu=%s\nincumbent_revision=%s\ncandidate_revision=%s\n' "$phase" "$arch" "$cpu" "$previous" "$(git rev-parse HEAD)"
    printf 'common_flags=%s\ntarget_flags=%s\n' "${common[*]}" "${target[*]}"
    sha256sum "$out/panel.csv" "$build/worker" "$dir/study.py" "$dir/tile-variants.inc"
} > "$out/tile-build.txt"
compile_entry() {
    local base=$1 dest=$2
    shift 2
    "$cc" "${common[@]}" "${target[@]}" "$@" -I"$base/include" -I"$base/src" \
        "$base/src/gemm.c" "$base/tools/world-rank/cob_adapter.c" -o "$dest"
}
registered=()
tile_plugin() {
    local name=$1 artifact=$2
    [[ $name =~ ^[A-Za-z0-9_.-]+$ && -s $artifact ]]
    registered+=(--candidate "$name=direct:$artifact")
    printf 'candidate=%s file=%s\n' "$name" "$artifact" >> "$out/tile-build.txt"
    sha256sum "$artifact" >> "$out/tile-build.txt"
}
tile_candidate() {
    local name=$1
    shift
    compile_entry "$repo" "$build/$name.so" "$@"
    printf 'variant_flags[%s]=%s\n' "$name" "$*" >> "$out/tile-build.txt"
    tile_plugin "$name" "$build/$name.so"
}
compile_entry "$build/previous" "$build/incumbent.so"
tile_plugin cob-incumbent "$build/incumbent.so"
if [[ $phase == tile-replay ]]; then
    source "$dir/entry-replay.inc"
    run_entry_replay
    exit $?
fi
# shellcheck source=/dev/null
source "$dir/tile-variants.inc"
register_tile_variants "$arch" "$phase"
if (( ${#registered[@]} < 4 )); then echo 'No tile candidate registered' >&2; exit 1; fi
study_failed=0
run_study() {
    taskset -c "$cpu" python3 "$dir/study.py" "$@" || study_failed=1
}
run_study "$build/worker" "$out/panel.csv" "$out" "$vendor/openblas.so" \
    --phase "$phase" --rounds 3 --stress "${peers[@]}" "${registered[@]}"

finite_peers=()
for name in "${names[@]}"; do
    case "$name" in
        openblas|libxsmm-wide|libxsmm-cached) continue ;;
    esac
    finite_peers+=(--vendor "$name=$vendor/$name.so")
done
native_vendor=armpl
if [[ $arch == x86_64 ]]; then native_vendor=mkl; fi
run_finite() {
    local finite_out=$(dirname "$out")/$2
    local phase=$1
    mkdir -p "$finite_out"
    bash "$dir/finite/build.sh" "$build/finite" "$build/incumbent.so"
    cp "$build/finite/hk233-certificate.txt" "$build/finite/hk233-smoke.txt" "$out/host.txt" "$out/build.txt" "$finite_out/"
    cp "$dir/hk-${phase#tile-}.csv" "$finite_out/panel.csv"
    for info in "$vendor"/*.txt; do if [[ -f $info ]]; then cp "$info" "$finite_out/"; fi; done
    sha256sum "$build/finite/hk233.so" "$build/incumbent.so" "$build/worker" "$finite_out/panel.csv" > "$finite_out/finite-build.txt"
    local extra_finite=()
    if [[ $phase == tile-confirm ]]; then
        bash "$dir/finite/build.sh" "$build/finite-stored" "$build/incumbent.so" stored
        cp "$build/finite-stored/hk233-certificate.txt" "$finite_out/stored-certificate.txt"
        cp "$build/finite-stored/hk233-smoke.txt" "$finite_out/stored-smoke.txt"
        sha256sum "$build/finite-stored/hk233.so" >> "$finite_out/finite-build.txt"
        extra_finite+=(--candidate "hk233-stored=hk233:$build/finite-stored/hk233.so")
    fi
    run_study "$build/worker" "$finite_out/panel.csv" "$finite_out" "$vendor/openblas.so" \
        --phase "$phase" --rounds 3 --stress "${finite_peers[@]}" \
        --candidate "cob-incumbent=direct:$build/incumbent.so" --candidate "hk233=hk233:$build/finite/hk233.so" \
        --candidate "hybrid-openblas=hybrid-next:$vendor/openblas.so" --candidate "hybrid-$native_vendor=hybrid-next:$vendor/$native_vendor.so" "${extra_finite[@]}"
}
if [[ $phase == tile-confirm && $arch == x86_64 ]]; then
    run_finite tile-screen finite-screen-replay
fi
run_finite "$phase" finite

if [[ $phase == tile-confirm ]]; then
    dual_out=$(dirname "$out")/dual
    mkdir -p "$dual_out"
    "$cc" -O3 -std=c11 "${target[@]}" -fPIC -shared "$dir/dual/dual.c" -lm -o "$build/dual.so"
    "$cc" -O3 -std=c11 "$dir/dual/check.c" "$build/dual.so" -lm -o "$build/check-dual"
    "$build/check-dual" | tee "$dual_out/dual-validation.txt"
    cp "$out/host.txt" "$out/build.txt" "$dual_out/"
    for info in "$vendor"/*.txt; do if [[ -f $info ]]; then cp "$info" "$dual_out/"; fi; done
    cp "$dir/dual/panel.csv" "$dual_out/panel.csv"
    sha256sum "$build/dual.so" "$build/worker" "$build/incumbent.so" "$dual_out/panel.csv" > "$dual_out/dual-build.txt"
    run_study "$build/worker" "$dual_out/panel.csv" "$dual_out" "$vendor/openblas.so" \
        --phase "$phase" --rounds 3 --stress "${finite_peers[@]}" \
        --candidate "cob-incumbent=direct:$build/incumbent.so" --candidate "dual-kernel=direct:$build/dual.so"
fi

exit "$study_failed"
