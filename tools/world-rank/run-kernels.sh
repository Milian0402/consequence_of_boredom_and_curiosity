#!/usr/bin/env bash
# Compare optimization candidates to the preceding native implementation and all eligible peers.
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?usage: run-kernels.sh OUTPUT VENDORS}" && cd "$1" && pwd)
vendor=$(cd "${2:?missing vendor directory}" && pwd)
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/rank-kernels.XXXXXX")
phase=$(cat "$dir/phase.txt")
case "$phase" in pilot|heldout) ;; *) exit 2 ;; esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
bash "$dir/build.sh" "$build"
cc=${CC:-cc}
common=(-std=c11 -O3 -fPIC -shared -Iinclude -Isrc -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1)
while read -r definition; do
    if [[ -n "$definition" ]]; then common+=("-D$definition"); fi
done < "$dir/challenger-flags.txt"
peers=()
case "$(uname -m)" in
    x86_64) common+=(-march=native); peers+=(--vendor "aocl=$vendor/aocl.so" --vendor "mkl=$vendor/mkl.so");;
    aarch64) common+=(-mcpu=native -DCOB_USE_NEON=1); peers+=(--vendor "armpl=$vendor/armpl.so");;
    *) exit 2;;
esac
cob_build() {
    local name=$1
    shift
    "$cc" "${common[@]}" "$@" src/gemm.c "$dir/cob_adapter.c" -o "$build/$name.so"
}
previous=22bbdbd919f56d2931dc5cbbba30145356b661d8
mkdir -p "$build/prior"
git archive "$previous" src include | tar -x -C "$build/prior"
"$cc" "${common[@]}" "$build/prior/src/gemm.c" "$dir/cob_adapter.c" -o "$build/prior.so"
capture() {
    local panel=$1
    shift
    mkdir -p "$out/$panel"
    cp "$dir/$panel-$phase.csv" "$out/$panel/panel.csv"
    cp "$build/build.txt" "$out/$panel/"
    { uname -a; lscpu; "$cc" --version; taskset -pc $$; git rev-parse HEAD; } > "$out/$panel/host.txt"
    for info in "$vendor"/*.txt; do if [[ -f "$info" ]]; then cp "$info" "$out/$panel/"; fi; done
    python3 "$dir/study.py" "$build/worker" "$out/$panel/panel.csv" "$out/$panel" "$vendor/openblas.so" \
        --phase "$phase" --stress "${peers[@]}" --candidate "cob-incumbent=direct:$build/prior.so" "$@"
}
medium=()
if [[ "$phase" == pilot ]]; then
    for depth in 128 256 512; do
        cob_build "panel-$depth" -DCOB_ENABLE_CACHE_PANEL=1 "-DCOB_PANEL_DEPTH=$depth"
        medium+=(--candidate "panel-$depth=direct:$build/panel-$depth.so")
    done
    if [[ $(uname -m) == x86_64 ]]; then
        cob_build wide -DCOB_ENABLE_WIDE_X86=1
        medium+=(--candidate "wide=direct:$build/wide.so")
        cob_build panel-wide -DCOB_ENABLE_WIDE_X86=1 -DCOB_ENABLE_CACHE_PANEL=1
        medium+=(--candidate "panel-wide=direct:$build/panel-wide.so")
        cob_build direct-a -DCOB_ENABLE_MEDIUM_DIRECT_A=1
        medium+=(--candidate "direct-a=direct:$build/direct-a.so")
        cob_build first-tile -DCOB_ENABLE_X86_FIRST_TILE_PACK=1
        medium+=(--candidate "first-tile=direct:$build/first-tile.so")
    fi
else
    medium+=(--candidate "cob-selected=direct:$build/challenger.so")
fi
capture medium "${medium[@]}"
if [[ $(uname -m) == aarch64 ]]; then
    source "$dir/run-n2-small.inc"
    run_n2_small
fi
"$cc" -std=c11 -O2 "$dir/test_hybrid_next.c" -lm -o "$build/test-hybrid"
"$build/test-hybrid" | tee "$out/hybrid-validation.txt"
recursive=()
for name in openblas aocl mkl armpl; do
    if [[ -f "$vendor/$name.so" ]]; then
        recursive+=(--candidate "previous-$name=hybrid:$vendor/$name.so"
                    --candidate "winograd-$name=hybrid-next:$vendor/$name.so")
    fi
done
capture recursive "${recursive[@]}"
