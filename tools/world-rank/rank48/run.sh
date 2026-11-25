#!/usr/bin/env bash
set -euo pipefail
dir=$(cd -- "$(dirname -- "$0")" && pwd)
dest=${1:?result directory required}
mkdir -p "$dest"
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/finite-ranks.XXXXXX")
trap 'rm -rf "$build"' EXIT
arch=$(uname -m)
case "$arch" in
    x86_64) provider=mkl; tune=-march=native ;;
    aarch64) provider=openblas; tune=-mcpu=native ;;
    *) echo "Unsupported native CPU $arch" >&2; exit 1 ;;
esac
export RANK_FINITE_LEAF="$VENDOR_DIR/$provider.so" RANK_VERIFY_INPUTS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
unset RANK_MEDIUM_BATCH
cc -O2 -std=c11 "$dir/check.c" -ldl -o "$build/check"
cp "$dir/panel.csv" "$dest/panel.csv"
lscpu > "$dest/cpu.txt"
(cd "$dir" && sha256sum schedules/*.slp adapter.c generate.py run.sh panel.csv) > "$dest/source-sha256.txt"
printf 'rank,m,n,k,pad,core_calls,fringe_calls,checked,errors,inputs_unchanged,route_verified\n' > "$dest/routes.csv"
entries=()
for rank in 48 49; do
    mkdir "$build/$rank"
    python3 -B "$dir/generate.py" "$rank" "$build/$rank/schedule.h" >> "$dest/certificates.txt"
    cc -O3 "$tune" -std=c11 -fPIC -shared -I"$build/$rank" "$dir/adapter.c" -ldl -o "$build/form$rank.so"
    while IFS=, read -r family m n k pad; do
        [[ $family != family ]] || continue
        taskset -c "$RANK_STUDY_CPU" "$build/check" "$build/form$rank.so" "$rank" "$m" "$n" "$k" "$pad" >> "$dest/routes.csv"
    done < "$dir/panel.csv"
    entries+=(--candidate "form$rank=direct:$build/form$rank.so")
done
if [[ $arch == x86_64 ]]; then
    entries+=(--vendor "mkl=$VENDOR_DIR/mkl.so" --vendor "mkl-jit=$VENDOR_DIR/mkl-jit.so" --vendor "aocl=$VENDOR_DIR/aocl.so")
else
    entries+=(--vendor "armpl=$VENDOR_DIR/armpl.so")
fi
printf '%s\n' "$provider" > "$dest/leaf.txt"
set +e
taskset -c "$RANK_STUDY_CPU" python3 -B "$dir/study_runner.py" "$RANK_STUDY_WORKER" "$dir/panel.csv" "$dest" "$VENDOR_DIR/openblas.so" \
    --phase finite-ranks --rounds 3 --stress-all "${entries[@]}"
study_status=$?
set -e
python3 -B "$dir/score.py" "$dest" "$study_status"
