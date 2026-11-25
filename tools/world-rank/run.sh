#!/usr/bin/env bash
set -euo pipefail
rank_dir=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?usage: run.sh OUTDIR VENDOR_DIR}" && cd "$1" && pwd)
vendors=$(cd "${2:?missing vendor directory}" && pwd)
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/gemm-rank.XXXXXX")
phase=$(cat "$rank_dir/phase.txt")
case "$phase" in discovery|confirm|replicate|short) ;; *) exit 2 ;; esac
test "$(uname -s)" = Linux
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
bash "$rank_dir/build.sh" "$build"
cp "$build/build.txt" "$out/"
{ uname -a; lscpu; gcc --version; taskset -pc $$; git rev-parse HEAD; } > "$out/host.txt"
for p in "$vendors"/*.txt; do if test -f "$p"; then cp "$p" "$out/"; fi; done
if test ! -f "$vendors/openblas.so"; then
    echo 'Pinned upstream OpenBLAS missing: using the installed OpenBLAS for the FP64 oracle and fallback peer.' >&2
    cc -O3 -std=c11 -fPIC -shared $(pkg-config --cflags openblas) "$rank_dir/openblas.c" $(pkg-config --libs openblas) -o "$vendors/openblas.so"
    printf 'Fallback system OpenBLAS %s\n' "$(pkg-config --modversion openblas)" > "$out/openblas-fallback.txt"
fi
peers=()
for name in aocl mkl armpl; do
    if test -f "$vendors/$name.so"; then peers+=(--vendor "$name=$vendors/$name.so"); fi
done
if test -f "$vendors/libxsmm.so"; then
    peers+=(--small-vendor "libxsmm=$vendors/libxsmm.so")
fi
if [[ "$phase" == replicate || "$phase" == short ]]; then
    peers+=(--candidate "cob-first-confirmation=direct:$build/previous.so")
fi
if [[ -f "$build/unfused.so" ]]; then
    peers+=(--candidate "cob-unfused=direct:$build/unfused.so")
fi
python3 "$rank_dir/study.py" "$build/worker" "$rank_dir/$phase.csv" "$out" "$vendors/openblas.so" \
    --phase "$phase" --stress "${peers[@]}" --candidate "cob-challenger=direct:$build/challenger.so"
