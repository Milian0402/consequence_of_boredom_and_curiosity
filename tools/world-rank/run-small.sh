#!/usr/bin/env bash
# Separate nine-case experiment for the fixed one-shot small kernel.
set -euo pipefail
rank_dir=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?usage: run-small.sh OUTDIR VENDOR_DIR [WORKER]}" && cd "$1" && pwd)
vendors=$(cd "${2:?missing vendor directory}" && pwd)
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/small-rank.XXXXXX")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
cc=${CC:-cc}
case "$(uname -sm)" in
    'Linux x86_64') flags=(-march=native); required=(aocl mkl) ;;
    'Linux aarch64') flags=(-mcpu=native); required=(armpl) ;;
    *) echo 'Small-panel measurements require native x86-64 or Arm64 Linux.' >&2; exit 2 ;;
esac
worker=${3:-$build/worker}
if [[ $# -lt 3 ]]; then bash "$rank_dir/build.sh" "$build"; fi
"$cc" -O3 -std=c11 -Wall -Wextra -Werror "${flags[@]}" -fPIC -shared \
    "$rank_dir/small-next/small.c" -o "$build/small-next.so"
cp "$rank_dir/small-panel.csv" "$out/panel.csv"
{
    uname -a
    lscpu
    "$cc" --version
    taskset -pc $$
    git rev-parse HEAD
    echo "flags=-O3 -std=c11 ${flags[*]} -fPIC -shared"
    (cd "$rank_dir/small-next" && sha256sum small.c arm.inc x86.inc)
    sha256sum "$build/small-next.so" "$worker"
} > "$out/build.txt"

# Direct comparators only. The wide explicit-JIT adapter is used only here.
# Preserve all available evidence and report any absent requested comparator.
peers=()
missing=()
for name in "${required[@]}" libxsmm-wide; do
    if [[ -f "$vendors/$name.so" ]]; then
        peers+=(--vendor "$name=$vendors/$name.so")
    else
        missing+=("$name")
    fi
done
if [[ ! -f "$vendors/openblas.so" ]]; then
    echo 'Missing FP64 oracle and direct OpenBLAS comparator.' >&2
    exit 1
fi
for info in "$vendors"/*.txt; do
    if [[ -f "$info" ]]; then cp "$info" "$out/"; fi
done
if (( ${#missing[@]} )); then
    printf '%s\n' "${missing[@]}" > "$out/missing-vendor.txt"
fi
python3 "$rank_dir/study.py" "$worker" "$out/panel.csv" "$out" "$vendors/openblas.so" \
    --phase small "${peers[@]}" --candidate "cob-small-next=direct:$build/small-next.so"
(( ${#missing[@]} == 0 ))
