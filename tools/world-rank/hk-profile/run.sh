#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo 'Usage: run.sh REPO OUT_DIRECTORY [CPU] [smoke]' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
revision=0cff6ec089a43e65eeb3bfefd12a1fc68f205be4
scratch=$(mktemp -d "${TMPDIR:-/tmp}/hk-profile.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
cc=${CC:-cc}
smoke=()
if [[ ${4:-} == smoke ]]; then smoke=(smoke); fi
if [[ $(uname -s) != Linux && ${#smoke[@]} == 0 ]]; then
    echo 'Native Linux required for performance evidence' >&2
    exit 2
fi
test ! -e "$out/raw.csv" || { echo 'Refusing to overwrite raw timing evidence' >&2; exit 1; }
mkdir "$scratch"/{control,profile}
git -C "$repo" archive "$revision" src include tools/world-rank/finite/hk233.inc tools/world-rank/challenger-flags.txt | tar xf - -C "$scratch/control"
cp -R "$scratch/control/." "$scratch/profile"
flags=(-O3 -std=c11 -fPIC -shared -D_POSIX_C_SOURCE=200809L -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_HK15=1)
while IFS= read -r flag; do [[ -z $flag ]] || flags+=("-D$flag"); done < "$scratch/control/tools/world-rank/challenger-flags.txt"
case "$(uname -m)" in
    aarch64|arm64) flags+=(-mcpu=native -DCOB_USE_NEON=1 -DCOB_ENABLE_N2_SMALL=1) ;;
    x86_64) flags+=(-march=native -DCOB_ENABLE_X86_MEDIUM_ROUTE=1) ;;
    *) exit 2 ;;
esac
if [[ $(uname -s) == Darwin ]]; then flags+=(-DCOB_DISABLE_APPLE_AMX=1 -DCOB_DISABLE_APPLE_SME=1); fi
{
    printf 'source=%s\nflags=%s\n' "$revision" "${flags[*]}"
    uname -a
    "$cc" --version
    if command -v lscpu >/dev/null; then lscpu; fi
} > "$out/build.txt"
for kind in control profile; do
    python3 "$here/instrument.py" "$scratch/$kind" "$kind" "${smoke[@]}"
    extra=()
    if [[ $kind == profile ]]; then extra=(-include "$here/clock.h"); fi
    for route in legacy1024 parent960; do
        route_flags=()
        if [[ $route == parent960 ]]; then route_flags=(-DCOB_ENABLE_HK15_PARENT_960=1); fi
        "$cc" "${flags[@]}" "${extra[@]}" "${route_flags[@]}" -I"$scratch/$kind/include" -I"$scratch/$kind/src" \
            "$scratch/$kind/src/gemm.c" -o "$scratch/$route-$kind.so"
        if command -v sha256sum >/dev/null; then
            sha256sum "$scratch/$route-$kind.so" >> "$out/build.txt"
        else
            shasum -a 256 "$scratch/$route-$kind.so" >> "$out/build.txt"
        fi
    done
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
if [[ ${#smoke[@]} == 0 ]]; then
    cpu=${3:-$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')}
    taskset -c "$cpu" python3 "$here/measure.py" "$scratch" "$out"
else
    python3 "$here/measure.py" "$scratch" "$out" smoke
fi
