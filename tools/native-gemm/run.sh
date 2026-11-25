#!/bin/sh
set -eu
base=5e40c1c4eb3f0f731a089f7215f50be0e045e030
expected_arch=${1:?supply expected uname -m}
out=${2:?supply output directory}
test "$(uname -s)" = Linux
test "$(uname -m)" = "$expected_arch"
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
export OPENBLAS_NUM_THREADS=1 OPENBLAS_DEFAULT_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
git show "$base:src/gemm.c" > "$stage/base.c"
test -n "$(git diff "$base" HEAD -- src/gemm.c)"
case "$expected_arch" in
    x86_64) tune=-march=native; native_path=native-x86-dispatch; native_flags= ;;
    aarch64) tune=-mcpu=native; native_path=native-arm-neon; native_flags=-DCOB_USE_NEON=1 ;;
    *) exit 2 ;;
esac
FLAGS="-O3 -std=c11 $tune"
# Ubuntu installs backend-specific metadata in this multiarch subdirectory.
openblas_pc="/usr/lib/$(gcc -print-multiarch)/openblas-pthread/pkgconfig"
if [ -d "$openblas_pc" ]; then
    export PKG_CONFIG_PATH="$openblas_pc${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi
{
    printf 'base=%s\ncandidate=%s\narch=%s\nflags=%s\n' "$base" "$(git rev-parse HEAD)" "$expected_arch" "$FLAGS"
    printf 'base_source=%s\ncandidate_source=%s\n' "$(git hash-object "$stage/base.c")" "$(git hash-object src/gemm.c)"
    uname -a
    lscpu
    gcc --version | head -1
    pkg-config --modversion openblas
    dpkg-query -W 'libopenblas*'
    printf 'linux affinity before pin: '
    taskset -pc $$
} > "$out/host.txt"
echo 'mode,m,n,k,block,seed,pair,calls,competitor,first,competitor_seconds,candidate_seconds,candidate_speedup' > "$out/pairs.csv"
# Rename all exported symbols on the baseline so both are resident in one process.
renames=""
for fn in sgemm_rowmajor sgemm_ref_rowmajor sgemm_pack_a sgemm_free_packed_a sgemm_pack_b sgemm_free_packed_b sgemm_rowmajor_packed_b sgemm_rowmajor_packed_ab; do
    renames="$renames -Dcob_$fn=baseline_$fn"
done
candidate_flag=-DCOB_ENABLE_PORTABLE_STRASSEN=1
grep -q COB_ENABLE_PORTABLE_STRASSEN src/gemm.c
# Verify the seven-product path and allocation-failure fallbacks at small sizes.
gcc $FLAGS $native_flags $candidate_flag -DCOB_PORTABLE_STRASSEN_MIN_DIM=128 \
    -Iinclude tests/test_portable_strassen.c -lm -o "$stage/check-strassen"
"$stage/check-strassen" > "$out/forced-route-tests.txt"
# shellcheck disable=SC2086
gcc $FLAGS $native_flags -Iinclude $renames -c "$stage/base.c" -o "$stage/base.o"
# shellcheck disable=SC2086
gcc $FLAGS $native_flags $candidate_flag -Iinclude -c src/gemm.c -o "$stage/candidate.o"
# shellcheck disable=SC2046
gcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude $(pkg-config --cflags openblas) \
    tools/native-gemm/paired.c "$stage/base.o" "$stage/candidate.o" $(pkg-config --libs openblas) -lm -o "$stage/paired"
printf 'mode=%s native_flags=%s candidate_only=%s\n' "$native_path" "$native_flags" "$candidate_flag" >> "$out/host.txt"
ldd "$stage/paired" >> "$out/host.txt"
failed=0
for block in 1 2 cancel; do
    "$stage/paired" "$native_path" "$block" >> "$out/pairs.csv" 2>> "$out/accuracy-and-affinity.log" || failed=1
done
python3 tools/native-gemm/summary.py "$out/pairs.csv" "$out/accuracy-and-affinity.log" "$expected_arch" > "$out/summary.md" || failed=1
(
    cd "$out"
    sha256sum host.txt pairs.csv accuracy-and-affinity.log summary.md > SHA256SUMS
)
exit "$failed"
