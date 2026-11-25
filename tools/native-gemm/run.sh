#!/bin/sh
set -eu
base=d2a59e8d2785961a4365e6f9ac982f19e3c28f61
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
echo 'mode,m,n,k,block,pair,calls,competitor,first,competitor_seconds,candidate_seconds,candidate_speedup' > "$out/pairs.csv"
# Rename all exported symbols on the baseline so both are resident in one process.
renames=""
for fn in sgemm_rowmajor sgemm_ref_rowmajor sgemm_pack_a sgemm_free_packed_a sgemm_pack_b sgemm_free_packed_b sgemm_rowmajor_packed_b sgemm_rowmajor_packed_ab; do
    renames="$renames -Dcob_$fn=baseline_$fn"
done
modes="$native_path plain-c-autovec"
if [ "$expected_arch" = x86_64 ]; then
    cat > "$stage/isa.c" <<'ISA'
#include <stdio.h>
int main(void) {
    int fma=!!__builtin_cpu_supports("fma"), avx2=!!__builtin_cpu_supports("avx2");
    printf("runtime_fma=%d runtime_avx2=%d runtime_avx512f=%d\n", fma, avx2, !!__builtin_cpu_supports("avx512f"));
    return fma && avx2 ? 0 : 1;
}
ISA
    gcc -O2 "$stage/isa.c" -o "$stage/isa"
    if "$stage/isa" >> "$out/host.txt"; then modes="$modes forced-avx2-smoke"; fi
fi
for mode in $modes; do
    feature=
    case "$mode" in
        "$native_path") feature=$native_flags ;;
        plain-c-autovec) feature=-DCOB_DISABLE_X86_SIMD=1 ;;
        forced-avx2-smoke)
            grep -q 'COB_DISABLE_X86_AVX512' src/gemm.c || { echo "AVX2-only switch missing"; exit 1; }
            feature="-DCOB_DISABLE_X86_AVX512=1 -mno-avx512f"
            ;;
    esac
    # shellcheck disable=SC2086
    gcc $FLAGS $feature -Iinclude $renames -c "$stage/base.c" -o "$stage/base.o"
    # shellcheck disable=SC2086
    gcc $FLAGS $feature -Iinclude -c src/gemm.c -o "$stage/candidate.o"
    # shellcheck disable=SC2046
    gcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude $(pkg-config --cflags openblas) \
        tools/native-gemm/paired.c "$stage/base.o" "$stage/candidate.o" $(pkg-config --libs openblas) -lm -o "$stage/paired"
    printf 'mode=%s feature=%s flags=%s compiled_candidate_blob=%s\n' "$mode" "$feature" "$FLAGS" "$(git hash-object src/gemm.c)" >> "$out/host.txt"
    ldd "$stage/paired" >> "$out/host.txt"
    if [ "$mode" = forced-avx2-smoke ]; then
        "$stage/paired" "$mode" 1 >> "$out/pairs.csv" 2>> "$out/accuracy-and-affinity.log"
        continue
    fi
    for block in 1 2; do
        "$stage/paired" "$mode" "$block" >> "$out/pairs.csv" 2>> "$out/accuracy-and-affinity.log"
    done
done
python3 tools/native-gemm/summary.py "$out/pairs.csv" "$expected_arch" > "$out/summary.md"
(
    cd "$out"
    sha256sum host.txt pairs.csv accuracy-and-affinity.log summary.md > SHA256SUMS
)
