#!/bin/sh
set -eu
base=377c47757311d1cd3c6540c6999bf7a93ec2a683
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
phase=$(cat tools/native-gemm/phase.txt)
case "$phase" in discovery|confirm) ;; *) echo "Invalid phase: $phase" >&2; exit 2 ;; esac
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
# Rename all exported symbols on the baseline so both are resident in one process.
renames=""
for fn in sgemm_rowmajor sgemm_ref_rowmajor sgemm_pack_a sgemm_free_packed_a sgemm_pack_b sgemm_free_packed_b sgemm_rowmajor_packed_b sgemm_rowmajor_packed_ab; do
    renames="$renames -Dcob_$fn=baseline_$fn"
done
strassen_flag=-DCOB_ENABLE_PORTABLE_STRASSEN=1
grep -q COB_ENABLE_PORTABLE_STRASSEN src/gemm.c
# One numeric preprocessor setting per line. Never interpret the input as shell.
read_flags() {
    awk '
        /^[[:space:]]*(#.*)?$/ { next }
        /^(COB_ENABLE_RECTANGULAR_STRASSEN|COB_ENABLE_RECURSIVE_STRASSEN|COB_ENABLE_SKINNY_GEMM|COB_ENABLE_NATIVE_PACKING|COB_ENABLE_BILINEAR336|COB_STRASSEN_RECURSION_MIN_DIM)=[0-9]+$/ {
            print "-D" $0
            next
        }
        { print "Invalid candidate compiler definition in " FILENAME ": " $0 > "/dev/stderr"; exit 2 }
    ' "$1"
}
primary_flags=$(read_flags tools/native-gemm/candidate-flags.txt)
recursive_flags=$(read_flags tools/native-gemm/recursive-flags.txt)
bilinear_flags=$(read_flags tools/native-gemm/bilinear-flags.txt)
# shellcheck disable=SC2086
gcc $FLAGS $native_flags $strassen_flag -Iinclude $renames -c "$stage/base.c" -o "$stage/base.o"
printf 'mode=%s native_flags=%s shared_strassen_flag=%s phase=%s\n' "$native_path" "$native_flags" "$strassen_flag" "$phase" >> "$out/host.txt"

run_variant() {
    job_phase=$1
    extra=$2
    result_dir=$3
    base_obj=$4
    mkdir -p "$result_dir"
    if [ "$result_dir" != "$out" ]; then cp "$out/host.txt" "$result_dir/host.txt"; fi
    printf 'measured_phase=%s candidate_only_flags=%s paired_baseline_object=%s\n' "$job_phase" "$extra" "${base_obj##*/}" >> "$result_dir/host.txt"
    echo 'phase,mode,m,n,k,pad,block,seed,pair,calls,competitor,first,competitor_seconds,candidate_seconds,candidate_speedup' > "$result_dir/pairs.csv"
    # shellcheck disable=SC2086
    gcc $FLAGS $native_flags $strassen_flag $extra -Iinclude -c src/gemm.c -o "$stage/candidate.o" || return 2
    # shellcheck disable=SC2046
    gcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude $(pkg-config --cflags openblas) \
        tools/native-gemm/paired.c "$base_obj" "$stage/candidate.o" $(pkg-config --libs openblas) -lm -o "$stage/paired" || return 2
    ldd "$stage/paired" >> "$result_dir/host.txt"
    failed=0
    for block in 1 2 cancel; do
        "$stage/paired" "$native_path" "$job_phase" "$block" >> "$result_dir/pairs.csv" 2>> "$result_dir/accuracy-and-affinity.log" || failed=1
    done
    python3 tools/native-gemm/summary.py "$result_dir/pairs.csv" "$result_dir/accuracy-and-affinity.log" "$expected_arch" "$job_phase" > "$result_dir/summary.md" || failed=1
    (
        cd "$result_dir"
        sha256sum host.txt pairs.csv accuracy-and-affinity.log summary.md > SHA256SUMS
    )
    return "$failed"
}
main_failed=0
run_variant "$phase" "$primary_flags" "$out" "$stage/base.o" || main_failed=1
if [ "$phase" = discovery ] && { [ -n "$recursive_flags" ] || [ -n "$bilinear_flags" ]; }; then
    # Compare optional algorithms directly against the same candidate source with primary routes enabled.
    # shellcheck disable=SC2086
    gcc $FLAGS $native_flags $strassen_flag $primary_flags $renames -Iinclude -c src/gemm.c -o "$stage/primary-as-base.o"
    if [ -n "$recursive_flags" ]; then
        run_variant recursive "$primary_flags $recursive_flags" "$out/recursive" "$stage/primary-as-base.o" || main_failed=1
    fi
    if [ -n "$bilinear_flags" ]; then
        run_variant bilinear "$primary_flags $bilinear_flags" "$out/bilinear" "$stage/primary-as-base.o" || main_failed=1
    fi
fi
exit "$main_failed"
