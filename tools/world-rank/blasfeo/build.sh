#!/usr/bin/env bash
# Use the upstream native kernels and ordinary, unpacked standard BLAS interface.
set -euo pipefail
if [[ $# != 1 || $(uname -s) != Linux ]]; then
    echo "Usage on Linux: build.sh OUTDIR" >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
out=$(cd "$1" && pwd)
revision=e5d1cbbb2f458acd7adb7d9e8a6522416b7e336f
digest=3e9276f686dad725049600b37459197e464fd3445a49fa7d87c690a5527de9b5
scratch=$(mktemp -d "${RUNNER_TEMP:-/tmp}/blasfeo-rank.XXXXXX")
archive=${BLASFEO_ARCHIVE:-$scratch/source.tar.gz}
src=$scratch/source
mkdir "$src"
if [[ -z ${BLASFEO_ARCHIVE:-} ]]; then
    curl --fail --location --silent --show-error --retry 3 \
        "https://codeload.github.com/giaf/blasfeo/tar.gz/$revision" --output "$archive"
fi
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
tar -xzf "$archive" -C "$src" --strip-components=1
case $(uname -m) in
    x86_64)
        # At this revision the standard column-major SGEMM on SKYLAKE_X
        # resolves only generic 4x4 kernels. HASWELL has the native 24x4
        # AVX2/FMA kernels for this API, including on AVX-512-capable hosts.
        if ! grep -qw avx2 /proc/cpuinfo || ! grep -qw fma /proc/cpuinfo; then
            echo "Native x86 BLASFEO SGEMM requires AVX2 and FMA" >&2
            exit 2
        fi
        target=X64_INTEL_HASWELL ;;
    # BLASFEO provides this Armv8-A NEON target for A76 and newer, including
    # Neoverse-N2. The pinned version has no dedicated N2/SVE microkernel.
    aarch64) target=ARMV8A_ARM_CORTEX_A76 ;;
    *) echo "Unsupported BLASFEO host: $(uname -m)" >&2; exit 2 ;;
esac
case ${RANK_BLASFEO_ROW_SLAB:-0} in
    0) row_slab=0; name=blasfeo ;;
    48) row_slab=48; name=blasfeo-row48 ;;
    *) echo 'RANK_BLASFEO_ROW_SLAB must be 0 or 48' >&2; exit 2 ;;
esac
cc=${CC:-cc}
make -C "$src" -j"${BLASFEO_BUILD_JOBS:-$(nproc)}" static_library \
    CC="$cc" TARGET="$target" LA=HIGH_PERFORMANCE SP_ROUTINES=1 DP_ROUTINES=0 \
    BLAS_API=1 FORTRAN_BLAS_API=0 EXTERNAL_BLAS=0 FALLBACK_TO_EXTERNAL_BLAS=0 \
    > "$out/$name-compile.log" 2>&1 || { tail -80 "$out/$name-compile.log" >&2; exit 1; }
link_plugin() {
    "$cc" -O3 -std=c11 -Wall -Wextra -Werror -fPIC -shared -I"$src/include" \
        "-DRANK_BLASFEO_TARGET=\"$target\"" "-DRANK_CLEAR_OUTPUT=$1" "-DRANK_ROW_SLAB=$row_slab" \
        "$here/adapter.c" "$src/lib/libblasfeo.a" -Wl,--exclude-libs,ALL \
        -Wl,-Bsymbolic-functions -Wl,-z,defs -lm -o "$scratch/blasfeo.so"
}
{
    printf 'source=https://github.com/giaf/blasfeo revision=%s tar_sha256=%s target=%s\n' "$revision" "$digest" "$target"
    "$cc" --version
} > "$out/$name-build.txt"
link_plugin 0
if python3 "$here/smoke.py" "$scratch/blasfeo.so" >> "$out/$name-build.txt" 2>&1; then
    echo 'Stock native beta-zero path passed.' >> "$out/$name-build.txt"
else
    status=$?
    if [[ $status != 3 ]]; then cat "$out/$name-build.txt" >&2; exit "$status"; fi
    echo 'Stock kernel retained NaN old C. Retesting with a timed logical output clear.' >> "$out/$name-build.txt"
    link_plugin 1
    python3 "$here/smoke.py" "$scratch/blasfeo.so" >> "$out/$name-build.txt" 2>&1 || {
        cat "$out/$name-build.txt" >&2
        exit 1
    }
fi
sha256sum "$scratch/blasfeo.so" >> "$out/$name-build.txt"
cat "$out/$name-build.txt"
mv "$scratch/blasfeo.so" "$out/$name.so"
