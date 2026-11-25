#!/usr/bin/env bash
# Build only the native single-thread Winograd API from the pinned FastMMW source.
set -euo pipefail
if [[ $(uname -s) != Linux || $# -lt 1 || $# -gt 2 ]]; then
    echo 'usage on Linux: build.sh OUTDIR [PINNED_SOURCE]' >&2
    exit 2
fi
revision=96949ce923a9ac10a13f98f76e88c76b1e756f52
here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
archive_sha=9747b6e121880e33f9c078f1743f7ef36987928da5a8d609c1d7d2d3ce62e1e8
if [[ $# -eq 2 && -d $2 ]]; then
    src=$(cd "$2" && pwd)
    test "$(git -c safe.directory="$src" -C "$src" rev-parse HEAD)" = "$revision"
    git -c safe.directory="$src" -C "$src" diff --quiet HEAD -- Add Matrices Mul PThread Error Sort
else
    if [[ $# -eq 2 ]]; then archive=$2; else
        archive="$out/FastMMW.tar.gz"
        curl --proto '=https' --fail --silent --show-error --location --retry 3 --connect-timeout 30 --max-time 180 \
            "https://codeload.github.com/paolodalberto/FastMMW/tar.gz/$revision" -o "$archive"
    fi
    printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum --check --status
    src=$(mktemp -d "$out/source.XXXXXX")
    tar -xzf "$archive" --strip-components=1 -C "$src"
fi
cc=${CC:-gcc}
leaf=${FASTMMW_LEAF:-768}
[[ $leaf =~ ^[1-9][0-9]{0,3}$ && $leaf -ge 64 ]] || { echo 'invalid leaf cutoff' >&2; exit 2; }
includes=()
for name in Add Matrices Mul PThread Error Sort; do includes+=("-I$src/$name"); done
# Upstream's architecture switch includes the scalar and unrolled serial
# helpers alongside the default pthread wrappers. Select its serial kernels.
mkdir -p "$out/config"
cat > "$out/config/architecture.h" <<'CONFIG'
#include_next <architecture.h>
#undef s_add
#undef s_sub
#undef s_add_t
#undef s_sub_t
#define s_add add_amd
#define s_sub sub_amd
#define s_add_t add_amd_t
#define s_sub_t sub_amd_t
CONFIG
flags=(-std=gnu99 -O3 -fPIC -fcommon -fvisibility=hidden -ffunction-sections -fdata-sections
    -D_GNU_SOURCE -DSINGLE_PRECISION=1 -DLIBRARY_PACKAGE=1 -DGOTO_BLAS=1
    -DAUTOMATIC=1 "-DLEAF=$leaf" -I"$out/config" "${includes[@]}" -Wno-implicit-function-declaration)
case $(uname -m) in aarch64) flags+=(-mcpu=native);; x86_64) flags+=(-march=native);; *) exit 2;; esac
for part in Add/mat-addkernels Mul/mat-mulkernels; do
    "$cc" "${flags[@]}" -c "$src/$part.c" -o "$out/${part##*/}.o" > "$out/${part##*/}.log" 2>&1 || {
        tail -90 "$out/${part##*/}.log" >&2; exit 1;
    }
done
if [[ -n ${OPENBLAS_PREFIX:-} ]]; then
    blas_cflags=("-I$OPENBLAS_PREFIX/include")
    blas_libs=("$OPENBLAS_PREFIX/lib/libopenblas.a" -lpthread -lm)
else
    read -r -a blas_cflags <<< "$(pkg-config --cflags openblas)"
    read -r -a blas_libs <<< "$(pkg-config --libs openblas)"
fi
# Drop unreferenced upstream routines (including its optional parallel APIs).
# Only the single-thread Winograd entry and its actual dependency graph enter
# the plugin. A second binary counts calls to the real Fortran BLAS symbol.
for variant in stock stock-trace padded padded-trace; do
    options=()
    if [[ $variant == padded* ]]; then options+=(-DFASTMMW_PAD=1); fi
    if [[ $variant == *-trace ]]; then options+=(-DFASTMMW_TRACE=1 "-Wl,--wrap=sgemm_"); fi
    "$cc" -std=gnu99 -O3 -fPIC -shared -DSINGLE_PRECISION=1 -DLIBRARY_PACKAGE=1 "-DLEAF=$leaf" \
        "${options[@]}" "${blas_cflags[@]}" "${includes[@]}" "$here/adapter.c" \
        "$out/mat-mulkernels.o" "$out/mat-addkernels.o" -Wl,--gc-sections,-z,defs,-Bsymbolic-functions,--exclude-libs,ALL \
        "${blas_libs[@]}" -lm -o "$out/fastmmw-$variant.so" || exit 1
    nm -D "$out/fastmmw-$variant.so" > "$out/$variant-symbols.txt"
    if grep -E ' (pthread_create|GOMP_parallel)(@|$)' "$out/$variant-symbols.txt"; then
        echo 'Unexpected parallel kernel retained' >&2; exit 1
    fi
done
{
    printf 'source=https://github.com/paolodalberto/FastMMW\nrevision=%s\nlicense=LGPL-3.0-or-later\ncutoff=%s\nentry=s_wm\n' "$revision" "$leaf"
    printf 'archive_sha256=%s\n' "$archive_sha"
    if [[ -n ${OPENBLAS_PREFIX:-} ]]; then sha256sum "$OPENBLAS_PREFIX/lib/libopenblas.a"; fi
    "$cc" --version
    uname -a
    ldd "$out/fastmmw-stock.so"
    sha256sum "$out"/fastmmw-{stock,stock-trace,padded,padded-trace}.so "$src/COPYING.LESSER"
} > "$out/build-info.txt"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$here/smoke.py" "$out" "$leaf" 2>&1 | tee "$out/smoke.log"
