#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?usage: build.sh VENDOR_OUTPUT_DIRECTORY}" && cd "$1" && pwd)
test "$(uname -s)" = Linux
tag=v0.3.34
revision=e0166008be8e466242aa76b2ff75ce3f0fbf574a
tmp=$(mktemp -d "${RUNNER_TEMP:-/tmp}/openblas-rank.XXXXXX")
src=$tmp/source
prefix=$tmp/install
cc=${CC:-cc}
git clone --quiet --depth 1 --branch "$tag" https://github.com/OpenMathLib/OpenBLAS.git "$src"
test "$(git -C "$src" rev-parse HEAD)" = "$revision"

# Compile for the current runner's CPU. Building one core and the two real
# precisions keeps the build small and includes the double-precision oracle.
opts=(CC="$cc" BINARY=64 USE_THREAD=0 USE_LOCKING=1 NO_SHARED=1
      NO_LAPACK=1 NO_LAPACKE=1 NOFORTRAN=1
      BUILD_SINGLE=1 BUILD_DOUBLE=1 BUILD_COMPLEX=0 BUILD_COMPLEX16=0)
make -C "$src" -j"$(nproc)" "${opts[@]}" libs > "$tmp/build.log" 2>&1 || {
    tail -100 "$tmp/build.log" >&2
    exit 1
}
make -C "$src" "${opts[@]}" PREFIX="$prefix" install
"$cc" -O3 -std=c11 -fPIC -shared -I"$prefix/include" "$here/../openblas.c" \
      "$prefix/lib/libopenblas.a" -Wl,--exclude-libs,ALL -Wl,-z,defs -lm -lpthread -o "$tmp/openblas.so"
# Check the ABI and capture the actual single-thread kernel before publishing.
python3 - "$tmp/openblas.so" > "$tmp/settings.txt" 2>&1 <<'PY'
import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1])
assert lib.rank_vendor_init() == 0
for symbol, scalar in (("rank_vendor_sgemm", ctypes.c_float),
                       ("rank_vendor_dgemm", ctypes.c_double)):
    gemm = getattr(lib, symbol)
    ptr, dim = ctypes.POINTER(scalar), ctypes.c_int
    gemm.argtypes = [dim, dim, dim, ptr, dim, ptr, dim, ptr, dim]
    gemm.restype = None
    a = (scalar * 3)(2, 3, 17)
    b = (scalar * 4)(5, 23, 7, 29)
    c = (scalar * 2)(0, 31)
    gemm(1, 1, 2, a, 3, b, 2, c, 2)
    assert list(c) == [31, 31], (symbol, list(c))
PY
{ printf 'Upstream OpenBLAS %s %s\n' "$tag" "$revision"; cat "$tmp/settings.txt"; } > "$out/openblas-build.txt"
cat "$out/openblas-build.txt"
mv "$tmp/openblas.so" "$out/openblas.so"
if [[ $(cat "$here/../phase.txt") =~ ^(tile-parent|leaf-wave|aligned-core|hk-forms|release-replay)$ ]]; then
    printf '%s\n' "$prefix" > "$out/openblas-prefix.txt"
fi
