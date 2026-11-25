#!/usr/bin/env bash
# Build the pinned AMD BLAS release for the native runner's CPU.
set -euo pipefail
if [[ $# != 1 || $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "Usage on Linux x86-64: build.sh OUTDIR" >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
out=$(cd -- "$1" && pwd)
source_sha=e5f75c24bdb8670ff648da7e483eb7e51b28a1db
dist="$out/aocl-source"
prefix="$out/aocl-prefix"
git clone --depth 1 --branch 6.0.0 https://github.com/amd/blis.git "$dist"
test "$(git -C "$dist" rev-parse HEAD)" = "$source_sha"
# BLIS auto compiles and runs AMD's CPUID selector on this machine. Stock
# release flags choose its host-specific microkernel and cache blocking.
export CC=gcc
(
    cd "$dist"
    ./configure --prefix="$prefix" --enable-cblas --blas-int-size=32 \
        --enable-static --enable-shared --disable-threading auto
) > "$out/aocl-configure.log" 2>&1 || { tail -100 "$out/aocl-configure.log" >&2; exit 1; }
make -C "$dist" -j"$(nproc)" install > "$out/aocl-build.log" 2>&1 || {
    tail -100 "$out/aocl-build.log" >&2
    exit 1
}
# Hide BLIS archive symbols in this one plugin. Otherwise another CBLAS in
# the benchmark's global ELF scope could accidentally serve AMD's calls.
"$CC" -std=c11 -O3 -fPIC -shared -I"$prefix/include/blis" "$here/adapter.c" \
    "$prefix/lib/libblis.a" -Wl,--exclude-libs,ALL -Wl,-Bsymbolic-functions \
    -Wl,-z,defs -lm -lpthread -o "$out/aocl.so"
(
    printf 'source_url=https://github.com/amd/blis\nrelease_tag=6.0.0\nsource_sha=%s\n' "$source_sha"
    "$CC" --version | head -1
    grep -F 'hardware detection driver returned' "$out/aocl-configure.log" || exit 1
    lscpu | grep -E 'Architecture:|Vendor ID:|Model name:'
    python3 "$here/smoke.py" "$out/aocl.so" || exit 1
    ldd "$out/aocl.so"
    sha256sum "$out/aocl.so"
) > "$out/aocl-configuration.txt" 2>&1 || { cat "$out/aocl-configuration.txt" >&2; exit 1; }
cat "$out/aocl-configuration.txt"
