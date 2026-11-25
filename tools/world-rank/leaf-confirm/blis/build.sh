#!/usr/bin/env bash
# Build ordinary and diagnostic single-threaded BLIS Arm SVE benchmark adapters.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=$(mkdir -p "${1:?vendor directory required}" && cd "$1" && pwd)
work=$(mkdir -p "${2:?build directory required}" && cd "$2" && pwd)
[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]]
revision=caf0db6be1202d9c83c79b51e75eceb96aa8b556
digest=065c34df94da95c23a3a857461e6a1387af709abfeba0af86b1dfb6cdc617f65
archive=${3:-$work/blis-2.1.tar.gz}
if [[ ! -s $archive ]]; then
    curl --fail --location --retry 3 --output "$archive" "https://codeload.github.com/flame/blis/tar.gz/$revision"
fi
printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
src=$(mktemp -d "$work/armsve-src.XXXXXX")
tar -xzf "$archive" --strip-components=1 -C "$src"
cc=${CC:-cc}
(
    cd "$src"
    CC="$cc" ./configure --enable-static --disable-shared --disable-threading --disable-blas \
        --disable-pba-pools --disable-sba-pools armsve
    make -j "${BLIS_BUILD_JOBS:-2}"
) > "$work/blis-upstream-build.log" 2>&1 || { tail -80 "$work/blis-upstream-build.log"; exit 1; }
lib="$src/lib/armsve/libblis.a"
kernel=bli_sgemm_armsve_asm_2vx10_unindexed
nm -g --defined-only "$lib" > "$work/blis-symbols.txt"
grep -Eq " T $kernel$" "$work/blis-symbols.txt"
flags=(-O3 -std=gnu11 -fPIC -shared -Wl,-z,defs -Wl,--exclude-libs,ALL -isystem "$src/include/armsve")
for variant in ordinary trace; do
    target="$dest/blis.so"; trace=()
    if [[ $variant == trace ]]; then
        target="$dest/blis-trace.so"; trace=(-DRANK_BLIS_TRACE=1 "-Wl,--wrap=$kernel")
    fi
    "$cc" "${flags[@]}" "${trace[@]}" "$here/adapter.c" "$lib" -lm -lpthread -o "$target"
    readelf -d "$target" > "$dest/blis-$variant-dynamic.txt"
    nm -D --defined-only "$target" > "$dest/blis-$variant-exports.txt"
    if grep -E 'NEEDED.*(mkl|openblas|armpl|gomp|libomp)' "$dest/blis-$variant-dynamic.txt"; then exit 1; fi
done
if grep -q rank_blis_snapshot "$dest/blis-ordinary-exports.txt"; then exit 1; fi
grep -q rank_blis_snapshot "$dest/blis-trace-exports.txt"
( cd "$dest"; sha256sum blis.so blis-trace.so ) > "$dest/blis-binaries.sha256"
{
    printf 'BLIS release=2.1 commit=%s source_tar_sha256=%s configuration=armsve threads=single pba_pools=0 sba_pools=0 kernel=%s\n' "$revision" "$digest" "$kernel"
    printf 'source_kernel_sha256='; sha256sum "$src/kernels/armsve/3/bli_gemm_armsve_asm_s2vx10_unindexed.c" | cut -d' ' -f1
    printf 'static_archive_sha256='; sha256sum "$lib" | cut -d' ' -f1
    "$cc" --version | head -1
} > "$dest/blis-build.txt"
python3 "$here/smoke.py" "$dest" >> "$dest/blis-build.txt" 2>&1 || { cat "$dest/blis-build.txt"; exit 1; }
