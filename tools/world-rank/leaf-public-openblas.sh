#!/usr/bin/env bash
# Link the complete public multiply against the same pinned OpenBLAS as the direct peer.
set -euo pipefail
repo=${1:?repo}
vendor=${2:?vendor}
output=${3:?output}
read -r prefix < "$vendor/openblas-prefix.txt"
test -r "$prefix/lib/libopenblas.a"
case $(uname -m) in
    x86_64) target=(-march=native) ;;
    aarch64) target=(-mcpu=native -DCOB_USE_NEON=1) ;;
    *) exit 2 ;;
esac
study="$repo/tools/world-rank"
flags=(-std=c11 -O3 -fPIC -shared "${target[@]}" -I"$repo/include" -I"$prefix/include"
    -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_OPENBLAS_LEAF=1)
for variant in timed trace; do
    extra=()
    dest=$output
    if [[ $variant == trace ]]; then
        extra=("$study/leaf-public-openblas-trace.c" '-Wl,--wrap=cblas_sgemm')
        dest=${output%.so}.trace.so
    fi
    "${CC:-cc}" "${flags[@]}" "$repo/src/gemm.c" "$study/cob_adapter.c" "${extra[@]}" \
        "$prefix/lib/libopenblas.a" -Wl,--exclude-libs,ALL -Wl,-z,defs -lm -lpthread -o "$dest"
done

previous=0cff6ec089a43e65eeb3bfefd12a1fc68f205be4
snapshot="${output%.so}.previous-source"
mkdir -p "$snapshot"
git -C "$repo" archive "$previous" src include tools/world-rank/cob_adapter.c tools/world-rank/vendor.h |
    tar -x -C "$snapshot"
"${CC:-cc}" -I"$snapshot/include" "${flags[@]}" "$snapshot/src/gemm.c" \
    "$snapshot/tools/world-rank/cob_adapter.c" "$prefix/lib/libopenblas.a" \
    -Wl,--exclude-libs,ALL -Wl,-z,defs -lm -lpthread -o "${output%.so}.previous.so"
{
    printf 'baseline=%s\nflags=%s\n' "$previous" "${flags[*]}"
    sha256sum "$prefix/lib/libopenblas.a" "$repo/src/cob_openblas_winograd.inc" \
        "$output" "${output%.so}.previous.so" "${output%.so}.trace.so"
} > "${output%.so}-build.txt"
