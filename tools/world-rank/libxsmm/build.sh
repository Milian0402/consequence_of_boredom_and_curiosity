#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?usage: build.sh VENDOR_OUTPUT_DIRECTORY}" && cd "$1" && pwd)
tag=2.1.0
sha=7944bf36cf847c846b3fa0eb194789295e00b624
cc=${CC:-cc}
tmp=$(mktemp -d "${RUNNER_TEMP:-/tmp}/libxsmm-rank.XXXXXX")
src=${LIBXSMM_SOURCE:-$tmp/source}
if [[ ! -d "$src/.git" ]]; then
    git clone --quiet --depth 1 --branch "$tag" https://github.com/libxsmm/libxsmm.git "$src"
fi
test "$(git -C "$src" rev-parse HEAD)" = "$sha"
case "$(uname -s)" in
    Linux) flags=(-fPIC -shared '-Wl,--exclude-libs,ALL' '-Wl,-z,defs' -ldl -lm -lpthread); jobs=$(nproc) ;;
    Darwin) flags=(-fPIC -dynamiclib -lm -lpthread); jobs=2 ;;
    *) echo "Unsupported OS for LIBXSMM comparator" >&2; exit 1 ;;
esac
make -C "$src" -j"${LIBXSMM_BUILD_JOBS:-$jobs}" CC="$cc" FORTRAN=0 BUILD=0 OMP=0 clib > "$tmp/build.log" 2>&1 || {
    tail -100 "$tmp/build.log" >&2
    exit 1
}
"$cc" -O3 -std=c11 -Wall -Wextra -Werror -I"$src/include" "$here/adapter.c" \
      "$src/lib/libxsmm.a" "${flags[@]}" -o "$tmp/libxsmm.so"
python3 "$here/smoke.py" "$tmp/libxsmm.so" > "$tmp/settings.txt" 2>&1 || {
    cat "$tmp/settings.txt" >&2
    exit 1
}
{ printf 'Upstream LIBXSMM %s %s\n' "$tag" "$sha"; cat "$tmp/settings.txt"; } > "$out/libxsmm-build.txt"
cat "$out/libxsmm-build.txt"
mv "$tmp/libxsmm.so" "$out/libxsmm.so"


# Explicit public JIT dispatch over the wider supplementary small panel.
"$cc" -O3 -std=c11 -Wall -Wextra -Werror -DRANK_LIBXSMM_MAX_MNK=2097152 \
      -I"$src/include" "$here/adapter.c" "$src/lib/libxsmm.a" "${flags[@]}" -o "$out/libxsmm-wide.so"
python3 "$here/smoke.py" "$out/libxsmm-wide.so"
