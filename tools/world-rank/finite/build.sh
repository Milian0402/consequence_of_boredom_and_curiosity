#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: build.sh OUTDIR STANDALONE_COB_LIBRARY' >&2; exit 2; fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
out=$(cd "$1" && pwd)
leaf=$2
python3 "$here/verify.py" > "$out/hk233-certificate.txt"
case $(uname -s) in
    Linux)
        cp "$leaf" "$out/hk233-leaf.so"
        "${CC:-cc}" -O3 -std=c11 -Wall -Wextra -Werror -fPIC -shared \
            "$here/adapter.c" -L"$out" -l:hk233-leaf.so '-Wl,-rpath,$ORIGIN' -Wl,-z,defs \
            -o "$out/hk233.so"
        plugin=$out/hk233.so ;;
    Darwin)
        cp "$leaf" "$out/hk233-leaf.dylib"
        install_name_tool -id @loader_path/hk233-leaf.dylib "$out/hk233-leaf.dylib"
        codesign --force --sign - "$out/hk233-leaf.dylib"
        "${CC:-cc}" -O3 -std=c11 -Wall -Wextra -Werror -fPIC -dynamiclib \
            "$here/adapter.c" "$out/hk233-leaf.dylib" -o "$out/hk233.dylib"
        plugin=$out/hk233.dylib ;;
    *) echo 'Unsupported host' >&2; exit 2 ;;
esac
python3 "$here/smoke.py" "$plugin" > "$out/hk233-smoke.txt"
cat "$out/hk233-certificate.txt" "$out/hk233-smoke.txt"
