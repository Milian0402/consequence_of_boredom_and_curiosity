#!/usr/bin/env bash
# Build a whole-call BF16 bridge against the previously authenticated SEQ runtime.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
sdk=$(cd "${1:?oneDNN installed prefix}" && pwd)
out=$(mkdir -p "${2:?output directory}" && cd "$2" && pwd)
[[ $(uname -s) == Linux ]]
case $(uname -m) in
  x86_64) digest=9a6c34dd237ed50d5c6b26522ffa2f4f5135e4899a9fef03ea7ea20cba1e1a0a ;;
  aarch64) digest=e8ec604abd0edb3c77656329bb76f64a5809d499ae4001c8182a9cb769e3287d ;;
  *) echo "Unsupported architecture" >&2;exit 1 ;;
esac
lib=$sdk/lib/libdnnl.so
printf '%s  %s\n' "$digest" "$lib" | sha256sum -c --status || { echo 'Pinned oneDNN runtime hash mismatch' >&2;exit 1; }
cxx=${CXX:-c++}
opts=(-std=c++17 -O3 -fno-fast-math -Wall -Wextra -Werror)
"$cxx" "${opts[@]}" -fPIC -I"$sdk/include" -c "$here/backend.cpp" -o "$out/backend.o"
"$cxx" -shared "$out/backend.o" -L"$sdk/lib" -Wl,-z,defs "-Wl,-rpath,$sdk/lib" -ldnnl -o "$out/libbf16_backend.so"
resolved=$(ldd "$out/libbf16_backend.so" | awk '$1=="libdnnl.so.3" && $2=="=>" {print $3}')
[[ -n $resolved && $(realpath -e "$resolved") == "$(realpath -e "$lib")" ]] || { echo 'Unexpected mapped BF16 runtime' >&2;exit 1; }
"$cxx" "${opts[@]}" "$here/probe.cpp" -L"$out" '-Wl,-rpath,$ORIGIN' -lbf16_backend -o "$out/bf16-probe"
{
    printf 'upstream_tag=v3.14.2\nsource_sha=4db284662a37cf46401655b5ee99391d9de52c00\nsource_tar_sha256=55e319d174c6c3884b66e7d643d9579a96d73b20159580dc3b1f89de4875da9c\nlibrary_version=3.14.1\ncpu_runtime=SEQ\ngpu_runtime=NONE\nlibdnnl_sha256=%s\n' "$digest"
    (cd "$here" && sha256sum backend.h backend.cpp probe.cpp build.sh)
    (cd "$out" && sha256sum backend.o libbf16_backend.so bf16-probe)
} > "$out/bf16-build.txt"
