#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -sm) == 'Linux aarch64' ]]
here=$(cd "$(dirname "$0")" && pwd)
vendor=$(mkdir -p "${1:?vendor directory required}" && cd "$1" && pwd)
out=$(mkdir -p "${2:?build directory required}" && cd "$2" && pwd)
sdk_sha=408cf4d5b3ea81031f915506f90977fba7549abc7dbff5929562b5767e215117
lib_sha=321fd6d02efe5d0cd1fa126f581d2addb5b36bdf7ab3c288c7b4d9c55bb5e8f0
symbol=_ZN8arm_gemm34a64_interleaved_bf16fp32_mmla_8x12EPKN11arm_compute8bfloat16ES3_Pfiii
url=https://github.com/ARM-software/ComputeLibrary/releases/download/v53.3.2/arm_compute-v53.3.2-linux-aarch64-cpu-bin.tar.gz
tarball=${3:-$vendor/acl-cpu-sdk.tar.gz}
if [[ ! -f $tarball ]]; then
    tmpfile=$(mktemp "$vendor/acl-download.XXXXXXXX")
    curl -fL --retry 3 "$url" -o "$tmpfile"
    printf '%s  %s\n' "$sdk_sha" "$tmpfile" | sha256sum -c --status
    mv "$tmpfile" "$tarball"
fi
printf '%s  %s\n' "$sdk_sha" "$tarball" | sha256sum -c --status
member=arm_compute-v53.3.2-linux-aarch64-cpu-bin/lib/armv8a-neon/libarm_compute.so
tar -xzf "$tarball" -C "$out" "$member"
lib="$out/$member"; dir=$(dirname "$lib")
printf '%s  %s\n' "$lib_sha" "$lib" | sha256sum -c --status
objdump -d --disassemble="$symbol" "$lib" > "$out/acl-kernel.txt"
grep -E '[[:space:]]bfmmla[[:space:]]' "$out/acl-kernel.txt" > "$out/acl-bfmmla-opcodes.txt"
opts=(-O3 -march=armv8-a -std=c11 -Wall -Wextra -Werror -fPIC "-DACL_PINNED_PATH=\"$lib\"")
shared=(-shared -Wl,-z,defs -L"$dir" -Wl,--disable-new-dtags,-rpath,"$dir" -larm_compute -ldl)
cc=${CC:-gcc}; cxx=${CXX:-g++}
"$cc" "${opts[@]}" -c "$here/backend.c" -o "$out/backend.o"
"$cc" "$out/backend.o" "${shared[@]}" -Wl,-soname,libacl_bf16_backend.so -o "$out/libacl_bf16_backend.so"
ln -sf libacl_bf16_backend.so "$out/backend.so"
"$cc" "${opts[@]}" -DACL_BF16_TRACE=1 -c "$here/backend.c" -o "$out/backend-trace.o"
"$cc" "$out/backend-trace.o" "${shared[@]}" -Wl,--wrap,"$symbol" -Wl,-soname,libacl_bf16_trace.so -o "$out/libacl_bf16_trace.so"
ln -sf libacl_bf16_trace.so "$out/backend-trace.so"
for mode in backend trace; do
    bin=acl-probe; [[ $mode != trace ]] || bin=acl-probe-trace
    "$cxx" -std=c++17 -O2 -Wall -Wextra -Werror "$here/../backend/probe.cpp" -L"$out" -Wl,--disable-new-dtags,-rpath,"$out" -l"acl_bf16_$mode" -o "$out/$bin"
    ldd "$out/$bin" > "$out/$bin-deps.txt"
    mapped=$(awk '/libarm_compute.so =>/{print $3}' "$out/$bin-deps.txt")
    [[ $(realpath "$mapped") == $(realpath "$lib") ]]
    nm -D --undefined-only "$out/libacl_bf16_$mode.so" | grep -F "$symbol" > "$out/$bin-kernel-import.txt"
done
python3 - "$out" "$sdk_sha" "$lib_sha" "$symbol" <<'PY'
from pathlib import Path
import hashlib,json,sys
d=Path(sys.argv[1])
files=('backend.o','libacl_bf16_backend.so','libacl_bf16_trace.so','acl-probe','acl-probe-trace','acl-kernel.txt')
hashes={item:hashlib.sha256((d/item).read_bytes()).hexdigest() for item in files}
record=dict(sdk_sha256=sys.argv[2],acl_dso_sha256=sys.argv[3],kernel_symbol=sys.argv[4],
            bfmmla_static_count=len((d/'acl-bfmmla-opcodes.txt').read_text().splitlines()),sha256=hashes)
(d/'acl-identity.json').write_text(json.dumps(record,indent=2)+'\n')
PY
