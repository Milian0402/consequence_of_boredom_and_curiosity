#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -sm) == 'Linux aarch64' ]]
here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
b=$(mkdir -p "$2" && cd "$2" && pwd)
url=https://github.com/ARM-software/ComputeLibrary/releases/download/v53.3.2/arm_compute-v53.3.2-linux-aarch64-cpu-bin.tar.gz
sdk_tar=${3:-$b/acl-cpu-sdk.tar.gz}
if [[ ! -e $sdk_tar ]]; then curl -fL --retry 3 "$url" -o "$sdk_tar"; fi
printf '408cf4d5b3ea81031f915506f90977fba7549abc7dbff5929562b5767e215117  %s\n' "$sdk_tar" | sha256sum -c -
tar -xzf "$sdk_tar" -C "$b"
sdk=$b/arm_compute-v53.3.2-linux-aarch64-cpu-bin
libdir=$sdk/lib/armv8a-neon
lib=$libdir/libarm_compute.so
c++ -std=c++17 -O3 -fPIC -I"$sdk" -I"$sdk/include" -c "$here/adapter.cpp" -o "$b/adapter.o"
c++ -std=c++17 -O3 -fPIC -DACL_TRACE -I"$sdk" -I"$sdk/include" -c "$here/adapter.cpp" -o "$b/adapter-trace.o"
python3 "$here/wrap.py" "$lib" "$b"
cc -O2 -fPIC -c "$b/counts.c" -o "$b/counts.o"
cc -fPIC -c "$b/wrap.S" -o "$b/wrap.o"
cat > "$b/exports" <<'EXPORT'
{global: rank_vendor_init; rank_vendor_sgemm; rank_acl_*; local: *;};
EXPORT
opts=(-Wl,-z,defs -shared -L"$libdir" -Wl,-rpath,"$libdir")
c++ "${opts[@]}" -Wl,--version-script="$b/exports" "$b/adapter.o" -larm_compute -pthread -ldl -o "$out/acl.so"
sed "s/global:/global: _ZN8arm_gemm*;/" "$b/exports" > "$b/trace-exports"
c++ "${opts[@]}" -Wl,--version-script="$b/trace-exports" "$b/adapter-trace.o" "$b/wrap.o" "$b/counts.o" -larm_compute -pthread -ldl -o "$out/acl-trace.so"
{ printf 'official_asset=%s\nsource_tag=v53.3.2\n' "$url"; sha256sum "$sdk_tar" "$lib" "$out/acl.so" "$out/acl-trace.so"; c++ --version; } > "$out/acl-build.txt"
for mode in ordinary trace; do
    python3 "$here/smoke.py" "$out" "$mode" >> "$out/acl-build.txt" 2>&1 || { cat "$out/acl-build.txt"; exit 1; }
done
readelf --wide -d "$out/acl.so" > "$out/acl-ordinary-dynamic.txt"
nm -D --defined-only "$out/acl.so" > "$out/acl-ordinary-exports.txt"
nm -D --defined-only "$out/acl-trace.so" > "$out/acl-trace-exports.txt"
if grep -q rank_acl_count "$out/acl-ordinary-exports.txt"; then exit 1; fi
grep -q rank_acl_count "$out/acl-trace-exports.txt"
