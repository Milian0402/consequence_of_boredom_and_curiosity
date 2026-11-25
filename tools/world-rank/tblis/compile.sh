#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]]
here=$(cd "$(dirname "$0")" && pwd)
out=$(cd "$1" && pwd)
archive=$(realpath "$2")
proof="$out/tblis-evidence"
mkdir -p "$proof"
src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT
sha=07bab80b9bde049705520c739ac291b90e7924ae637f56732a9039c7dfc817a8
printf '%s  %s\n' "$sha" "$archive" | sha256sum -c > "$proof/source.txt"
tar -xzf "$archive" -C "$src" --strip-components=1
cd "$src"
opts=(-O3 -fPIC -ffunction-sections -fdata-sections)
incs=(-iquote . -Isrc -iquote src/util -Isrc/external/tci -Isrc/external/tci/tci)
# These translation units contain both upstream AB schedules and the serial
# classical matrix dependency graph. Tensor contraction is a separate API.
objects='src/iface/3m/mult.lo src/internal/3m/mult.lo src/internal/1m/add.lo src/internal/1m/scale.lo src/internal/1m/set.lo src/stra_internal/3m/stra_mult.lo src/stra_internal/3m/stra_mult_2level.lo src/stra_iface/3m/stra_mult.lo src/stra_iface/3m/stra_mult_2level.lo src/configs/configs.lo src/util/cpuid.lo src/util/thread.lo'
(
    ./configure --enable-config=haswell --enable-thread-model=none --disable-shared --with-pic --with-blas=no \
        CC=gcc CXX=g++ CFLAGS="${opts[*]}" CXXFLAGS="${opts[*]} -std=gnu++11 -include limits" ac_cv_header_hwloc_h=no \
        'ac_cv_f77_mangling=lower case, underscore, no extra underscore' &&
    make -s -j3 -C src/external/tci &&
    make -s -j3 DEFAULT_INCLUDES= AM_CPPFLAGS="${incs[*]}" \
        "lib_libtblis_la_OBJECTS=$objects" lib/libtblis.la
) > "$proof/build.log" 2>&1 || { tail -60 "$proof/build.log" >&2; exit 1; }
archives=(lib/.libs/libtblis.a src/external/tci/lib/.libs/libtci.a)
shared=(-shared -Wl,-z,defs,-Bsymbolic-functions,--gc-sections,--exclude-libs,ALL)
printf 'extern int rank_tblis_haswell(void); int main(void){return rank_tblis_haswell()?0:3;}\n' |
    gcc -x c -c - -o "$src/probe.o"
g++ -std=gnu++11 -include limits "${opts[@]}" "${incs[@]}" -c "$here/backend.cpp" -o "$src/backend.o"
g++ "$src/probe.o" "$src/backend.o" "${archives[@]}" -lpthread -lm -o "$src/probe"
if "$src/probe" > "$proof/selected-backend.txt" 2>&1; then
    cat "$proof/selected-backend.txt"
else
    code=$?
    if [[ $code == 3 ]]; then
        echo 'skipped: upstream selected a backend other than Haswell FP32 16x6' | tee "$out/tblis-status.txt"
        exit 0
    fi
    exit "$code"
fi
for variant in timed trace; do
    flags=()
    if [[ $variant == trace ]]; then flags=(-DTBLIS_TRACE=1 -Wl,--wrap=bli_sgemm_asm_16x6); fi
    g++ -std=gnu++11 -include limits "${opts[@]}" "${incs[@]}" "${flags[@]}" -c "$here/backend.cpp" -o "$src/backend.o"
    for level in 1 2; do
        name="tblis-ab-$level"
        gcc -std=gnu11 "${opts[@]}" "${incs[@]}" -DTBLIS_AB_LEVEL="$level" -c "$here/adapter.c" -o "$src/adapter.o"
        binary="$src/$name-$variant.so"
        g++ "${shared[@]}" "${flags[@]}" "$src/adapter.o" "$src/backend.o" "${archives[@]}" -lpthread -lm -o "$binary"
        if [[ $variant == timed ]]; then
            readelf -d "$binary" > "$proof/$name-dynamic.txt"
            nm -D "$binary" > "$proof/$name-symbols.txt"
            if grep -Ei '(libgomp|libiomp|libtbb|libblas|libopenblas|libmkl)' "$proof/$name-dynamic.txt" ||
               grep -E ' (pthread_create|GOMP_parallel|cblas_sgemm|sgemm_)(@|$)' "$proof/$name-symbols.txt"; then
                echo "Unexpected runtime dependency" >&2; exit 1
            fi
        fi
        gcc -std=gnu11 -O2 "$here/check.c" "$binary" -Wl,-rpath,"$src" -lm -o "$src/check"
        "$src/check" > "$proof/$name-$variant.txt" 2>&1
        if [[ $variant == trace ]]; then
            gcc -std=gnu11 -O2 -DTBLIS_AB_LEVEL="$level" "$here/route.c" "$binary" -Wl,-rpath,"$src" -o "$src/route"
            "$src/route" >> "$proof/$name-$variant.txt" 2>&1
        fi
    done
done
for level in 1 2; do
    cp "$src/tblis-ab-$level-timed.so" "$out/tblis-ab-$level.so"
done
sha256sum "$out/tblis-ab-1.so" "$out/tblis-ab-2.so" > "$proof/binaries.sha256"
echo 'qualified: native Intel haswell_d8x6, FP32 16x6 assembly, serial AB levels 1 and 2' | tee "$out/tblis-status.txt"
