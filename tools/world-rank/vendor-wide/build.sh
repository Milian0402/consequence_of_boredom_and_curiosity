#!/usr/bin/env bash
set -euo pipefail
src=$(cd "${1:?verified base source}" && pwd)
sdk=$(cd "${2:?provider SDK directory}" && pwd)
build=$(mkdir -p "${3:?output build}" && cd "$3" && pwd)
out=$(mkdir -p "${4:?evidence}" && cd "$4" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
cc=${CC:-cc}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
python3 -B "$dir/prepare.py" "$src" "$build"
repo=$(git -C "$dir" rev-parse --show-toplevel)
git -C "$repo" show b83033d199137b370c03a5a3d63365f1025162b7:tools/world-rank/challenger-flags.txt > "$out/dispatch-flags.txt"
target=(-O3 -std=c11 -march=native -fPIC -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_X86_MEDIUM_ROUTE=1)
while IFS= read -r flag; do [[ -z $flag ]] || target+=("-D$flag"); done < "$out/dispatch-flags.txt"
source=$build/candidate
printf 'implementation,provider,mode,timed_sha256,trace_sha256\n' > "$out/builds.csv"
sha(){ sha256sum "$1" | cut -d' ' -f1; }
for provider in openblas mkl; do
    opts=()
    if [[ $provider == openblas ]]; then
        read -r prefix < "$sdk/openblas-prefix.txt"
        inc=$prefix/include
        archive=$prefix/lib/libopenblas.a
        lib=$build/provider
        mkdir -p "$lib"
        # Extract SGEMM and its actual dependencies from the pinned archive.
        # All three interfaces load this same shared object in private namespaces.
        "$cc" -shared -Wl,-z,defs -Wl,-soname,libcob_rank_openblas.so \
            -Wl,-u,cblas_sgemm,-u,sgemm_,-u,openblas_get_num_threads,-u,openblas_get_config,-u,openblas_get_parallel \
            "$archive" -lpthread -lm -o "$lib/libcob_rank_openblas.so"
        nm -D --defined-only "$lib/libcob_rank_openblas.so" > "$out/openblas-bindings.txt"
        for sym in cblas_sgemm sgemm_ openblas_get_num_threads openblas_get_config openblas_get_parallel; do
            grep -Eq "[[:space:]][TW] $sym$" "$out/openblas-bindings.txt"
        done
        sha256sum "$archive" "$lib/libcob_rank_openblas.so" > "$out/openblas-provider.sha256"
        libraries=(-lcob_rank_openblas)
        selector=COB_ENABLE_OPENBLAS_LEAF
    else
        mapfile -t dirs < "$sdk/mkl-prefix.txt"
        [[ ${#dirs[@]} == 2 ]]
        inc=${dirs[0]}; lib=${dirs[1]}
        libraries=(-Wl,--no-as-needed -l:libmkl_intel_lp64.so.3 -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 -Wl,--as-needed)
        selector=COB_ENABLE_MKL_SEQUENTIAL_LEAF
        opts=(-DWIDE_MKL=1)
        sha256sum "$lib/libmkl_intel_lp64.so.3" "$lib/libmkl_sequential.so.3" "$lib/libmkl_core.so.3" > "$out/mkl-provider.sha256"
    fi
    for mode in direct full split; do
        name="vendor-wide-$provider-$mode"
        if [[ $mode == direct ]]; then name="matched-$provider"; fi
        for variant in timed trace; do
            suffix=; wrapped=(); defs=(-DCOB_ENABLE_VENDOR_WIDE=1 "-D$selector=1"); units=()
            if [[ $variant == trace ]]; then
                [[ $mode != direct ]] || continue
                suffix=.trace; wrapped=(-DWIDE_TRACE=1); units=("$build/trace.c")
            elif [[ $mode != direct ]]; then units=("$source/src/gemm.c"); fi
            [[ $mode != direct ]] || defs+=(-DWIDE_DIRECT=1)
            [[ $mode != full ]] || defs+=(-DCOB_VENDOR_WIDE_FULL_ONLY=1)
            backend="$build/$name$suffix.backend.so"
            "$cc" "${target[@]}" "${defs[@]}" "${opts[@]}" -I"$inc" -I"$source/include" -I"$source/src" \
                -shared "${units[@]}" "$dir/backend.c" -L"$lib" "-Wl,--disable-new-dtags,-rpath,$lib" \
                "${libraries[@]}" -Wl,-z,defs -ldl -lm -lpthread -o "$backend"
            "$cc" -O3 -std=c11 -fPIC -shared "${wrapped[@]}" "-DWIDE_BACKEND_PATH=\"$backend\"" \
                "$dir/adapter.c" -Wl,-z,defs -ldl -o "$build/$name$suffix.so"
            ldd "$backend" > "$out/$name$suffix.ldd"
            if grep -q 'not found' "$out/$name$suffix.ldd"; then exit 1; fi
            if [[ $provider == mkl ]]; then
                grep -q 'libmkl_sequential.so.3' "$out/$name$suffix.ldd"
                if grep -Eq 'libmkl_(gnu|intel|tbb)_thread' "$out/$name$suffix.ldd"; then exit 1; fi
            else grep -q 'libcob_rank_openblas.so' "$out/$name$suffix.ldd"; fi
        done
        trace_hash=; if [[ $mode != direct ]]; then trace_hash=$(sha "$build/$name.trace.so"); fi
        printf '%s,%s,%s,%s,%s\n' "$name" "$provider" "$mode" "$(sha "$build/$name.so")" "$trace_hash" >> "$out/builds.csv"
    done
done
