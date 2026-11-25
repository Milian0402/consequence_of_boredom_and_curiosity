#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "${1:?repository}" && pwd)
vendor=$(cd "${2:?vendor artifacts}" && pwd)
plugins=$(mkdir -p "${3:?plugin directory}" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
work=$plugins/.vendor-public
mkdir -p "$work"
printf '%s\n' "$repo/tools/world-rank" > "$work/helpers.path"
python3 "$dir/prepare.py" "$repo" "$work/source"
cc=${CC:-cc}
case $(uname -m) in
    x86_64) family=mkl; target=(-march=native) ;;
    aarch64) family=armpl; target=(-mcpu=native -DCOB_USE_NEON=1) ;;
    *) exit 2 ;;
esac
for provider in openblas "$family"; do
    if [[ $provider == openblas ]]; then
        read -r prefix < "$vendor/openblas-prefix.txt"
        inc=$prefix/include; lib=$prefix/lib; selector=COB_ENABLE_OPENBLAS_LEAF
        library=("$lib/libopenblas.a" -Wl,--exclude-libs,ALL)
        identity="$dir/identity.c"
    else
        mapfile -t paths < "$vendor/$provider-prefix.txt"
        test "${#paths[@]}" -eq 2
        inc=${paths[0]}; lib=${paths[1]}
        identity="$dir/provider.c"
        if [[ $provider == mkl ]]; then
            selector=COB_ENABLE_MKL_SEQUENTIAL_LEAF
            library=(-Wl,--no-as-needed -l:libmkl_intel_lp64.so.3 -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 -Wl,--as-needed)
        else
            selector=COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
            library=(-larmpl_lp64)
        fi
    fi
    for mode in one two aligned-two; do
        levels=2; aligned=0
        case $mode in one) levels=1 ;; aligned-two) aligned=1 ;; esac
        for variant in timed trace; do
            base=cob-$provider-$mode; opt=(); wrapper=()
            if [[ $variant == trace ]]; then base+=.trace; opt=(-DRANK_TRACE=1); wrapper=("$dir/trace.c" -Wl,--wrap=cblas_sgemm); fi
            backend="$work/$base.so"
            flags=(-std=c11 -O3 -fPIC -I"$inc" -I"$work/source/include" "${target[@]}" "-D$selector=1"
                   -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1
                   "-DRANK_CBLAS_LEVELS=$levels" "-DRANK_CBLAS_ALIGNED=$aligned")
            if [[ -n ${RANK_SMOKE_MIN:-} ]]; then flags+=("-DCOB_CBLAS_WINOGRAD_MIN_HARMONIC=$RANK_SMOKE_MIN"); fi
            "$cc" "${flags[@]}" -shared "$work/source/src/gemm.c" "$identity" "${wrapper[@]}" -L"$lib" \
                -Wl,-z,defs "-Wl,--disable-new-dtags,-rpath,$lib" "${library[@]}" -lpthread -lm -ldl -o "$backend"
            "$cc" -std=c11 -O3 -fPIC -shared "-DRANK_BACKEND_PATH=\"$backend\"" "${opt[@]}" "$dir/adapter.c" -Wl,-z,defs -ldl -o "$plugins/$base.so"
            ldd "$backend" > "$work/$base.ldd"
            if grep -Eq '(libmkl_rt|libmkl_.*_thread|libarmpl_.*_mp|libiomp|libgomp|libomp\.|libtbb|not found)' "$work/$base.ldd"; then cat "$work/$base.ldd" >&2; exit 1; fi
            {
                printf 'method=%s provider=%s include=%s library=%s flags=%s\n' "$base" "$provider" "$inc" "$lib" "${flags[*]}"
                readelf -d "$backend"; cat "$work/$base.ldd"
                sha256sum "$plugins/$base.so" "$backend" "$work/source/src/gemm.c" "$work/source/src/cob_openblas_winograd.inc"
                if [[ $provider == openblas ]]; then sha256sum "$lib/libopenblas.a"; fi
            } > "$plugins/$base-build.txt"
        done
    done
done
