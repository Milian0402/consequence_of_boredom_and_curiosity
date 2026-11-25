#!/usr/bin/env bash
set -euo pipefail

dest=$(mkdir -p "${1:?usage: build.sh OUTDIR}" && cd "$1" && pwd)
src=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d "${RUNNER_TEMP:-/tmp}/gemm-vendor.XXXXXX")
cc=${CC:-cc}

# Keep third-party distributions in runner-local storage, outside the evidence directory.
fetch() {
    curl --fail --location --silent --show-error --retry 3 --retry-all-errors \
        --connect-timeout 30 --max-time 360 "$1" -o "$2"
    printf '%s  %s\n' "$3" "$2" | sha256sum --check --status
}

case "$(uname -s)/$(uname -m)" in
Linux/x86_64)
    name=mkl
    version=2026.2.1
    runtime=mkl-2026.2.1-py2.py3-none-manylinux_2_28_x86_64.whl
    headers=mkl_include-2026.2.1-py2.py3-none-manylinux_2_28_x86_64.whl
    fetch "https://files.pythonhosted.org/packages/e9/31/53bb8fd1098caaf2c634cae75a1abbc1d3c67089a085afa38eeea79f7ea7/$runtime" \
        "$scratch/$runtime" 76322a3abf512594ab7b4e82cb72df91e4e5d195498769aa8b292373c1ddd6d1
    fetch "https://files.pythonhosted.org/packages/f7/c0/0e5f1b0cdcb82d249cd1439383296fea0b92c9e5c6eb6d00c2927e4eaec6/$headers" \
        "$scratch/$headers" 0c4e6f27b1ee435cc7fac115f6df259eb5be38328428cb73927c68dbac32d39f
    unzip -q "$scratch/$runtime" -d "$scratch/runtime"
    unzip -q "$scratch/$headers" -d "$scratch/headers"
    inc="$scratch/headers/mkl_include-$version.data/data/include"
    lib="$scratch/runtime/mkl-$version.data/data/lib"
    options=(-DRANK_INTEL -isystem "$inc")
    libraries=(-L"$lib" -Wl,--no-as-needed -l:libmkl_intel_lp64.so.3
               -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 -Wl,--as-needed -lpthread -lm -ldl)
    ;;
Linux/aarch64)
    name=armpl
    version=26.07.0-1530
    deb=arm-performance-libraries_${version}_arm64.deb
    fetch "https://developer.arm.com/packages/arm-toolchains/ubuntu/pool/$deb" \
        "$scratch/$deb" faacd495d26d58ad444c02b08efcbdbd0c4663b678d77146fd757639945fbfb5
    # This is the Ubuntu repository distribution, whose control archive has
    # no maintainer scripts. Unpack its runtime into ephemeral CI storage.
    dpkg-deb -x "$scratch/$deb" "$scratch/runtime"
    so=$(find "$scratch/runtime/opt/arm" -name 'libarmpl_lp64.so' -print -quit)
    test -n "$so" || { echo "missing serial LP64 ArmPL library" >&2; exit 1; }
    lib=$(dirname "$so")
    inc="$(dirname "$lib")/include"
    options=(-DRANK_ARM -I"$inc")
    libraries=(-L"$lib" -larmpl_lp64 -lm -ldl)
    ;;
*) echo "The vendor adapters require a native Linux x86-64 or aarch64 runner." >&2; exit 2 ;;
esac

if [[ $(cat "$src/../phase.txt") == aligned-core || $(cat "$src/../phase.txt") == release-replay || $(cat "$src/../phase.txt") == signed-leaves || $(cat "$src/../phase.txt") == stream-b || $(cat "$src/../phase.txt") == prefix-eight || $(cat "$src/../phase.txt") == ragged-hk ]]; then
    printf '%s\n%s\n' "$inc" "$lib" > "$dest/$name-prefix.txt"
fi

build_adapter() {
    local variant=$1
    shift
    local backend="$scratch/$variant-backend.so"
    "$cc" -std=c11 -O3 -Wall -Wextra -Werror -fPIC -shared "${options[@]}" "$@" \
        "$src/vendor_backend.c" -Wl,-z,defs -Wl,--disable-new-dtags,-rpath,"$lib" \
        "${libraries[@]}" -o "$backend"
    "$cc" -std=c11 -O3 -Wall -Wextra -Werror -fPIC -shared \
        -DRANK_BACKEND_PATH="\"$backend\"" \
        "$src/vendor_adapter.c" -Wl,-z,defs -ldl -o "$dest/$variant.so"
    {
        printf 'vendor=%s\nversion=%s\n' "$variant" "$version"
        readelf -d "$backend"
        ldd "$backend"
        if [[ "$variant" == mkl-jit ]]; then
            # The installed header must retain both runtime branches at -O3.
            nm -D --undefined-only "$backend" > "$scratch/jit-imports.txt"
            for symbol in mkl_sgemm_jit sgemm_direct; do
                grep -E "[[:space:]]U[[:space:]]+$symbol(@.*)?$" "$scratch/jit-imports.txt"
            done
        fi
        if [[ "$name" == mkl ]]; then
            MKL_VERBOSE=1 MKL_NUM_THREADS=1 python3 "$src/smoke_mkl.py" "$dest/$variant.so"
        fi
    } > "$dest/$variant-build.txt" 2>&1
    if [[ $(cat "$src/../phase.txt") == prefix-eight || $(cat "$src/../phase.txt") == explicit-handles ]]; then
        local identity="$dest/$variant-identity.sha256"
        : > "$identity"
        for member in "$dest/$variant.so" "$backend" "$lib/libmkl_intel_lp64.so.3" "$lib/libmkl_sequential.so.3" "$lib/libmkl_core.so.3" "$src/vendor_backend.c" "$src/vendor_adapter.c" "$src/build.sh"; do
            printf '%s  %s\n' "$(sha256sum "$member" | cut -d' ' -f1)" "$(basename "$member")" >> "$identity"
        done
        nm -D --undefined-only "$backend" > "$dest/$variant-backend-imports.txt"
        readelf -d "$backend" > "$scratch/$variant-dynamic.txt"
        for library in libmkl_intel_lp64.so.3 libmkl_sequential.so.3 libmkl_core.so.3; do
            grep -F "[$library]" "$scratch/$variant-dynamic.txt" >/dev/null
        done
        if grep -E 'NEEDED.*(iomp|gomp|mkl_intel_thread|mkl_gnu_thread|mkl_tbb_thread)' "$scratch/$variant-dynamic.txt"; then return 1; fi
    fi
    cat "$dest/$variant-build.txt"
}

build_adapter "$name"
if [[ "$name" == mkl ]]; then
    build_adapter mkl-jit -DMKL_DIRECT_CALL_SEQ_JIT
fi
