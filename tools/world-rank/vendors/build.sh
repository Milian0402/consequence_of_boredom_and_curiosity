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
    options=(-DRANK_INTEL -I"$inc")
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

"$cc" -std=c11 -O3 -Wall -Wextra -Werror -fPIC -shared "${options[@]}" \
    "$src/vendor_backend.c" -Wl,-z,defs -Wl,--disable-new-dtags,-rpath,"$lib" \
    "${libraries[@]}" -o "$scratch/backend.so"
"$cc" -std=c11 -O3 -Wall -Wextra -Werror -fPIC -shared \
    -DRANK_BACKEND_PATH="\"$scratch/backend.so\"" \
    "$src/vendor_adapter.c" -Wl,-z,defs -ldl -o "$dest/$name.so"
{
    printf 'vendor=%s\nversion=%s\n' "$name" "$version"
    readelf -d "$scratch/backend.so"
    ldd "$scratch/backend.so"
} > "$dest/$name-build.txt"
cat "$dest/$name-build.txt"
