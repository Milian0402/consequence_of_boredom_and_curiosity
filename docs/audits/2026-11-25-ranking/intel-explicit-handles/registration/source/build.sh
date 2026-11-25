#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
vendor=$(cd "${1:?vendor directory required}" && pwd)
out=$(mkdir -p "${2:?output directory required}" && cd "$2" && pwd)
readarray -t sdk < "$vendor/mkl-prefix.txt"
[[ ${#sdk[@]} == 2 && $(uname -sm) == 'Linux x86_64' ]]
inc=$(realpath "${sdk[0]}"); lib=$(realpath "${sdk[1]}")
cc=${CC:-gcc}
flags=(-O3 -march=x86-64 -std=c11 -Wall -Wextra -Werror -fPIC -isystem "$inc" "-DMKL_PIN_DIR=\"$lib\"")
dsos=(libmkl_intel_lp64.so.3 libmkl_sequential.so.3 libmkl_core.so.3)
link=(-L"$lib" "-Wl,--disable-new-dtags,-rpath,$lib" -Wl,--no-as-needed)
for dso in "${dsos[@]}"; do link+=("-l:$dso"); done
link+=(-Wl,--as-needed -lm -ldl -lpthread)
wrap=(-Wl,--wrap=mkl_cblas_jit_create_sgemm,--wrap=mkl_jit_get_sgemm_ptr,--wrap=mkl_jit_destroy)
hash(){ sha256sum "$1"|cut -d' ' -f1; }
: > "$out/handle-provider.sha256"
for f in "$inc/mkl_cblas.h" "$inc/mkl_service.h" "$inc/mkl_types.h" "${dsos[@]/#/$lib/}"; do
    printf '%s  %s\n' "$(hash "$f")" "$(basename "$f")" >> "$out/handle-provider.sha256"
done
if [[ -f "$vendor/mkl-identity.sha256" ]]; then
    for dso in "${dsos[@]}"; do
        digest=$(hash "$lib/$dso")
        grep -Fxq "$digest  $dso" "$vendor/mkl-identity.sha256"
    done
fi
printf 'method,source_sha256,observer_sha256,ordinary_sha256,trace_sha256,lp64_sha256,seq_sha256,core_sha256\n' > "$out/handle-builds.csv"
for variant in oneshot prepared; do
    name=mkl-handle-$variant; p=0; [[ $variant != prepared ]] || p=1
    "$cc" "${flags[@]}" -DHANDLE_PREPARED=$p -shared -Wl,-z,defs "$here/handle.c" "${link[@]}" -o "$out/$name.so"
    "$cc" "${flags[@]}" -DHANDLE_PREPARED=$p -shared -Wl,-z,defs "$here/handle.c" "$here/observe.c" "${wrap[@]}" "${link[@]}" -o "$out/$name.trace.so"
    nm -D --undefined-only "$out/$name.so" > "$out/$name.imports"
    for symbol in mkl_cblas_jit_create_sgemm mkl_jit_get_sgemm_ptr mkl_jit_destroy mkl_get_max_threads; do
        grep -Eiq "[[:space:]]U[[:space:]]+$symbol(@.*)?$" "$out/$name.imports"
    done
    if grep -Eiq ' U (cblas_sgemm|sgemm_|mkl_sgemm_jit|sgemm_direct)(@.*)?$' "$out/$name.imports"; then exit 1; fi
    nm -D --defined-only "$out/$name.so" > "$out/$name.symbols"
    if grep -q 'rank_handle_trace' "$out/$name.symbols"; then exit 1; fi
    if [[ $p == 1 ]]; then grep -q ' T rank_handle_prepare$' "$out/$name.symbols"; fi
    ldd "$out/$name.so" > "$out/$name.dependencies"
    readelf -d "$out/$name.so" > "$out/$name.dynamic"
    for dso in "${dsos[@]}"; do
        grep -Fq "[$dso]" "$out/$name.dynamic"
        resolved=$(awk -v libname="$dso" '$1==libname{print $3}' "$out/$name.dependencies")
        [[ $(realpath "$resolved") == $(realpath "$lib/$dso") ]]
    done
    if grep -Eqi 'not found|lib(iomp|gomp|mkl_.*thread|tbb|openblas|blas\.so)' "$out/$name.dependencies"; then exit 1; fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$name" "$(hash "$here/handle.c")" "$(hash "$here/observe.c")" "$(hash "$out/$name.so")" "$(hash "$out/$name.trace.so")" \
        "$(hash "$lib/${dsos[0]}")" "$(hash "$lib/${dsos[1]}")" "$(hash "$lib/${dsos[2]}")" >> "$out/handle-builds.csv"
done
"$cc" "${flags[@]}" "$here/plan-cost.c" "${link[@]}" -o "$out/handle-plan-cost"
"$cc" -O3 -march=x86-64 -std=c11 -Wall -Wextra -Werror -fPIC -shared "$here/fixture.c" -lm -o "$out/handle-fixture.so"
