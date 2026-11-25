#!/usr/bin/env bash
# Build each released, vendor-assisted configuration independently.
set -euo pipefail
src=$(cd "${1:?release checkout}" && pwd)
vendor=$(cd "${2:?pinned vendors}" && pwd)
work=$(mkdir -p "${3:?build directory}" && cd "$3" && pwd)
out=$(mkdir -p "${4:?evidence directory}" && cd "$4" && pwd)
cpu=${5:?pinned CPU}
here=$(cd "$(dirname "$0")" && pwd)
release=f322a9ab58454762563b97323fdc5f29016cc39f
shared=$here/../ragged-hk
[[ $(uname -s) == Linux && $(git -C "$src" rev-parse HEAD) == "$release" ]]
git -C "$src" diff --quiet HEAD -- CMakeLists.txt src include tests
case "$(uname -m)" in
    x86_64) native=mkl; tuning=-march=native ;;
    aarch64) native=armpl; tuning=-mcpu=native ;;
    *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
export PYTHONDONTWRITEBYTECODE=1

if [[ $# == 5 ]]; then
    inventory=$work/vendor-release.csv
    status=$out/vendor-release-status.csv
    printf 'method,worker,trace_worker,plugin,role\n' > "$inventory"
    printf 'method,provider,status\n' > "$status"
    incomplete=0
    for provider in openblas "$native"; do
      for mode in aligned-two two; do
        label=release-$provider-$mode
        log="$out/$label-build.log"
        if bash "$0" "$src" "$vendor" "$work" "$out" "$cpu" "$provider" "$mode" > "$log" 2>&1; then
            printf '%s,%s,%s,public,vendor-assisted\n' "$label" "$work/$label.worker" "$work/$label.trace.worker" >> "$inventory"
            printf '%s,%s,ready\n' "$label" "$provider" >> "$status"
        else
            printf '%s,%s,build_or_public_fixture_failed\n' "$label" "$provider" >> "$status"
            incomplete=1
            tail -25 "$log" >&2
        fi
      done
    done
    if ((incomplete)); then
        echo "All four released configurations must link and pass provider verification before timing" >&2
        exit 1
    fi
    python3 "$here/vendor-smoke.py" "$inventory" "$vendor/openblas.so" "$work" "$out" "$cpu"
    exit 0
fi
provider=$6
[[ $provider == openblas || $provider == "$native" ]]
mode=$7
trap 'printf "Release proof failed for %s/%s at line %d: %s\n" "$provider" "$mode" "$LINENO" "$BASH_COMMAND" >&2' ERR
case "$mode" in aligned-two) align=ON; align_bit=1 ;; two) align=OFF; align_bit=0 ;; *) exit 2 ;; esac
label=release-$provider-$mode
build=$work/$label-build
install=$work/$label-install
mkdir -p "$build" "$install"
libs=()
files=()
sonames=()
opts=(-DCMAKE_BUILD_TYPE=Release -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON
      -DCMAKE_INSTALL_LIBDIR=lib "-DCMAKE_INSTALL_PREFIX=$install" "-DCMAKE_C_FLAGS=$tuning"
      -DCOB_ENABLE_HK15=OFF -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON
      "-DCOB_ENABLE_ALIGNED_WINOGRAD=$align" "-DCOB_BLAS_LEAF=${provider^^}")
if [[ $provider == openblas ]]; then
    read -r prefix < "$vendor/openblas-prefix.txt"
    inc="$prefix/include"; lib="$prefix/lib"
    archive="$lib/libopenblas.a"
    [[ -s $archive ]]
    if compgen -G "$lib/libopenblas.so*" >/dev/null; then
        echo 'Expected pinned static serial OpenBLAS' >&2
        exit 1
    fi
    selector=COB_ENABLE_OPENBLAS_LEAF
    export PKG_CONFIG_PATH=
    export PKG_CONFIG_LIBDIR="$build/pinned-pc"
    mkdir -p "$PKG_CONFIG_LIBDIR"
    cat > "$PKG_CONFIG_LIBDIR/openblas.pc" <<PC
prefix=$prefix
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: OpenBLAS
Description: Pinned single-thread static build
Version: 0.3.34
Libs: -L\${libdir} -lopenblas -lpthread -lm
Cflags: -I\${includedir}
PC
    [[ $(pkg-config --variable=libdir openblas) == "$lib" ]]
    files+=("$archive")
    libs+=("$archive")
else
    mapfile -t sdk < "$vendor/$provider-prefix.txt"
    [[ ${#sdk[@]} == 2 ]]
    inc=${sdk[0]}; lib=${sdk[1]}
    prefix="$build/pinned-dso"
    mkdir -p "$prefix"
    ln -sfn "$inc" "$prefix/include"
    ln -sfn "$lib" "$prefix/lib"
    opts+=("-DCOB_BLAS_ROOT=$prefix")
    if [[ $provider == mkl ]]; then
        selector=COB_ENABLE_MKL_SEQUENTIAL_LEAF
        sonames=(libmkl_intel_lp64.so.3 libmkl_sequential.so.3 libmkl_core.so.3)
    else
        selector=COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
        sonames=(libarmpl_lp64.so)
    fi
    libs+=("-L$lib" "-Wl,--push-state,--no-as-needed")
    for library in "${sonames[@]}"; do
        [[ -s $lib/$library ]]
        libs+=("-l:$library")
        files+=("$lib/$library")
    done
    libs+=("-Wl,--pop-state" "-Wl,-rpath,$lib")
fi
libs+=(-lcrypto -lpthread -lm -ldl)
cmake -S "$src" -B "$build/cmake" -G 'Unix Makefiles' "${opts[@]}"
cmake --build "$build/cmake" --target cob_gemm_test cob_openblas_winograd_test --parallel 2
taskset -c "$cpu" ctest --test-dir "$build/cmake" --output-on-failure -R '^(cob_gemm_test|cob_openblas_winograd_test)$'
cmake --install "$build/cmake"
archive="$install/lib/libcob_matmul.a"
cmp "$archive" "$build/cmake/libcob_matmul.a"
cmp "$install/include/cob_gemm.h" "$src/include/cob_gemm.h"
if [[ $provider == openblas ]]; then
    grep -F -- "$lib/libopenblas.a" "$build/cmake/CMakeFiles/cob_gemm_test.dir/link.txt" >/dev/null
fi
required_defs=("$selector" COB_ENABLE_PORTABLE_STRASSEN COB_ENABLE_RECURSIVE_STRASSEN)
if [[ $align == ON ]]; then required_defs+=(COB_ENABLE_ALIGNED_WINOGRAD); fi
for opt in "${required_defs[@]}"; do
    grep -F -- "-D$opt=1" "$build/cmake/CMakeFiles/cob_matmul.dir/flags.make" >/dev/null
done
cc=${CC:-cc}
common=(-std=c11 -O3 "$tuning" -ffunction-sections -fdata-sections "-Wl,--gc-sections"
        -DRANK_VIEW_RECORD_LD=1 -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor
        "-I$install/include" "-I$here/..")
timed=$work/$label.worker
trace=$work/$label.trace.worker
link_proof=()
if [[ $provider == openblas ]]; then
    link_proof=(-Wl,--trace-symbol=cblas_sgemm,--trace-symbol=openblas_get_num_threads)
fi
LC_ALL=C "$cc" "${common[@]}" "${link_proof[@]}" "$shared/worker.c" "$archive" "${libs[@]}" -o "$timed" \
    2> "$build/ordinary-link.log" || { cat "$build/ordinary-link.log" >&2; exit 1; }
"$cc" "${common[@]}" "-D$selector=1" "-DRAGGED_ALIGNED=$align_bit" "-I$inc" "$shared/worker.c" "$shared/vendor-witness.c" \
    "$archive" -Wl,--wrap=cblas_sgemm,--wrap=cob_sgemm_rowmajor "${libs[@]}" -o "$trace"

if [[ $provider != openblas ]]; then
    python3 "$here/vendor-dso.py" "$provider" "$lib" "$timed" "$trace" > "$out/$label-dso.txt"
fi
for exe in "$timed" "$trace"; do
    base=$(basename "$exe")
    ldd "$exe" > "$build/$base.ldd"
    if grep -Eq 'lib(mkl_rt|mkl_.*_thread|armpl_.*_mp|openblas|iomp|gomp|omp\.|tbb)|not found' "$build/$base.ldd"; then
        echo 'Unexpected parallel, fallback or missing provider' >&2
        exit 1
    fi
    nm --defined-only "$exe" > "$build/$base.symbols"
    grep -Eq ' T cob_sgemm_rowmajor$' "$build/$base.symbols"
done
if grep -Eq ' (__wrap_(cblas_sgemm|cob_sgemm_rowmajor)|rank_backend_leaf_count)$' "$build/$(basename "$timed").symbols"; then
    echo 'Ordinary worker has a trace wrapper' >&2
    exit 1
fi
for symbol in __wrap_cob_sgemm_rowmajor __wrap_cblas_sgemm; do
    grep -F " T $symbol" "$build/$(basename "$trace").symbols" >/dev/null
done
if [[ $provider == openblas ]]; then
    python3 "$here/vendor-static.py" "$archive" "$lib/libopenblas.a" \
        "$build/ordinary-link.log" "$build/$(basename "$timed").symbols" > "$out/$label-static-openblas.txt"
fi
identity=$out/$label-identity.sha256
: > "$identity"
for file in "$timed" "$trace" "$archive" "$install/include/cob_gemm.h" "$src/CMakeLists.txt" \
        "$src/src/gemm.c" "$src/src/cob_cblas_leaf.inc" "$src/src/cob_openblas_winograd.inc" \
        "$shared/worker.c" "$shared/vendor-witness.c" "$here/../vendor-public/trace.c" "${files[@]}"; do
    printf '%s  %s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(basename "$file")" >> "$identity"
done
{
    printf 'release=%s\ntree=%s\nprovider=%s\nmode=%s\narchitecture=%s\nattributed_role=vendor-assisted\n' \
        "$release" "$(git -C "$src" rev-parse 'HEAD^{tree}')" "$provider" "$mode" "$(uname -m)"
    printf 'compiler=%s\ntuning=%s\ncmake_definitions=' "$("$cc" -dumpfullversion)" "$tuning"
    sed -n 's/^C_DEFINES = //p' "$build/cmake/CMakeFiles/cob_matmul.dir/flags.make"
} > "$out/$label-configuration.txt"
cat "$identity"
# Independent accuracy diagnostic. The ordinary release worker is already certified.
if [[ -f $here/dyad.c ]]; then
    if "$cc" "${common[@]}" -D_POSIX_C_SOURCE=200809L "$here/dyad.c" "$archive" "${libs[@]}" \
            -o "$build/dyad" > "$out/$label-dyad-build.log" 2>&1; then
        mv "$build/dyad" "$work/$label.dyad"
        printf '%s  %s\n' "$(sha256sum "$work/$label.dyad" | cut -d' ' -f1)" "$label.dyad" >> "$identity"
    else
        printf 'Optional dyad executable failed to link for %s\n' "$label" >&2
        cat "$out/$label-dyad-build.log" >&2
    fi
fi

# Other registered studies can build their own independently guarded worker against this verified archive.
if [[ -n ${PRECISION_VENDOR_WORKER_SOURCE:-} ]]; then
    "$cc" "${common[@]}" "$PRECISION_VENDOR_WORKER_SOURCE" "$archive" "${libs[@]}" -o "$work/$label.precision.worker"
    sha256sum "$work/$label.precision.worker" >> "$out/$label-configuration.txt"
fi
