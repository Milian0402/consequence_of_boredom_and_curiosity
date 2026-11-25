#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "${1:?repository}" && pwd)
vendor=$(cd "${2:?vendor SDKs}" && pwd)
build=$(mkdir -p "${3:?build directory}" && cd "$3" && pwd)
out=$(mkdir -p "${4:?results directory}" && cd "$4" && pwd)
dir=$repo/tools/world-rank/signed
study=$repo/tools/world-rank
cc=${CC:-cc}
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
case $(uname -m) in
    aarch64) native=armpl; tuning=-mcpu=native ;;
    x86_64) native=mkl; tuning=-march=native ;;
    *) exit 2 ;;
esac
if [[ $# == 4 ]]; then
    python3 "$dir/prepare.py" "$repo" "$build" "$out" > "$out/source-check.txt"
    printf 'implementation,provider,signed,base_commit,patch_sha256,gemm_sha256,winograd_sha256,library_sha256,worker_sha256,trace_worker_sha256,vendor_sha256,pristine_library_sha256,public_symbol,trace_absent,tests_passed\n' > "$out/builds.csv"
    status=0
    for p in "$native" openblas; do
        for mode in original signed; do
            label=$mode-$p
            if ! bash "$0" "$repo" "$vendor" "$build" "$out" "$p" "$mode" > "$out/$label-build.txt" 2>&1; then
                tail -50 "$out/$label-build.txt" >&2
                status=1
            fi
        done
    done
    exit "$status"
fi
provider=${5:?provider}
mode=${6:?mode}
label=$mode-$provider
case $mode in
    original) sign=0; src=$build/source.signed ;;
    signed) sign=1; src=$build/source.signed ;;
    *) exit 2 ;;
esac
home=$build/$label
mkdir -p "$home"
vendor_files=()
needed=()
if [[ $provider == openblas ]]; then
    read -r sdk < "$vendor/openblas-prefix.txt"
    test -s "$sdk/lib/libopenblas.a"
    if compgen -G "$sdk/lib/libopenblas.so*" > /dev/null; then echo "static OpenBLAS required" >&2; exit 1; fi
    export PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=$home/pc
    mkdir -p "$PKG_CONFIG_LIBDIR"
    cat > "$PKG_CONFIG_LIBDIR/openblas.pc" <<PC
prefix=$sdk
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: OpenBLAS
Description: Serial static OpenBLAS used by the direct peer
Version: 0.3.34
Libs: -L\${libdir} -lopenblas -lpthread -lm
Cflags: -I\${includedir}
PC
    selector=COB_ENABLE_OPENBLAS_LEAF
    libraries=("$sdk/lib/libopenblas.a" -lpthread -lm -ldl)
    vendor_files=("$sdk/lib/libopenblas.a")
else
    [[ $provider == "$native" ]]
    mapfile -t dirs < "$vendor/$provider-prefix.txt"
    [[ ${#dirs[@]} == 2 && -d ${dirs[0]} && -d ${dirs[1]} ]]
    sdk=$home/sdk
    mkdir -p "$sdk"
    ln -sfn "${dirs[0]}" "$sdk/include"
    ln -sfn "${dirs[1]}" "$sdk/lib"
    if [[ $provider == mkl ]]; then
        selector=COB_ENABLE_MKL_SEQUENTIAL_LEAF
        needed=(libmkl_intel_lp64.so.3 libmkl_sequential.so.3 libmkl_core.so.3)
    else
        selector=COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
        needed=(libarmpl_lp64.so)
    fi
    libraries=(-L"$sdk/lib" -Wl,--push-state,--no-as-needed)
    for lib in "${needed[@]}"; do libraries+=("-l:$lib"); vendor_files+=("$sdk/lib/$lib"); done
    libraries+=(-Wl,--pop-state "-Wl,-rpath,$sdk/lib" -lpthread -lm -ldl)
fi

configure() {
    cmake -G 'Unix Makefiles' -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    "-DCMAKE_C_FLAGS=$tuning -DCOB_ENABLE_SIGNED_WINOGRAD_LEAVES=$sign" -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON \
    -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_ENABLE_ALIGNED_WINOGRAD=ON \
    "-DCOB_BLAS_LEAF=${provider^^}" "-DCOB_BLAS_ROOT=$sdk"
}
configure "$src" "$home/cmake"
cmake --build "$home/cmake" --parallel 2
taskset -c "$cpu" ctest --test-dir "$home/cmake" --output-on-failure | tee "$out/$label-tests.txt"
taskset -c "$cpu" "$home/cmake/cob_openblas_winograd_test" | tee -a "$out/$label-tests.txt"
test -s "$home/cmake/libcob_matmul.a"
archive=$home/cmake/libcob_matmul.a
common=(-std=c11 -O3 "$tuning" -I"$src/include" -I"$study" -I"$sdk/include" "-D$selector=1"
    -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
if [[ $provider == openblas ]]; then
    mkdir -p "$home/fixture"
    ln -sfn "$src" "$home/fixture/source"
    test_flags=(-DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1
        -DCOB_ENABLE_ALIGNED_WINOGRAD=1 "-DCOB_ENABLE_SIGNED_WINOGRAD_LEAVES=$sign")
    if [[ $(uname -m) == aarch64 ]]; then test_flags+=(-DCOB_USE_NEON=1); fi
    "$cc" "${common[@]}" "${test_flags[@]}" -I"$home/fixture" "$dir/frozen/validate.c" \
        "${libraries[@]}" -o "$home/fixture/transition"
    taskset -c "$cpu" "$home/fixture/transition" | tee "$out/$label-fallback.txt"
fi
"$cc" "${common[@]}" "$study/worker.c" "$archive" "${libraries[@]}" -o "$build/$label.worker"
"$cc" "${common[@]}" "-DSIGNED_MODE=$sign" "$study/worker.c" "$dir/trace.c" "$archive" \
    -Wl,--wrap=cblas_sgemm,--wrap=cob_sgemm_rowmajor "${libraries[@]}" -o "$build/$label.trace.worker"
pristine="$build/original-$provider/pristine/libcob_matmul.a"
if [[ $mode == original ]]; then
    configure "$build/source" "$home/pristine"
    cmake --build "$home/pristine" --target cob_matmul --parallel 2
    "$cc" "${common[@]}" "$dir/parity.c" "$pristine" "${libraries[@]}" -o "$build/pristine-$provider.parity"
    "$cc" "${common[@]}" "$dir/parity.c" "$archive" "${libraries[@]}" -o "$build/$label.parity"
fi
cat "$home/cmake/CMakeFiles/cob_matmul.dir/flags.make" "$home/cmake/CMakeFiles/cob_gemm_test.dir/link.txt"
if [[ $provider == openblas ]]; then grep -F "$sdk/lib/libopenblas.a" "$home/cmake/CMakeFiles/cob_gemm_test.dir/link.txt"; fi
for binary in "$build/$label.worker" "$build/$label.trace.worker"; do
    ldd "$binary" > "$home/ldd.txt"; cat "$home/ldd.txt"
    if grep -Eq 'lib(mkl_rt|mkl_.*_thread|armpl_.*_mp|openblas|iomp|gomp|omp\.|tbb)|not found' "$home/ldd.txt"; then exit 1; fi
    for lib in "${needed[@]}"; do
        actual=$(awk -v name="$lib" '$1 == name && $2 == "=>" { print $3 }' "$home/ldd.txt")
        [[ -n $actual && $(realpath -e "$actual") == "$(realpath -e "$sdk/lib/$lib")" ]]
    done
done
nm --defined-only "$build/$label.worker" > "$home/nm.txt"
grep -E '[[:space:]]T cob_sgemm_rowmajor$' "$home/nm.txt"
if grep -Eq '[[:space:]](__wrap_cblas_sgemm|__wrap_cob_sgemm_rowmajor|rank_vendor_leaf_count)$' "$home/nm.txt"; then exit 1; fi
nm --defined-only "$build/$label.trace.worker" > "$home/trace-nm.txt"
grep -E '[[:space:]]T __wrap_(cblas_sgemm|cob_sgemm_rowmajor)$' "$home/trace-nm.txt"
[[ $(grep -Ec '[[:space:]]T __wrap_(cblas_sgemm|cob_sgemm_rowmajor)$' "$home/trace-nm.txt") == 2 ]]
sha256sum "$pristine" "$archive" "$build/$label.worker" "$build/$label.trace.worker" "$src/src/gemm.c" \
    "$src/src/cob_openblas_winograd.inc" "$study/worker.c" "$dir/trace.c" "${vendor_files[@]}"
sha256sum "${vendor_files[@]}" | awk '{n=split($2,p,"/"); print $1"  "p[n]}' > "$out/$label-sdk.sha256"
hash(){ sha256sum "$1" | cut -d' ' -f1; }
printf '%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,1,1\n' "$label" "$provider" "$sign" \
    b218beff1eaf8fd606d0934405cdc1e28ecdfe15 "$(hash "$dir/signed-leaves.patch")" \
    "$(hash "$src/src/gemm.c")" "$(hash "$src/src/cob_openblas_winograd.inc")" "$(hash "$archive")" \
    "$(hash "$build/$label.worker")" "$(hash "$build/$label.trace.worker")" "$(hash "$out/$label-sdk.sha256")" "$(hash "$pristine")" >> "$out/builds.csv"
