#!/usr/bin/env bash
# Real public, ordinarily linked executables plus separate diagnostic adapters.
set -euo pipefail
release=$(cd "${1:?release source}" && pwd)
repo=$(cd "${2:?benchmark repository}" && pwd)
vendor=$(cd "${3:?vendor artifacts}" && pwd)
plugins=$(mkdir -p "${4:?plugin directory}" && cd "$4" && pwd)
previous=$(cd "${5:-$release.previous}" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
study=$repo/tools/world-rank
scratch=$plugins/.release-public
mkdir -p "$scratch"
cc=${CC:-cc}
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
case $(uname -m) in
    x86_64) vendor_leaf=mkl; tuning=-march=native ;;
    aarch64) vendor_leaf=armpl; tuning=-mcpu=native ;;
    *) exit 2 ;;
esac
read -r openblas_root < "$vendor/openblas-prefix.txt"
test -s "$openblas_root/lib/libopenblas.a"
# The benchmarking build of upstream OpenBLAS contains only a static archive.
# Pin pkg-config to it in both normal build systems, even when the host has
# another OpenBLAS development package installed.
export PKG_CONFIG_PATH=
export PKG_CONFIG_LIBDIR=$scratch/pkgconfig
mkdir -p "$PKG_CONFIG_LIBDIR"
cat > "$PKG_CONFIG_LIBDIR/openblas.pc" <<PC
prefix=$openblas_root
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: OpenBLAS
Description: Pinned single-thread build for the native benchmark
Version: 0.3.34
Libs: -L\${libdir} -lopenblas -lpthread -lm
Cflags: -I\${includedir}
PC
[[ $(pkg-config --variable=libdir openblas) == "$openblas_root/lib" ]]
if compgen -G "$openblas_root/lib/libopenblas.so*" >/dev/null; then
    echo "The pinned OpenBLAS installation must contain only the serial static library" >&2
    exit 1
fi
mapfile -t paths < "$vendor/$vendor_leaf-prefix.txt"
[[ ${#paths[@]} -eq 2 ]]
native_root=$scratch/$vendor_leaf-prefix
mkdir -p "$native_root"
ln -sfn "${paths[0]}" "$native_root/include"
ln -sfn "${paths[1]}" "$native_root/lib"
export OPENBLAS_NUM_THREADS=1

build_release() {
    local name=$1 src=$2 provider=$3 mode=$4
    local home=$scratch/$name logs=$plugins/$name-tests.txt
    local align=OFF add=0 root selector
    local archive build make_build dependency resolved
    local -a library cmake_opts make_opts proof pinned_dsos=()
    mkdir -p "$home"
    build=$home/cmake
    make_build=$home/make
    [[ $mode != aligned-two ]] || { align=ON; add=1; }
    if [[ $provider == openblas ]]; then
        root=$openblas_root
        selector=COB_ENABLE_OPENBLAS_LEAF
        library=("$root/lib/libopenblas.a" -lpthread -lm -ldl)
    else
        root=$native_root
        if [[ $provider == mkl ]]; then
            selector=COB_ENABLE_MKL_SEQUENTIAL_LEAF
            pinned_dsos=(libmkl_intel_lp64.so.3 libmkl_sequential.so.3 libmkl_core.so.3)
            library=(-L"$root/lib" "-Wl,--push-state,--no-as-needed" -l:libmkl_intel_lp64.so.3
                -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 "-Wl,--pop-state" -lpthread -lm -ldl)
        else
            selector=COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
            pinned_dsos=(libarmpl_lp64.so)
            library=(-L"$root/lib" -larmpl_lp64 -lm -ldl)
        fi
        library+=("-Wl,-rpath,$root/lib")
    fi
    cmake_opts=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON "-DCMAKE_C_FLAGS=$tuning"
        -DCOB_BUILD_BENCH=OFF -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON)
    make_opts=("BUILD_DIR=$make_build" COB_ENABLE_PORTABLE_STRASSEN=ON COB_ENABLE_RECURSIVE_STRASSEN=ON)
    if [[ $name == release-previous ]]; then
        cmake_opts+=(-DCOB_ENABLE_OPENBLAS_LEAF=ON)
        make_opts+=(COB_ENABLE_OPENBLAS_LEAF=ON)
    else
        cmake_opts+=("-DCOB_BLAS_LEAF=${provider^^}" "-DCOB_ENABLE_ALIGNED_WINOGRAD=$align" "-DCOB_BLAS_ROOT=$root")
        make_opts+=("COB_BLAS_LEAF=${provider^^}" "COB_ENABLE_ALIGNED_WINOGRAD=$align" "COB_BLAS_ROOT=$root")
    fi
    {
        cmake -G "Unix Makefiles" -S "$src" -B "$build" "${cmake_opts[@]}" &&
        cmake --build "$build" --parallel 2 &&
        ctest --test-dir "$build" --output-on-failure &&
        env CFLAGS="-O3 -fPIC $tuning" make -j2 -C "$src" "${make_opts[@]}" test
    } > "$logs" 2>&1 || { tail -100 "$logs" >&2; return 1; }
    if [[ $provider == openblas ]]; then
        grep -F -- "$root/lib/libopenblas.a" "$build/CMakeFiles/cob_gemm_test.dir/link.txt" >/dev/null ||
            { echo "CMake did not link the pinned OpenBLAS archive" >&2; return 1; }
    fi
    archive=$build/libcob_matmul.a
    proof=(-std=c11 -O3 -fPIC "$tuning" "-D$selector=1" -I"$src/include" -I"$root/include")
    local -a trace=("$study/vendor-public/trace.c" "-Wl,--wrap=cblas_sgemm"
        -Drank_backend_leaf_count=rank_vendor_leaf_count -Drank_backend_leaf_field=rank_vendor_leaf_field -DRANK_TRACE=1)
    # Worker calls the archive directly. No adapter, wrapper or symbol
    # isolation participates in the ranked multiply.
    "$cc" -std=c11 -O3 "$tuning" -I"$src/include" -I"$study" \
        -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor \
        "$study/worker.c" "$archive" "${library[@]}" -o "$plugins/$name.worker"
    local variant obj
    for variant in cmake make; do
        obj=$archive
        [[ $variant == cmake ]] || obj=$make_build/gemm.o
        "$cc" "${proof[@]}" "$dir/smoke.c" "$dir/adapter.c" "${trace[@]}" "$obj" "${library[@]}" -o "$home/$variant-smoke"
        printf '%s public CLI: ' "$variant" >> "$logs"
        taskset -c "$cpu" "$home/$variant-smoke" "$add" >> "$logs" 2>&1 || { tail -12 "$logs" >&2; return 1; }
    done
    # Only parity DSOs hide archive symbols, so a worker or loaded peer cannot
    # replace their public entry point. Dynamic vendor symbols remain ordinary.
    "$cc" "${proof[@]}" -shared "$dir/adapter.c" "$archive" "${library[@]}" \
        -Wl,-z,defs,--exclude-libs,ALL -o "$plugins/$name.so"
    "$cc" "${proof[@]}" -shared "$dir/adapter.c" "${trace[@]}" "$archive" "${library[@]}" \
        -Wl,-z,defs,--exclude-libs,ALL -o "$plugins/$name.trace.so"
    for variant in worker so trace.so; do
        ldd "$plugins/$name.$variant" > "$home/$variant.ldd"
        if grep -Eq 'lib(mkl_rt|mkl_.*_thread|armpl_.*_mp|openblas|iomp|gomp|omp\.|tbb)|not found' "$home/$variant.ldd"; then
            cat "$home/$variant.ldd" >&2; return 1
        fi
        for dependency in "${pinned_dsos[@]}"; do
            resolved=$(awk -v name="$dependency" '$1 == name && $2 == "=>" { print $3 }' "$home/$variant.ldd")
            if [[ -z $resolved || $(realpath -e "$resolved") != "$(realpath -e "$root/lib/$dependency")" ]]; then
                echo "$name.$variant resolved $dependency outside the pinned installation: $resolved" >&2
                return 1
            fi
        done
    done
    nm --defined-only "$plugins/$name.worker" > "$home/worker-nm.txt"
    grep -Eq '[[:space:]]T cob_sgemm_rowmajor$' "$home/worker-nm.txt"
    if grep -Eq '[[:space:]](__wrap_cblas_sgemm|rank_vendor_leaf_count|rank_vendor_init)$' "$home/worker-nm.txt"; then
        echo "Unexpected ranking-worker instrumentation" >&2; return 1
    fi
    {
        printf 'method=%s provider=%s alignment=%s normal_public_worker=1\n' "$name" "$provider" "$align"
        printf 'cmake_args=%s\nmake_args=%s\nworker_libraries=%s\n' "${cmake_opts[*]}" "${make_opts[*]}" "${library[*]}"
        cat "$build/CMakeFiles/cob_matmul.dir/flags.make" "$build/CMakeFiles/cob_gemm_test.dir/link.txt"
        grep 'public CLI:\|production public entry\|tests passed' "$logs"
        for variant in worker so trace.so; do readelf -d "$plugins/$name.$variant"; cat "$home/$variant.ldd"; done
        grep -E '[[:space:]](cob_sgemm_rowmajor|cblas_sgemm|sgemm_)$' "$home/worker-nm.txt"
        sha256sum "$plugins/$name.worker" "$plugins/$name.so" "$plugins/$name.trace.so" "$archive" \
            "$make_build/gemm.o" "$src/src/gemm.c" "$src/src/cob_openblas_winograd.inc" "$src/CMakeLists.txt" "$src/Makefile"
        sha256sum "$dir/adapter.c" "$dir/smoke.c" "$study/vendor-public/trace.c" "$study/worker.c" "$src/tests/test_openblas_winograd.c"
        if [[ -f "$src/src/cob_cblas_leaf.inc" ]]; then sha256sum "$src/src/cob_cblas_leaf.inc"; fi
        case $provider in
            openblas) sha256sum "$root/lib/libopenblas.a" "$PKG_CONFIG_LIBDIR/openblas.pc" ;;
            mkl) sha256sum "$root/lib"/{libmkl_intel_lp64.so.3,libmkl_sequential.so.3,libmkl_core.so.3} ;;
            armpl) sha256sum "$root/lib/libarmpl_lp64.so" ;;
        esac
    } > "$plugins/$name-build.txt"
}
build_release release-previous "$previous" openblas two
for provider in openblas "$vendor_leaf"; do
    for mode in two aligned-two; do build_release "release-$provider-$mode" "$release" "$provider" "$mode"; done
done
