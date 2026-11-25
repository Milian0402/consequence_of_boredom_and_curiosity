#!/usr/bin/env bash
# Build an installed research archive and separate ordinary and diagnostic executables.
set -euo pipefail
src=$(cd "${1:?release checkout}" && pwd)
vendor=$(cd "${2:?pinned vendors}" && pwd)
work=$(mkdir -p "${3:?work directory}" && cd "$3" && pwd)
out=$(mkdir -p "${4:?evidence directory}" && cd "$4" && pwd)
cpu=${5:?pinned CPU}
here=$(cd "$(dirname "$0")" && pwd)
shared=$here/../ragged-hk
release=f322a9ab58454762563b97323fdc5f29016cc39f
[[ $(uname -s) == Linux && $(git -C "$src" rev-parse HEAD) == "$release" ]]
case $(uname -m) in
  x86_64) native=mkl; tuning=-march=native ;;
  aarch64) native=armpl; tuning=-mcpu=native ;;
  *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
patched=$work/hk-leaf-source
if (( $# == 5 )); then
    mkdir "$patched"
    git -C "$src" archive "$release" CMakeLists.txt Makefile include src tests third_party bench | tar -x -C "$patched"
    patch --directory="$patched" --batch --fuzz=0 --forward -p1 --input="$here/candidate.patch" > "$out/hk-leaf-patch.log"
    for path in third_party/fast-matmul/hk233.inc src/cob_hk15_ragged.c; do
        git -C "$src" show "$release:$path" | cmp - "$patched/$path"
    done
    (cd "$patched" && sha256sum CMakeLists.txt Makefile include/*.h src/*.c src/*.h src/*.inc third_party/fast-matmul/hk233.inc) > "$out/hk-leaf-source.sha256"
    (cd "$here" && sha256sum candidate.patch hk-leaf-build.sh hk-leaf-witness.c) > "$out/hk-leaf-tools.sha256"
    inventory=$work/hk-leaf-methods.csv
    status=$out/hk-leaf-build-status.csv
    printf 'method,worker,trace_worker,plugin,role\n' > "$inventory"
    printf 'method,provider,status\n' > "$status"
    for provider in openblas "$native"; do
        label=hk15-$provider-leaf
        if bash "$here/hk-leaf-build.sh" "$src" "$vendor" "$work" "$out" "$cpu" "$provider" > "$out/$label-build.log" 2>&1; then
            printf '%s,%s,%s,public,vendor-assisted\n' "$label" "$work/$label.worker" "$work/$label.trace.worker" >> "$inventory"
            printf '%s,%s,ready\n' "$label" "$provider" >> "$status"
        else
            printf '%s,%s,build_or_public_fixture_failed\n' "$label" "$provider" >> "$status"
            tail -30 "$out/$label-build.log" >&2
        fi
    done
    exit 0
fi

[[ $# == 6 && -d $patched ]]
provider=$6
[[ $provider == openblas || $provider == "$native" ]]
label=hk15-$provider-leaf
build=$work/$label-build
install=$work/$label-install
mkdir -p "$build" "$install"
trap 'printf "%s build failed at line %d: %s\n" "$label" "$LINENO" "$BASH_COMMAND" >&2' ERR
cc=${CC:-cc}
libs=()
files=()
sonames=()
wrap=(cblas_sgemm cob_sgemm_rowmajor cob_hk15_ragged_finish)
opts=(-DCMAKE_BUILD_TYPE=Release -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON -DCMAKE_INSTALL_LIBDIR=lib
      "-DCMAKE_C_COMPILER=$cc" "-DCMAKE_INSTALL_PREFIX=$install" "-DCMAKE_C_FLAGS=$tuning"
      -DCOB_ENABLE_HK15=ON -DCOB_ENABLE_HK15_RAGGED=ON -DCOB_ENABLE_HK15_CBLAS_LEAF=ON
      -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON
      -DCOB_ENABLE_ALIGNED_WINOGRAD=OFF "-DCOB_BLAS_LEAF=${provider^^}")
if [[ $provider == openblas ]]; then
    bash "$here/hk-leaf-selftest.sh" "$patched" "$work/hk-leaf-selftest" "$out" "$cpu"
    read -r prefix < "$vendor/openblas-prefix.txt"
    inc=$prefix/include; lib=$prefix/lib
    [[ -s $lib/libopenblas.a && -s $inc/cblas.h ]]
    if compgen -G "$lib/libopenblas.so*" > /dev/null; then
        echo 'Pinned static serial OpenBLAS required' >&2; exit 1
    fi
    selector=COB_ENABLE_OPENBLAS_LEAF
    export PKG_CONFIG_PATH=
    export PKG_CONFIG_LIBDIR=$build/pinned-pc
    mkdir -p "$PKG_CONFIG_LIBDIR"
    cat > "$PKG_CONFIG_LIBDIR/openblas.pc" <<PC
prefix=$prefix
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: OpenBLAS
Description: Pinned serial archive
Version: 0.3.34
Libs: -L\${libdir} -lopenblas -lpthread -lm
Cflags: -I\${includedir}
PC
    [[ $(pkg-config --variable=libdir openblas) == "$lib" ]]
    libs+=("$lib/libopenblas.a")
    files+=("$lib/libopenblas.a" "$inc/cblas.h")
    wrap+=(openblas_get_num_threads)
else
    mapfile -t sdk < "$vendor/$provider-prefix.txt"
    [[ ${#sdk[@]} == 2 ]]
    inc=${sdk[0]}; lib=${sdk[1]}
    root=$build/pinned-dso
    mkdir -p "$root"
    ln -sfn "$inc" "$root/include"
    ln -sfn "$lib" "$root/lib"
    opts+=("-DCOB_BLAS_ROOT=$root")
    if [[ $provider == mkl ]]; then
        selector=COB_ENABLE_MKL_SEQUENTIAL_LEAF
        sonames=(libmkl_intel_lp64.so.3 libmkl_sequential.so.3 libmkl_core.so.3)
        files+=("$inc/mkl.h")
        wrap+=(MKL_Get_Max_Threads)
    else
        selector=COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
        sonames=(libarmpl_lp64.so)
        files+=("$inc/armpl.h")
    fi
    libs+=("-L$lib" "-Wl,--push-state,--no-as-needed")
    for so in "${sonames[@]}"; do
        [[ -s $lib/$so ]]
        libs+=("-l:$so")
        files+=("$lib/$so")
    done
    libs+=("-Wl,--pop-state" "-Wl,-rpath,$lib")
fi
libs+=(-lcrypto -lpthread -lm -ldl)
cmake -S "$patched" -B "$build/cmake" -G 'Unix Makefiles' "${opts[@]}"
cmake --build "$build/cmake" --parallel 2
taskset -c "$cpu" ctest --test-dir "$build/cmake" --output-on-failure | tee "$out/$label-tests.log"
cmake --install "$build/cmake"
archive=$install/lib/libcob_matmul.a
cmp "$archive" "$build/cmake/libcob_matmul.a"
cmp "$install/include/cob_gemm.h" "$patched/include/cob_gemm.h"
flags=$build/cmake/CMakeFiles/cob_matmul.dir/flags.make
if [[ $provider == openblas ]]; then
    grep -F -- "$lib/libopenblas.a" "$build/cmake/CMakeFiles/cob_gemm_test.dir/link.txt" > /dev/null
fi
for opt in "$selector" COB_ENABLE_HK15 COB_ENABLE_HK15_RAGGED COB_ENABLE_HK15_CBLAS_LEAF COB_ENABLE_PORTABLE_STRASSEN COB_ENABLE_RECURSIVE_STRASSEN; do
    grep -F -- "-D$opt=1" "$flags" > /dev/null
done
if grep -F -- '-DCOB_ENABLE_ALIGNED_WINOGRAD=1' "$flags" > /dev/null; then exit 1; fi
common=(-std=c11 -O3 "$tuning" -ffunction-sections -fdata-sections "-Wl,--gc-sections"
        -DRANK_VIEW_RECORD_LD=1 -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor
        "-I$install/include" "-I$here/..")
timed=$work/$label.worker; trace=$work/$label.trace.worker
link_proof=(); wrap_flags=()
if [[ $provider == openblas ]]; then
    link_proof=("-Wl,--trace-symbol=cblas_sgemm,--trace-symbol=openblas_get_num_threads")
fi
for symbol in "${wrap[@]}"; do wrap_flags+=("-Wl,--wrap=$symbol"); done
LC_ALL=C "$cc" "${common[@]}" "${link_proof[@]}" "$shared/worker.c" "$archive" "${libs[@]}" -o "$timed" \
    2> "$build/ordinary-link.log" || { cat "$build/ordinary-link.log" >&2; exit 1; }
"$cc" "${common[@]}" "-D$selector=1" "-I$inc" "$shared/worker.c" "$here/hk-leaf-witness.c" \
    "$archive" "${wrap_flags[@]}" "${libs[@]}" -o "$trace"
dyad=$work/$label.dyad
"$cc" "${common[@]}" -D_POSIX_C_SOURCE=200809L "$here/dyad.c" "$archive" "${libs[@]}" -o "$dyad"
if [[ $provider != openblas ]]; then
    python3 "$here/vendor-dso.py" "$provider" "$lib" "$timed" "$trace" "$dyad" > "$out/$label-dso.txt"
fi
for exe in "$timed" "$trace" "$dyad"; do
    base=$(basename "$exe")
    ldd "$exe" > "$build/$base.ldd"
    if grep -Eq 'lib(mkl_rt|mkl_.*_thread|armpl_.*_mp|openblas|iomp|gomp|omp\.|tbb)|not found' "$build/$base.ldd"; then
        echo 'Unexpected shared provider or thread library' >&2; exit 1
    fi
    nm --defined-only "$exe" > "$build/$base.symbols"
    grep -Eq ' T cob_sgemm_rowmajor$' "$build/$base.symbols"
done
if grep -Eq ' (T|t) __wrap_' "$build/$label.worker.symbols"; then
    echo 'Timed binary contains a diagnostic wrapper' >&2; exit 1
fi
for symbol in "${wrap[@]}"; do
    grep -F " T __wrap_$symbol" "$build/$label.trace.worker.symbols" > /dev/null
done
if [[ $provider == openblas ]]; then
    python3 "$here/vendor-static.py" "$archive" "$lib/libopenblas.a" \
      "$build/ordinary-link.log" "$build/$label.worker.symbols" > "$out/$label-static-openblas.txt"
fi
identity=$out/$label-identity.sha256
: > "$identity"
for file in "$timed" "$trace" "$dyad" "$archive" "$install/include/cob_gemm.h" "$patched/CMakeLists.txt" "$patched/Makefile" \
            "$patched/src/gemm.c" "$patched/src/cob_hk15.inc" "$patched/src/cob_cblas_leaf.inc" "$patched/src/cob_hk15_ragged.c" \
            "$patched/third_party/fast-matmul/hk233.inc" "$shared/worker.c" "$here/../views/worker.c" "$here/hk-leaf-witness.c" "$here/dyad.c" "${files[@]}"; do
    printf '%s  %s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(basename "$file")" >> "$identity"
done
{
    printf 'release=%s\ntree=%s\nprovider=%s\narchitecture=%s\nattributed_role=vendor-assisted\n' \
        "$release" "$(git -C "$src" rev-parse 'HEAD^{tree}')" "$provider" "$(uname -m)"
    printf 'patch_sha256=%s\ncompiler=%s\ntuning=%s\n' "$(sha256sum "$here/candidate.patch" | cut -d' ' -f1)" "$("$cc" -dumpfullversion)" "$tuning"
    sed -n 's/^C_DEFINES = /cmake_definitions=/p' "$flags"
    for exe in "$timed" "$trace"; do
        printf 'linked executable %s\n' "$(basename "$exe")"
        cat "$build/$(basename "$exe").ldd"
    done
} > "$out/$label-configuration.txt"
