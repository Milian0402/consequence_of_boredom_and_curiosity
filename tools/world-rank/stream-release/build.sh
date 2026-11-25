#!/usr/bin/env bash
set -euo pipefail
src=$1; shared=$2; build=$3; out=$4; cpu=$5
dir="$shared/stream-release"
cc=${CC:-gcc}
sha(){ sha256sum "$1" | cut -d' ' -f1; }
worker_flags=(-std=c11 -O3 -march=x86-64 -I"$shared" -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
printf 'implementation,build_system,option,dialect,archive_sha256,worker_sha256,public_symbol,has_avx2,has_avx512\n' > "$out/product-builds.csv"
printf 'implementation,header_sha256,build_archive_sha256,linked_archive_sha256,worker_sha256,trace_sha256,public_symbol,trace_wrappers\n' > "$out/linked-products.csv"
for method in cmake-default cmake-att cmake-intel make-default make-att make-intel cmake-native-default cmake-native-on; do
    target=$build/$method
    mkdir -p "$target" "$target/sdk/include" "$target/sdk/lib"
    dialect=att; mode=ON; cpu_flags=-march=x86-64
    case $method in *-default) mode=OFF ;; *-intel) dialect=intel ;; esac
    options=()
    if [[ $mode == ON ]]; then options=(-DCOB_ENABLE_X86_STREAM_B=ON); fi
    if [[ $method == cmake-native-* ]]; then
        cpu_flags=-march=native
        options+=(-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_ENABLE_X86_MEDIUM_ROUTE=ON)
        while IFS= read -r flag; do [[ -z $flag ]] || cpu_flags+=" -D$flag"; done < "$out/dispatch-flags.txt"
    fi
    cflags="$cpu_flags -masm=$dialect"
    sdk="$target/sdk"
    if [[ $method == cmake-* ]]; then
        system=cmake
        cmake -S "$src" -B "$target/objects" -DCMAKE_C_COMPILER="$cc" -DCMAKE_C_FLAGS="$cflags" \
            -DCMAKE_BUILD_TYPE=Release -DCOB_BUILD_TESTS=ON -DCOB_BUILD_BENCH=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_INSTALL_LIBDIR=lib "${options[@]}" > "$out/$method-configure.txt"
        cmake --build "$target/objects" --parallel 2 > "$out/$method-build.txt" 2>&1 ||
            { tail -40 "$out/$method-build.txt"; exit 1; }
        taskset -c "$cpu" ctest --test-dir "$target/objects" --output-on-failure > "$out/$method-tests.txt" ||
            { cat "$out/$method-tests.txt"; exit 1; }
        cmake --install "$target/objects" --prefix "$sdk" > "$out/$method-install.txt"
        cmp "$target/objects/libcob_matmul.a" "$sdk/lib/libcob_matmul.a"
        built="$target/objects/libcob_matmul.a"
        if [[ $mode == ON ]]; then
            taskset -c "$cpu" "$target/objects/cob_stream_b_test" >> "$out/$method-tests.txt"
        fi
        python3 -B "$dir/flags.py" "$target/objects/compile_commands.json" "$method" "$out/$method-flags.json"
    else
        system=make-objects
        make_args=("BUILD_DIR=$target/objects" "CC=$cc")
        if [[ $mode == ON ]]; then make_args+=(COB_ENABLE_X86_STREAM_B=ON); fi
        env CFLAGS="-O3 $cflags" taskset -c "$cpu" make -C "$src" --no-print-directory -j2 "${make_args[@]}" test > "$out/$method-tests.txt" 2>&1 ||
            { tail -40 "$out/$method-tests.txt"; exit 1; }
        objects=("$target/objects/gemm.o")
        if [[ $mode == ON ]]; then
            objects+=("$target/objects/stream_b_avx2.o" "$target/objects/stream_b_avx512.o")
        fi
        ar rcsD "$sdk/lib/libcob_matmul.a" "${objects[@]}"
        cp "$src/include/cob_gemm.h" "$sdk/include/"
        built="$sdk/lib/libcob_matmul.a"
        grep -E ' -c src/(gemm|cob_stream_b_kernel)\.c( |$)' "$out/$method-tests.txt" > "$out/$method-commands.txt"
    fi
    cmp "$src/include/cob_gemm.h" "$sdk/include/cob_gemm.h"
    archive="$sdk/lib/libcob_matmul.a"
    nm --defined-only "$archive" > "$target/archive.nm"
    grep -Eq ' T cob_sgemm_rowmajor$' "$target/archive.nm"
    printf 'method=%s option=%s dialect=%s test=pass' "$method" "$mode" "$dialect" >> "$out/generic-build-tests.txt"
    wrap=(); flags=()
    if [[ $mode == ON ]]; then
        for width in avx2 avx512; do grep -Eq " T cob_stream_b_kernel_$width$" "$target/archive.nm"; done
        row=$(grep '^PASS native_width=' "$out/$method-tests.txt" | tail -1)
        [[ -n $row ]]
        printf ' %s' "${row#PASS }" >> "$out/generic-build-tests.txt"
        wrap=(-Wl,--wrap=cob_stream_b_kernel_avx2,--wrap=cob_stream_b_kernel_avx512); flags=(-DSTREAM_RELEASE_ENABLED=1)
        has=1
    else
        if grep -Eq ' T cob_stream_b_kernel_' "$target/archive.nm"; then exit 1; fi
        has=0
    fi
    printf '\n' >> "$out/generic-build-tests.txt"
    "$cc" "${worker_flags[@]}" -I"$sdk/include" "$shared/worker.c" "$archive" -lm -ldl -o "$build/$method.worker"
    "$cc" "${worker_flags[@]}" -I"$sdk/include" "${flags[@]}" "$shared/worker.c" "$dir/witness.c" "${wrap[@]}" "$archive" -lm -ldl -o "$build/$method.trace.worker"
    nm --defined-only "$build/$method.worker" > "$target/worker.nm"
    grep -Eq ' T cob_sgemm_rowmajor$' "$target/worker.nm"
    if grep -Eq ' (T|t) __wrap_|cob_stream_b_count|stream_trace_' "$target/worker.nm"; then exit 1; fi
    ldd "$build/$method.worker" > "$out/$method-libraries.txt"
    if grep -Ei 'lib(openblas|blas|mkl|gomp|iomp)|not found' "$out/$method-libraries.txt"; then exit 1; fi
    nm --defined-only "$build/$method.trace.worker" > "$target/trace.nm"
    wrappers=$(awk '$2=="T" && $3~/^__wrap_cob_stream_b_kernel_avx(2|512)$/ {n++} END{print n+0}' "$target/trace.nm")
    [[ $wrappers == $((2*has)) ]]
    printf '%s,%s,%s,%s,%s,%s,cob_sgemm_rowmajor,%s\n' "$method" "$(sha "$sdk/include/cob_gemm.h")" "$(sha "$built")" \
        "$(sha "$archive")" "$(sha "$build/$method.worker")" "$(sha "$build/$method.trace.worker")" "$wrappers" >> "$out/linked-products.csv"
    printf '%s,%s,%s,%s,%s,%s,cob_sgemm_rowmajor,%s,%s\n' "$method" "$system" "$mode" "$dialect" \
        "$(sha "$archive")" "$(sha "$build/$method.worker")" "$has" "$has" >> "$out/product-builds.csv"
done
