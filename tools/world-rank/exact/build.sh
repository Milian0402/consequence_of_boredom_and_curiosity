#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$1" && pwd); vendor=$(cd "$2" && pwd); work=$(cd "$3" && pwd); out=$(cd "$4" && pwd)
cpu=$5
here=$(cd "$(dirname "$0")" && pwd); rank=$(dirname "$here"); precision=$rank/precision
cc=${CC:-cc};cxx=${CXX:-c++}
[[ $(git -C "$repo" rev-parse HEAD) == e4b7ce19cdf49f9175c86d3e91ea8d3a756eae9b ]]
case $(uname -m) in
 x86_64) tune=-march=native; scoped=(-DCOB_ENABLE_X86_MEDIUM_ROUTE=ON -DCOB_ENABLE_X86_STREAM_B=ON); native=mkl ;;
 aarch64) tune=-mcpu=native; scoped=(-DCOB_ENABLE_N2_SMALL=ON); native=armpl ;;
 *) exit 2;;
esac
mkdir -p "$out/build" "$work/adapters"
plugins=$work/adapters
cm=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON "-DCMAKE_C_FLAGS=$tune" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON)
cflags=(-O3 -std=c11 "$tune" -fno-fast-math -fPIC)
build_cob() {
 local name=$1;shift; local build=$work/build-$name
 cmake -S "$repo" -B "$build" "${cm[@]}" "$@" > "$out/build/$name.configure"
 cmake --build "$build" --target cob_matmul cob_gemm_test --parallel 3 > "$out/build/$name.build"
 taskset -c "$cpu" ctest --test-dir "$build" -R '^cob_gemm_test$' --output-on-failure > "$out/build/$name.test"
 cp "$build/compile_commands.json" "$out/build/$name.compile.json"
}
build_cob cob-standalone -DCOB_BLAS_LEAF=NONE "${scoped[@]}"
archive=$work/build-cob-standalone/libcob_matmul.a
"$cc" "${cflags[@]}" -shared -Wl,-z,defs -Wl,--exclude-libs,ALL -I"$repo/include" "$here/providers/cob.c" "$archive" -lm -o "$plugins/cob-standalone.so"
if [[ $native == mkl ]];then
 mapfile -t sdk < "$vendor/mkl-prefix.txt";prefix=$work/mkl-root;mkdir "$prefix";ln -s "${sdk[0]}" "$prefix/include";ln -s "${sdk[1]}" "$prefix/lib"
 build_cob release-mkl-aligned-two -DCOB_BLAS_LEAF=MKL "-DCOB_BLAS_ROOT=$prefix" -DCOB_ENABLE_ALIGNED_WINOGRAD=ON
 lib=${sdk[1]}
 "$cc" "${cflags[@]}" -shared -Wl,-z,defs -Wl,--exclude-libs,ALL -I"$repo/include" "$here/providers/cob.c" "$work/build-release-mkl-aligned-two/libcob_matmul.a" -Wl,--push-state,--no-as-needed -L"$lib" -l:libmkl_intel_lp64.so.3 -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 -Wl,--pop-state "-Wl,-rpath,$lib" -lpthread -ldl -lm -o "$plugins/release-mkl-aligned-two.so"
 ldd "$plugins/release-mkl-aligned-two.so" > "$out/build/release-mkl-aligned-two.ldd"
 grep -F 'libmkl_sequential.so.3 =>' "$out/build/release-mkl-aligned-two.ldd"
fi
if [[ $native == mkl ]];then
 bash "$precision/backend/build.sh" "$vendor/onednn/prefix" "$work/bf16"
 bf16=libbf16_backend.so
else
 bash "$precision/acl-backend/build.sh" "$vendor" "$work/bf16"
 bf16=libacl_bf16_backend.so
fi
"$cc" "${cflags[@]}" -shared -Wl,-z,defs -Wl,--exclude-libs,ALL -DPRECISION_MODE=1 "$precision/core/plugin.c" "$precision/core/precision.c" "$precision/core/fp32.c" "$archive" -L"$work/bf16" "-Wl,-rpath,$work/bf16" -l:"$bf16" -lm -o "$plugins/raw-bf16.so"
"$cc" "${cflags[@]}" -shared -Wl,-z,defs "$here/dispatch/dispatch.c" -ldl -pthread -lm -o "$plugins/exact-dispatch.so"
"$cc" "${cflags[@]}" -shared -Wl,-z,defs "$here/providers/policy.c" -ldl -lpthread -o "$plugins/fp32-policy.so"
"$cc" "${cflags[@]}" -shared "$here/providers/binary.c" -o "$plugins/bitset-popcount.so"
"$cxx" -std=c++17 -fPIC -O3 "$tune" -fno-fast-math -shared -Wl,-z,defs -I"$vendor/onednn/prefix/include" "$here/providers/int8.cpp" -L"$vendor/onednn/prefix/lib" "-Wl,-rpath,$vendor/onednn/prefix/lib" -ldnnl -o "$plugins/onednn-int8.so"
if [[ ${EXACT_CHECKED_S8:-0} == 1 ]]; then
 "$cxx" -std=c++17 -fPIC -O3 -fno-fast-math -shared -Wl,-z,defs -I"$vendor/onednn/prefix/include" "-DEXACT_INT8_DNNL_PATH=\"$vendor/onednn/prefix/lib/libdnnl.so.3\"" "$here/s8/checked-int8.cpp" -ldl -pthread -o "$plugins/s8-dispatch.so"
 "$cc" "${cflags[@]}" -I"$here/fixtures" "$here/s8/probe.c" "$here/fixtures/exact_fixture.c" -lcrypto -ldl -lm -o "$work/checked-int8-probe"
 (cd "$here" && sha256sum s8/checked-int8.cpp s8/checked-int8.h s8/jit.py s8/probe.c s8/test.sh s8/smoke.cpp driver/{study.py,score.py,worker.c} providers/{int8.cpp,int8-probe.c} fixtures/{exact_fixture.c,exact_fixture.h} build.sh run.sh prove.py) > "$out/build/s8-sources.sha256"
 cp "$here/fixtures/data/SHA256SUMS" "$out/build/s8-inputs.sha256"
 ldd "$plugins/s8-dispatch.so" > "$out/build/s8-dispatch.ldd"
 ldd "$plugins/onednn-int8.so" > "$out/build/onednn-int8.ldd"
 sha256sum "$vendor/onednn/prefix/lib/libdnnl.so.3" "$work/checked-int8-probe" "$plugins/s8-dispatch.so" "$plugins/onednn-int8.so" > "$out/build/s8-runtime.sha256"
fi
"$cc" -std=c11 -O2 "$here/providers/int8-probe.c" -ldl -o "$work/int8-probe"
"$cc" "${cflags[@]}" -I"$here/fixtures" "$here/driver/worker.c" "$here/fixtures/exact_fixture.c" -lcrypto -ldl -lm -o "$work/exact.worker"
objdump -d "$plugins/bitset-popcount.so" | awk '/popcnt|cnt[[:space:]]+v[0-9]+/{print}' > "$out/build/popcount-instructions.txt"
sha256sum "$work/exact.worker" "$archive" "$plugins"/*.so > "$out/build/products.sha256"
ldd "$plugins/raw-bf16.so" > "$out/build/raw-bf16.ldd"
cp "$work/bf16/"*-build.txt "$out/build/" 2>/dev/null || true
if [[ -f $work/bf16/acl-identity.json ]];then cp "$work/bf16/acl-identity.json" "$out/build/";fi
