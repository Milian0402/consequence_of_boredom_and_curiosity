#!/usr/bin/env bash
# Build the standalone numerical guard on the same installed ordinary worker as ragged-hk15.
set -euo pipefail
base=$(cd "${1:?release checkout}" && pwd)
: "${2:?vendor directory required by the study interface}"
work=$(cd "${3:?working directory}" && pwd)
out=$(cd "${4:?evidence directory}" && pwd)
cpu=${5:?pinned CPU}
here=$(cd "$(dirname "$0")" && pwd)
shared=$here/../ragged-hk
cc=${CC:-gcc}
base_sha=f322a9ab58454762563b97323fdc5f29016cc39f
patch_sha=eba736a36d44e2a3ddab2b849d2373bc61061bcb1c0931a33bae91f8b2803c6f
[[ $(git -C "$base" rev-parse HEAD) == "$base_sha" ]]
[[ $(git -C "$base" rev-parse 'HEAD^{tree}') == db8d3b50584af7632355314888857ddafa0a41d7 ]]
[[ $(uname -s) == Linux ]]
case $(uname -m) in
    x86_64) tuning=-march=native ;;
    aarch64) tuning=-mcpu=native ;;
    *) exit 2 ;;
esac
export ZERO_AR_DATE=1
target=$work/guarded-hk15
proof=$out/standalone/guarded-hk15
src=$target/source
mkdir -p "$target" "$proof"
mkdir "$src"
git -C "$base" archive "$base_sha" CMakeLists.txt Makefile src include tests bench third_party | tar -x -C "$src"
(cd "$here" && printf '%s  %s\n' "$patch_sha" stable-hk-source.patch | sha256sum -c)
[[ $(git apply --numstat "$here/stable-hk-source.patch" | cut -f3) == src/cob_hk15.inc ]]
patch --directory="$src" --batch --forward --fuzz=0 -p1 --input="$here/stable-hk-source.patch" > "$proof/patch.log"
(cd "$src" && sha256sum CMakeLists.txt Makefile include/*.h src/*.c src/*.h src/*.inc third_party/fast-matmul/hk233.inc) > "$proof/source.sha256"
opts=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=$cc" "-DCMAKE_C_FLAGS=$tuning" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON -DCOB_BLAS_LEAF=NONE -DCOB_ENABLE_PORTABLE_STRASSEN=ON
      -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_ENABLE_HK15=ON -DCOB_ENABLE_HK15_RAGGED=ON -DCMAKE_INSTALL_LIBDIR=lib)
cmake -S "$src" -B "$target/build" "${opts[@]}" > "$proof/configure.log"
cmake --build "$target/build" --parallel 2 > "$proof/build.log" 2>&1
taskset -c "$cpu" ctest --test-dir "$target/build" --output-on-failure > "$proof/tests.log"
cmake --install "$target/build" --prefix "$target/sdk" > "$proof/install.log"
archive=$target/sdk/lib/libcob_matmul.a
cmp "$archive" "$target/build/libcob_matmul.a"
cmp "$target/sdk/include/cob_gemm.h" "$base/include/cob_gemm.h"
cf=(-std=c11 -O3 "$tuning" -ffunction-sections -fdata-sections -Wl,--gc-sections "-I$target/sdk/include"
    -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
"$cc" "${cf[@]}" "$shared/worker.c" "$archive" -lcrypto -lm -ldl -o "$target/ordinary.worker"
"$cc" -std=gnu11 -O2 "$tuning" "-I$target/sdk/include" "$here/dyad.c" "$archive" -lm -ldl -pthread -o "$work/guarded-hk15.dyad"
for exe in "$target/ordinary.worker" "$work/guarded-hk15.dyad"; do
    name=$(basename "$exe")
    nm --defined-only "$exe" > "$target/$name.symbols"
    grep -E ' T cob_sgemm_rowmajor$' "$target/$name.symbols" > "$proof/$name.public-symbol.txt"
    if grep -E ' [Tt] (__wrap_|hk_trace_)' "$target/$name.symbols"; then exit 1; fi
    ldd "$exe" > "$proof/$name.linked.txt"
    if grep -Ei 'lib(mkl|blas|openblas|armpl|gomp|iomp)|not found' "$proof/$name.linked.txt"; then exit 1; fi
done
cp "$target/build/compile_commands.json" "$proof/compile-commands.json"
{
    for file in "$archive" "$target/ordinary.worker" "$work/guarded-hk15.dyad" \
                "$target/sdk/include/cob_gemm.h" "$src/src/cob_hk15.inc" \
                "$shared/worker.c" "$here/dyad.c" "$here/stable-hk-source.patch" "$here/stable-build.sh"; do
        printf '%s  %s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(basename "$file")"
    done
} > "$proof/products.sha256"
