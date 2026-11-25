#!/usr/bin/env bash
# Linux, GCC/Clang and CMake. JOBS defaults to one. Optional SOURCE_ARCHIVE is
# accepted only when its SHA-256 matches the upstream commit archive below.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?pass an empty build/output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
test "$(uname -s)" = Linux
sha=4db284662a37cf46401655b5ee99391d9de52c00
digest=55e319d174c6c3884b66e7d643d9579a96d73b20159580dc3b1f89de4875da9c
archive=${SOURCE_ARCHIVE:-"$out/source.tar.gz"}
if [[ ! -f $archive ]]; then
    curl --fail --show-error --location --retry 3 "https://codeload.github.com/uxlfoundation/oneDNN/tar.gz/$sha" -o "$archive"
fi
echo "$digest  $archive" | sha256sum --check
# An existing source/build tree could have different sources or CMake switches.
for dir in src cmake prefix; do
    if [[ -e "$out/$dir" ]]; then echo "$out/$dir exists; use a fresh output directory" >&2; exit 1; fi
done
mkdir "$out/src"
tar -xzf "$archive" --strip-components=1 -C "$out/src"
# JIT kernels select the execution ISA. Separate standard runners can have different models.
case $(uname -m) in
    x86_64) distribution_isa=-msse4.1 ;;
    aarch64) distribution_isa=-march=armv8-a ;;
    *) echo "unsupported CPU family" >&2; exit 1 ;;
esac
cmake -S "$out/src" -B "$out/cmake" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$out/prefix" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_FLAGS=-fno-fast-math -DCMAKE_CXX_FLAGS=-fno-fast-math -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DONEDNN_ARCH_OPT_FLAGS="$distribution_isa" -DONEDNN_LIBRARY_TYPE=SHARED -DONEDNN_CPU_RUNTIME=SEQ -DONEDNN_GPU_RUNTIME=NONE \
    -DONEDNN_BLAS_VENDOR=NONE -DONEDNN_AARCH64_USE_ACL=OFF -DONEDNN_AARCH64_USE_KAI=OFF \
    -DONEDNN_X64_USE_ZEN=OFF -DONEDNN_BUILD_GRAPH=OFF -DONEDNN_BUILD_TESTS=OFF \
    -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_ENABLE_PRIMITIVE=MATMUL \
    -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE_CACHE=OFF 2>&1 | tee "$out/configure.log"
cmake --build "$out/cmake" -j "${JOBS:-1}" 2>&1 | tee "$out/build.log"
cmake --install "$out/cmake" > "$out/install.log"
lib="$out/prefix/lib"
cxx=${CXX:-c++}
for spec in "oneshot 0" "cached-plan 1"; do
    read -r label cached <<<"$spec"
    "$cxx" -std=c++17 -O3 -fno-fast-math -fPIC -shared -DONEDNN_CACHED_PLAN="$cached" \
        -I"$out/prefix/include" "$here/adapter.cpp" -L"$lib" -Wl,-z,defs \
        '-Wl,-rpath,$ORIGIN/prefix/lib' -ldnnl -o "$out/onednn-f32-$label.so"
done
# Verify the exact mapped library and audit the shared-library dependency closure.
for shim in "$out"/onednn-f32-*.so; do
    mapped=$(ldd "$shim" | awk '/libdnnl\.so/{print $3;exit}')
    [[ $(realpath "$mapped") == "$(realpath "$lib/libdnnl.so")" ]] || { echo "wrong oneDNN linkage" >&2; exit 1; }
done
ldd "$lib/libdnnl.so" | tee "$out/dependencies.txt"
nm -D --undefined-only "$lib/libdnnl.so" > "$out/imports.txt"
if grep -Eiq '(openblas|libblas[.]|mkl|blis|aocl|armpl|libarm_compute|KleidiAI|ZenDNN|lib(gomp|omp|iomp))' "$out/dependencies.txt" ||
   grep -Eiq '[[:space:]](cblas_|[sd]gemm_|bli_|mkl_|armpl_|acl_|kai_)' "$out/imports.txt"; then
    echo "external math or threading backend found" >&2; exit 1
fi
{ echo "source_tag=v3.14.2"; echo "source_commit=$sha"; echo "source_sha256=$digest";
  cmake --version; "$cxx" --version; sha256sum "$lib/libdnnl.so" "$out"/onednn-f32-*.so;
} > "$out/build-fingerprint.txt"
"$here/qualify.sh" "$out" "$out/prefix/include"
