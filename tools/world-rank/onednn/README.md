# oneDNN single-thread FP32 matmul comparator

The source build pins the official v3.14.2 tag to commit `4db284662a37cf46401655b5ee99391d9de52c00`. Upstream left `PROJECT_VERSION` at 3.14.1, so the library and headers report that version. The script verifies the source archive SHA-256, records the built library hash, and checks its exact linked path and runtime version. The source tag and binary metadata both belong in results. [Release](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.14.2) · [Pinned CMake version](https://github.com/uxlfoundation/oneDNN/blob/4db284662a37cf46401655b5ee99391d9de52c00/CMakeLists.txt#L61)

Run on each native Linux host with a C++17 compiler, CMake, curl, GNU binutils and util-linux. Use a new output directory. The default build parallelism is one. An already downloaded, checksum-verified archive can be passed through the environment.

```sh
./build.sh /tmp/onednn-qualified
# Offline alternative using the archive alongside these sources
SOURCE_ARCHIVE="$PWD/onednn-v3.14.2.tar.gz" ./build.sh /tmp/onednn-qualified-offline
```

The script builds CPU-only, matmul-only oneDNN with sequential threading. It disables external BLAS, Arm Compute Library, KleidiAI, AMD ZenDNN and the primitive cache. The dynamic-dependency and imported-symbol audits reject known external BLAS or threading backends. The CMake cache and compile commands record the source configuration, including disabled optional backends that could otherwise be statically linked. The upstream matmul list includes native x86 and Arm BRGEMM kernels, a generic FP32 GEMM implementation and a reference fallback. In the pinned configuration, the generic GEMM route uses oneDNN’s own implementation. Record the selected implementation from `rank_onednn_implementation()` immediately after a representative untimed call on **each target layout**. Give a reference route its actual label. [Build options](https://github.com/uxlfoundation/oneDNN/blob/4db284662a37cf46401655b5ee99391d9de52c00/cmake/options.cmake) · [External BLAS selection](https://github.com/uxlfoundation/oneDNN/blob/4db284662a37cf46401655b5ee99391d9de52c00/cmake/blas.cmake) · [Matmul kernel selection](https://github.com/uxlfoundation/oneDNN/blob/4db284662a37cf46401655b5ee99391d9de52c00/src/cpu/matmul/cpu_matmul_list.cpp) · [FP32 GEMM matmul](https://github.com/uxlfoundation/oneDNN/blob/4db284662a37cf46401655b5ee99391d9de52c00/src/cpu/matmul/gemm_f32_matmul.cpp)

Both plugins use the existing `rank_vendor_init` and `rank_vendor_sgemm(m,n,k,a,lda,b,ldb,c,ldc)` ABI. The CPU engine and its stream are initialized once on the benchmark thread. Operands use explicit caller-provided row strides, unit column stride, FP32 storage, strict FP32 math and strict accumulation. The matmul has default unit scale and overwrites C. The caller supplies disjoint A, B and C. Nonpositive M or N returns immediately, and K=0 zeroes the logical output. Per-call binding and synchronous completion are inside the timing. The supported precision policy permits FP32 FMA. [Floating-point math modes](https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_fpmath_mode.html) · [Primitive attributes](https://github.com/uxlfoundation/oneDNN/blob/4db284662a37cf46401655b5ee99391d9de52c00/include/oneapi/dnnl/dnnl.hpp)

| Plugin | Work included in each measured call |
| --- | --- |
| `onednn-f32-oneshot.so` | Creates memory descriptors and the matmul primitive, binds three buffers, executes, waits and destroys the primitive. Use as the one-shot comparison. |
| `onednn-f32-cached-plan.so` | Reuses one thread-local primitive for an identical M, N, K and all three strides. A key change recreates it. Each call rebinds the current buffers, executes and waits. Report as a separate warm-plan comparison. |

The shared library’s primitive cache is disabled in both builds. Process-wide kernel initialization and OS caches can still persist between calls. For the warm-plan trial, use one unmeasured call with the same dimensions and leading dimensions ahead of the timed calls. Keep this policy explicit in timing metadata. The plugins cache no caller data or reordered weights.

The build finishes by pinning a fresh process to one allowed CPU for each plugin and confirming a single observed OS thread before and after the test calls. The pinned binary also reports and enforces the sequential CPU runtime. The checker covers five layouts with two deterministic inputs and disjoint allocations apiece, including identical shapes with different leading dimensions. It scans every logical result against an FP64 dot product, verifies exact A/B bytes and output padding, tests that NaN-initialized C is overwritten, checks a one-ULP cancellation residual and the smallest FP32 subnormal, and audits actual plan creations. Independent target-host performance qualification still requires the ranking harness’s full registered layout checks and native timing.

### Local evidence

The linked Linux Arm64 Ubuntu 24.04 package 3.1.1-2 executed `gemm:jit` for all five layouts. Both policies passed 73,096 FP64 outputs each, with maximum absolute error 0.000009562. Both returned the exact FP32 residual and smallest subnormal. Affinity and observed worker counts were one. The older package uses OpenMP with its team limited to one and has no explicit accumulation-mode attribute. The dependency audit lists libgomp and no external BLAS. These results qualify the binding and stride behavior on this particular guest. The pinned source configured and both shim policies compiled against its headers. The pinned sequential Linux binary, Intel/AMD AVX-512 and Arm SVE/SME performance remain untested here. The local diagnostic measured numerical accuracy only.

To reproduce the Ubuntu package diagnostic, install `g++ libdnnl-dev` in an Ubuntu 24.04 Arm64 container. Build both shared objects with the following commands and run the same checker. The diagnostic macro is reserved for the older package. The production build rejects it through its build script.

```sh
mkdir -p /tmp/onednn-distro
g++ -std=c++17 -O1 -fPIC -shared -DONEDNN_DISTRO_DIAGNOSTIC=1 \
    -DONEDNN_CACHED_PLAN=0 adapter.cpp -ldnnl -o /tmp/onednn-distro/onednn-f32-oneshot.so
g++ -std=c++17 -O1 -fPIC -shared -DONEDNN_DISTRO_DIAGNOSTIC=1 \
    -DONEDNN_CACHED_PLAN=1 adapter.cpp -ldnnl -o /tmp/onednn-distro/onednn-f32-cached-plan.so
./qualify.sh /tmp/onednn-distro
```
