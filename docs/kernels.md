# Kernels and build options

COB chooses a kernel on each public call using the matrix dimensions, leading dimensions, build options and CPU features. The one-shot call includes temporary packing and workspace. The packed APIs let the caller keep a packed B, or both packed operands, across calls. [Public interface](../include/cob_gemm.h)

## Configuring a build

Pass an option to CMake as `-DNAME=VALUE` or to Make as `NAME=VALUE`. For example, this builds and tests the Linux Strassen implementation with sequential OpenBLAS subproducts.

```sh
cmake -S . -B build-openblas -DCMAKE_BUILD_TYPE=Release \
  -DCOB_ENABLE_PORTABLE_STRASSEN=ON \
  -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_BLAS_LEAF=OPENBLAS
cmake --build build-openblas --parallel
OPENBLAS_NUM_THREADS=1 ctest --test-dir build-openblas --output-on-failure
```

| Option | Default | Use |
|---|---|---|
| `COB_STRASSEN_SME` | `AUTO` | On native Apple arm64, probe for the MaxMulSK-derived SME leaf. `ON` requires a compatible compiler. Make accepts a separate `COB_SME_CC`. |
| `COB_ENABLE_APPLE_MEDIUM_WIDE` | `OFF` | Apple SME kernel for 24–55 output rows, 1024–4096 columns and reduction depths of 96–384. |
| `COB_ENABLE_N2_SMALL` | `OFF` | Small-matrix kernel tuned on Linux Arm Neoverse N2. |
| `COB_ENABLE_X86_STREAM_B` | `OFF` | Linux x86-64 packed-B kernel for wide outputs. Dispatches to AVX-512F or AVX2/FMA on supported CPUs. |
| `COB_ENABLE_X86_MEDIUM_ROUTE` | `OFF` | Reuse packed A panels on selected medium, wide Linux x86-64 products. |
| `COB_ENABLE_PORTABLE_STRASSEN` | `OFF` | One level of Strassen on eligible large native x86-64 and Arm64 products. |
| `COB_ENABLE_RECURSIVE_STRASSEN` | `OFF` | Add a second Winograd level on eligible Linux products. Requires portable Strassen. |
| `COB_BLAS_LEAF` | `NONE` | Select `OPENBLAS`, `MKL` on Linux x86-64, or `ARMPL` on Linux Arm64 for eligible Strassen subproducts. `COB_ENABLE_OPENBLAS_LEAF=ON` also selects OpenBLAS. |
| `COB_ENABLE_ALIGNED_WINOGRAD` | `OFF` | Align terminal Winograd products on big rectangles. Requires portable and recursive Strassen and a selected BLAS provider. |
| `COB_ENABLE_HK15` | `OFF` | Compute an eligible 2 × 3 × 3 block product with fifteen subproducts. |
| `COB_ENABLE_HK15_RAGGED` | `OFF` | Enable HK15 on the divisible interior and compute the remaining rows, columns and reduction terms separately. |

## Native kernels

On Apple Silicon, COB uses AMX and, where available, SME2.1. The dispatcher checks hardware support and the streaming vector length. For 17–32 output rows and columns and a reduction depth of 16–48, a 16-lane FP32 SME kernel holds small products in its matrix accumulator. Selected few-row products read B directly. The optional medium-wide kernel packs A once. On compact products with 24–32 rows it reads B directly for both output tiles. For padded matrices and 33–55 rows it packs B while computing the first output tile. The compile definition `COB_DISABLE_APPLE_AMX=1` disables the Apple AMX path. [Apple compact measurements](audits/2026-11-25-ranking/m5-two-row-tiles/README.md)

Arm64 also has an 8 × 8 NEON kernel. On Linux x86-64, COB can select AVX-512 or AVX2/FMA at runtime. The optional streamed-B kernel covers 24–160 rows, 1024–4096 columns and depths of 96–384, with at least eight columns per row and at most 49,152 elements in A. [x86 installed-library comparison](audits/2026-11-25-ranking/stream-b-release/README.md)

## Strassen and Winograd

One level of Strassen replaces eight half-size products with seven. In the default Apple configuration, the crossover for eligible compact squares is 5632 and for other eligible balanced products it is 6144. These cases have dimensions divisible by 64 and a largest-to-smallest dimension ratio of at most 4:3. Portable Strassen has a separate crossover of 1536. Eligible rectangles can use Strassen on the aligned interior, with direct edge products. The three outer scratch buffers on that route use at most 256 MiB. Leaf kernels may allocate additional packing buffers.

For balanced Apple inputs with dimensions between 2048 and 8192 and a partial 32-row or 32-column output tile, COB can pad to multiples of 64 before dispatch. At 8191³ the three padding buffers add 768 MiB. Allocation failure selects the unpadded path. The optional Apple SME leaf handles complete 64-column half-product panels. [Apple crossover measurements](audits/2026-11-25-strassen-sme.md)

On Linux, recursive Winograd handles compact square inputs from 3072 to 5120 in steps of 128. It adds 25% to the one-level temporary workspace and retries one level if allocation fails. With a BLAS provider, two levels can also handle rectangles with a shortest dimension of at least 1024, harmonic mean of at least 2048 and aspect ratio of at most 8. The core makes 49 BLAS calls. With aligned Winograd the terminal dimensions align to 16 rows, 16 columns and eight depth elements. Up to three more calls complete the depth and output fringes.

OpenBLAS needs its development headers and `openblas.pc`. Each eligible subproduct checks the current OpenBLAS thread count. For MKL use the LP64 interface, sequential and core shared libraries. For ArmPL use the sequential LP64 shared library. Set `COB_BLAS_ROOT` for a nonstandard prefix, or supply `MKLROOT` or `ARMPL_DIR`. Make the shared libraries available to the loader at runtime. COB uses its native kernel if the C or Fortran SGEMM symbols resolve to a different BLAS provider.

## Hopcroft–Kerr

HK15 splits the output rows in two and the columns and reduction dimension in three. Each child dimension must be between 768 and 1280 inclusive, and the largest may be at most 1.25 times the smallest. The regular option requires exact divisions. The ragged option also uses the largest divisible core. On eligible inputs HK15 takes precedence over BLAS-backed Winograd. In native Linux builds with fused Strassen, the HK15 child crossover is a harmonic mean of 960. The ordinary fused crossover is 1024. [Coefficient table and exact verifier](../third_party/fast-matmul)

The public implementation checks geometry, strides and overlap before considering the transformations. It switches to classical multiplication for zero rows or columns, or when the largest row or column maximum in an operand is over four times its smallest. At least three quarters of each A row and B column must reach one eighth of that operand's global maximum. This guarantees overlap at half or more of the reduction positions in every output dot. To identify strongly biased dense inputs, COB samples 16 rows, 16 columns and 64 shared depth positions. It selects the classical route when four sampled products have absolute cosine similarity above one half. Sampling can miss an unusual group of rows or columns. HK workspace allocation failure also selects the classical route.

For nonzero input magnitudes outside [2⁻⁴⁸, 2⁴⁸], COB can exactly rescale each finite operand by a power of two if the nonzero exponents in each span at most 24 powers and `K × maxabs(A) × maxabs(B)` is at most half the largest finite binary32 value. Otherwise it accumulates in binary64. Nonfinite inputs and failure to allocate scaling workspace also use binary64. The numerical effect of ordinary FP32 rounding depends on the scale and conditioning of the dot products. [Full numerical and timing study](audits/2026-11-25-ranking/hk15-stability/README.md)

To exercise the regular and ragged public paths and verify the integer identity, run these checks from the repository root.

```sh
python3 tests/check_hk15.py
python3 tests/check_hk15_ragged.py
(cd third_party/fast-matmul && python3 verify.py)
```
