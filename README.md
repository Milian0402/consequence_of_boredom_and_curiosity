# COB GEMM

A fast, experimental, single-threaded matrix multiplication library.

COB focuses on one job: multiplying FP32 row-major matrices on a CPU as fast
as possible, especially on Apple Silicon. It is not a full BLAS replacement.
The deliberately narrow contract leaves more room for specialized AMX, SME,
and packing strategies.

## Rectangular and strided benchmarks

On previously unmeasured large layouts, COB improved over its preceding
single-thread release on all six Apple M5 Max cases and all seven cases
on each Linux CPU. Geometric mean speedups were 1.20× on the M5 Max,
1.13× on an AMD EPYC 9V45 and 1.15× on an Arm Neoverse-N2. The
[rectangular benchmark audit](docs/audits/2026-11-25-general-gemm.md)
gives the external-library comparisons, numerical checks and limits.

## M5 Max benchmark

On one Apple M5 Max, COB had the fastest median one-thread, one-shot FP32 time on six prespecified large square and near-square products against Apple Accelerate, MpGEMM and MaxMulSK v5.1. Geometric mean paired speedups were 1.067× over previous COB, 1.067× over MaxMulSK, 1.197× over MpGEMM and 1.209× over Accelerate. MaxMulSK led on the 4096-square control. All times include packing and allocation. The public multiply entry point selects the new kernel automatically in supported builds.

The [November 25 audit](docs/audits/2026-11-25-strassen-sme.md) gives the dimensions, controls and accuracy checks. The [July crossover study](docs/audits/2026-07-12-square-crossover.md) records earlier results at 5632 square.

## Native Linux benchmarks

COB selects AVX-512 or AVX2/FMA at runtime on supported x86-64 CPUs. On Arm64, the NEON path handles ragged output tiles and one-row products directly. On an Intel Xeon 6975P-C, the vector kernel improved on COB's preceding C kernel by 32–36× across five medium shapes. On an Arm Neoverse-N2, the ragged 383 × 385 × 389 product improved by 1.26×. The [native Linux audit](docs/audits/2026-11-25-linux-native.md) includes the comparisons against single-thread OpenBLAS.

Enable one Hopcroft–Kerr level with `COB_ENABLE_HK15=ON` in Make or CMake.
The optional route splits the output rows in two, the columns in three and
the inner dimension in three. All divisions must be exact. Each resulting
dimension must be between 768 and 1280 inclusive, and the largest may be at
most 1.25 times the smallest. The fifteen subproducts each use the configured
COB dispatcher. The required coefficient notice and exact integer verifier
are in [third_party/fast-matmul](third_party/fast-matmul). Run
`python3 tests/check_hk15.py` to test routing, fallback and FP64 accuracy.

The Linux [paired benchmark](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45657809673)
enabled both `COB_ENABLE_PORTABLE_STRASSEN` and `COB_ENABLE_RECURSIVE_STRASSEN`
for the subproducts. On the padded 2432 × 3648 × 3648 case, the fifteen-product
schedule ran 1.114× as fast as single-threaded OpenBLAS on an AMD EPYC 9V74
and 1.225× as fast on a Neoverse N2. OpenBLAS was the fastest independent
library in both comparisons. The M5 Max's existing route was faster on both
tested Apple rectangles. The Hopcroft–Kerr option defaults to off.

In native Linux builds with fused Strassen enabled, the fifteen subproducts
can start fused Strassen when the harmonic mean of their dimensions reaches
960. The ordinary fused entry retains its cutoff of 1024. On two fresh padded
rectangles this improved the standalone route by 6.5–7.8% on Neoverse N2 and
1.0–1.7% on AMD EPYC 7763. The project's Winograd schedule with OpenBLAS
leaves was faster on both rectangles on each CPU. All standalone COB output
checks passed. Two-level FastMMW exceeded the accuracy limit on the Arm
control. [Native comparisons](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45664272660)

For large products on native Linux x86-64 or Arm64, enable the one-level Strassen path with `make COB_ENABLE_PORTABLE_STRASSEN=ON` or configure CMake with `-DCOB_ENABLE_PORTABLE_STRASSEN=ON`. It uses the native single-thread kernels for each subproduct. Contiguous balanced products use the original 1536 crossover. For rectangular or strided inputs, the dispatcher also weighs the number of products saved against the workspace needed for the three temporary matrices.

Linux builds can also select a second Winograd level by enabling
`COB_ENABLE_RECURSIVE_STRASSEN=ON` together with portable Strassen in Make
or CMake. It applies to contiguous square inputs from 3072 through 5120 in
steps of 128 and adds 25% to the one-level temporary workspace. If the
larger allocation fails, COB retries the one-level schedule.

The experimental `COB_ENABLE_OPENBLAS_LEAF=ON` option in Make and CMake
uses OpenBLAS for the Strassen subproducts on native Linux x86-64 and Arm64.
Enable portable Strassen as well and install the OpenBLAS development package
with its `openblas.pc` file. Each subproduct uses OpenBLAS when its current
thread count is one. Other thread counts select COB's native kernel.

Make and CMake also accept `COB_BLAS_LEAF` with the values `NONE` (the
default), `OPENBLAS`, `MKL` on Linux x86-64, and `ARMPL` on Linux Arm64.
The existing OpenBLAS flag selects `OPENBLAS`. For Intel use an LP64
development installation providing the Intel LP64 interface, sequential and
core shared libraries. For Arm use the LP64 sequential shared library.
Supply `COB_BLAS_ROOT` for a nonstandard prefix containing the vendor
headers and libraries, or use the vendor's `MKLROOT` or `ARMPL_DIR`.
For example, configure `cmake -S . -B build -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_BLAS_LEAF=MKL`, or build and test with `make test COB_ENABLE_PORTABLE_STRASSEN=ON COB_ENABLE_RECURSIVE_STRASSEN=ON COB_BLAS_LEAF=ARMPL`.
Make the selected shared libraries available through the system loader when
running the resulting program. If these vendors' C or Fortran SGEMM symbols
resolve to another BLAS in the process, COB takes its native fallback.

With portable and recursive Strassen enabled, the selected BLAS provider
can supply two Winograd levels on large rectangular products. For example,
configure an OpenBLAS build with
`cmake -S . -B build -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON -DCOB_ENABLE_OPENBLAS_LEAF=ON`.
Set `OPENBLAS_NUM_THREADS=1` for a threaded OpenBLAS installation. The
rectangular route requires a shortest dimension of at least 1024, a harmonic
mean of at least 2048 and a longest dimension no more than eight times the
shortest. The core uses 49 BLAS calls. Fringes use up to two further calls.
Workspace is allocated inside each multiplication and capped at 256 MiB.
If HK15 is explicitly enabled, its existing eligible shapes take priority.

For these rectangular products, setting `COB_ENABLE_ALIGNED_WINOGRAD=ON`
in Make or CMake aligns the dimensions of the 49 terminal products to 16
rows, 16 columns and 8 depth elements. An additional BLAS call accumulates
the leftover depth into the leading output using beta equal to one. Up to
two overwrite calls cover disjoint right and bottom strips. This option
requires portable and recursive Strassen and a selected BLAS leaf. It is
off by default. The same workspace cap and allocation fallback apply.

### Where the change applies

One-level Strassen runs seven smaller products on eligible balanced matrices.
The new leaf uses SME for complete 64-column half-product panels. Below the
crossover or on unsupported widths, the dispatcher keeps its existing kernels.
Odd dimensions may reach the new leaf through the existing padding path.

Large dense multiplies occur in covariance and Gram matrices, dense layers in CPU-side machine learning, and blocked solvers. Reported throughput uses the classical-equivalent operation count `2*M*N*K` divided by elapsed time.

## Supported operation

| Property | Current contract |
| --- | --- |
| Operation | `C = A * B` |
| Data type | FP32 |
| Layout | Row-major |
| Threads | One |
| Transposes | None |
| BLAS scalars | `alpha = 1`, `beta = 0` |
| Main target | Apple Silicon |

COB exposes three ways to run the same multiplication:

- **One-shot:** pass normal `A` and `B` matrices. Best when both change.
- **Packed B:** pack `B` once and reuse it across many calls.
- **Packed A + B:** pack both inputs once for repeated-use workloads.

Packing rearranges matrix data into the format the hardware kernels want. It
costs time once, then avoids repeating that work on later multiplications.

## Quick start

Requirements: a C11 compiler and `make`.

```sh
make
make test
./build/cob_gemm_bench 512
```

The test command runs the normal scheduler plus lowered-threshold Strassen test
configurations. On Apple Silicon, the suite currently checks 667 shapes in each
configuration.

You can also benchmark rectangular matrices. Arguments use `M x N x K`, where
`A[M x K] * B[K x N] = C[M x N]`:

```sh
./build/cob_gemm_bench 832x960x896
COB_BENCH_REPEATS=11 COB_BENCH_ROUTE=1 \
  ./build/cob_gemm_bench 5632
```

On macOS, benchmark output includes Apple Accelerate when it is available.

### CMake

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Using the API

Include [`include/cob_gemm.h`](include/cob_gemm.h) and compile
[`src/gemm.c`](src/gemm.c) into your program.

The one-shot API looks like this:

```c
#include "cob_gemm.h"

int main(void) {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {7, 8, 9, 10, 11, 12};
    float c[4];

    /* A is 2x3, B is 3x2, and C is 2x2. */
    cob_sgemm_rowmajor(2, 2, 3, a, 3, b, 2, c, 2);
    return 0;
}
```

If `B` will be reused, pack it once. Inside a function where the matrix
dimensions and buffers are already defined:

```c
cob_packed_b_f32 packed_b = {0};

if (cob_sgemm_pack_b(&packed_b, k, n, b, ldb) == 0) {
    cob_sgemm_rowmajor_packed_b(m, n, k, a, lda, &packed_b, c, ldc);
    cob_sgemm_free_packed_b(&packed_b);
}
```

The header also contains the packed-`A` + packed-`B` API. Packed objects own
their allocated memory and must be released with their matching free function.

## How it gets fast

The public API stays simple while the dispatcher chooses a route from the
matrix shape, layout, and available hardware:

- A scalar reference path handles portability and edge cases.
- ARM64 systems can use an `8x8` NEON kernel.
- On Apple Silicon, single-row products stream the rows of `B` directly.
  Single-column products with a contiguous input vector use a NEON dot-product
  kernel. Both paths skip packing.
- Apple Silicon uses a `32x32` AMX kernel and hardware-assisted packing.
- SME2.1 routes reduce packing work and reuse `B` data on selected shapes.
  With 17–32 output rows and columns and a reduction depth of 16–48,
  the 16-lane FP32 SME kernel keeps the first twelve depth slices of `A`
  in registers. For 33–48 rows, 17–48 columns and the same depths, it
  transposes the next panel of `A` alongside two or three output tiles.
  Both routes accept independently padded row strides.
- For wide outputs with a handful of rows, Arm NEON and x86 AVX2/FMA stream
  the right-hand matrix directly. The minimum width is 256 and the minimum
  inner dimension is 64. Apple SME2.1 selects it for two to three rows, Arm
  NEON on Linux for two to eight, AVX-512 systems for two to seven and
  AVX2-only systems for two to five.
- Large balanced inputs use one level of Strassen, reducing eight half-size
  products to seven.

On Apple silicon, eligible exact square matrices switch to Strassen at `5632`,
and other eligible balanced matrices switch at `6144`. These crossovers
apply to contiguous `A` and `B`,
dimensions divisible by 64, and a largest-to-smallest dimension ratio no
greater than `4:3`.

For other eligible rectangles and strided inputs, COB applies seven products
to the aligned interior and computes the remaining rows, columns and inner
dimension directly. Its three outer scratch buffers together occupy at most
256 MiB. Leaf kernels can allocate additional packing buffers. The longest
dimension can be up to eight times the shortest. This route uses the
optional native Strassen kernels on Linux or the SME2.1 leaf on Apple.

On Apple Silicon, balanced inputs with all dimensions between 2048 and 8192
and a partial 32-row or 32-column output tile can be copied into a shape rounded
up to multiples of 64. COB applies the usual classical or Strassen crossover
to those padded dimensions and copies the active result back to the caller.
Compatible inputs are reused directly. At `8191³` the three padding buffers
add 768 MiB of temporary storage on top of the aligned kernel's workspace.
If that allocation fails, COB uses its existing unpadded implementation.

The Apple AMX path is enabled by default. Define
`COB_DISABLE_APPLE_AMX=1` at compile time to disable it. Route thresholds and
blocking constants are tuned for the tested M5 Max and need fresh measurements
on other Apple Silicon generations.

## Benchmarking and development

For routine work:

```sh
make bench
COB_BENCH_ROUTE=1 COB_BENCH_CSV=1 sh tools/bench_grid.sh 512 1024
```

Use the [benchmarking guide](docs/benchmarking.md) for paired source A/B tests,
external BLAS comparisons, grid reports, hardware counters, and profiler
commands. Performance changes should pass the paired statistical checks and
the `0.002` full-output maximum absolute error limit before landing.

Useful project notes:

- [Claim status and audit recipe](docs/claims.md)
- [Current project status](docs/project-status.md)
- [Optimization rules and rejected ideas](docs/optimization-design-rules.md)
- [Full optimization timeline](docs/optimization-timeline.md)

## Goal

The goal is still to become the fastest open-source single-threaded matrix
multiplication implementation inside this narrow contract. Measurements cover one M5 Max, one Intel Xeon 6975P-C and one Arm Neoverse-N2 runner type. Additional processors and workloads remain open.

## Optional Apple SME2 Strassen leaf

On a native Apple arm64 build, Make and CMake probe the compiler for the single-threaded MaxMulSK v5.1 leaf adapted for COB’s strided Strassen subproducts. The dispatcher checks the CPU feature and streaming vector length before using the leaf on full 64-column panels. The existing kernel handles other shapes and CPUs.

Use `make COB_STRASSEN_SME=OFF` or configure CMake with `-DCOB_STRASSEN_SME=OFF` to disable it. Setting either option to `ON` requires a compatible compiler. Make supports a separate leaf compiler through `COB_SME_CC`. Upstream provenance, its Apache-2.0 license and its copyright notice accompany the adapted file in [third_party/maxmulsk](third_party/maxmulsk).

## License

COB is MIT-licensed. The optional MaxMulSK-derived leaf is Apache-2.0-licensed. See the [COB license](LICENSE) and [MaxMulSK license and notice](third_party/maxmulsk).
