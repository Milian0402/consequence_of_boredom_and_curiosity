# COB GEMM

A fast, experimental, single-threaded matrix multiplication library.

COB focuses on one job: multiplying FP32 row-major matrices on a CPU as fast
as possible, especially on Apple Silicon. It is not a full BLAS replacement.
The deliberately narrow contract leaves more room for specialized AMX, SME,
and packing strategies.

## M5 Max benchmark

On one Apple M5 Max, COB had the fastest median one-thread, one-shot FP32 time on six prespecified large square and near-square products against Apple Accelerate, MpGEMM and MaxMulSK v5.1. Geometric mean paired speedups were 1.067× over previous COB, 1.067× over MaxMulSK, 1.197× over MpGEMM and 1.209× over Accelerate. MaxMulSK led on the 4096-square control. All times include packing and allocation. The public multiply entry point selects the new kernel automatically in supported builds.

The [November 25 audit](docs/audits/2026-11-25-strassen-sme.md) gives the dimensions, controls and accuracy checks. The [July crossover study](docs/audits/2026-07-12-square-crossover.md) records earlier results at 5632 square.

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
configurations. On Apple Silicon, the suite currently checks 657 shapes in each
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
- Large balanced inputs use one level of Strassen, reducing eight half-size
  products to seven.

Eligible exact square matrices switch to Strassen at `5632`. Other eligible
balanced matrices switch at `6144`. The route requires contiguous `A` and `B`,
dimensions divisible by 64, and a largest-to-smallest dimension ratio no
greater than `4:3`.

On Apple Silicon, balanced inputs with all dimensions between 2048 and 8192
and a partial 32-row or 32-column output tile are copied into a shape rounded
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
multiplication implementation inside this narrow contract. The current benchmark covers large balanced matrices on one M5 Max. Coverage across other dimensions and processors remains open.

## Optional Apple SME2 Strassen leaf

On a native Apple arm64 build, Make and CMake probe the compiler for the single-threaded MaxMulSK v5.1 leaf adapted for COB’s strided Strassen subproducts. The dispatcher checks the CPU feature and streaming vector length before using the leaf on full 64-column panels. The existing kernel handles other shapes and CPUs.

Use `make COB_STRASSEN_SME=OFF` or configure CMake with `-DCOB_STRASSEN_SME=OFF` to disable it. Setting either option to `ON` requires a compatible compiler. Make supports a separate leaf compiler through `COB_SME_CC`. Upstream provenance, its Apache-2.0 license and its copyright notice accompany the adapted file in [third_party/maxmulsk](third_party/maxmulsk).

## License

COB is MIT-licensed. The optional MaxMulSK-derived leaf is Apache-2.0-licensed. See the [COB license](LICENSE) and [MaxMulSK license and notice](third_party/maxmulsk).
