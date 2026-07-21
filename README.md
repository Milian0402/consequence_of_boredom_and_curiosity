# COB GEMM

A fast, experimental, single-threaded matrix multiplication library.

COB focuses on one job: multiplying FP32 row-major matrices on a CPU as fast
as possible, especially on Apple Silicon. It is not a full BLAS replacement.
The deliberately narrow contract leaves more room for specialized AMX, SME,
and packing strategies.

> **Current status:** COB has verified wins on selected large balanced
> multiplications on the tested Apple M5 Max. The most recent competitor audit
> found the July 12 source fastest at the exact `5632 x 5632 x 5632` shape.
> Current head includes a later Strassen recombination change, so that audit is
> historical rather than a current-head benchmark. The broader "fastest
> open-source GEMM" goal is not yet proven.

## Performance snapshot

The latest cross-implementation snapshot is one single-threaded FP32
multiplication at `5632^3` on an Apple M5 Max:

| Implementation | Throughput |
| --- | ---: |
| COB (July 12 source) | **1769 GF/s** |
| MpGEMM | 1642 GF/s |
| Apple Accelerate | 1549 GF/s |
| OpenBLAS | 894 GF/s |

These numbers are a shape-specific historical snapshot, not a universal or
current-head ranking. The competitors were measured with the best reliable
harness available for each, so they were not all collected in one perfectly
paired run. A July 13 fused-recombination change subsequently improved COB by
`1.0495x` against the audited source in a paired run, but the competitor audit
has not been repeated on that source. COB still loses to OpenBLAS, Accelerate,
and MpGEMM at other shapes.

See the [full `5632^3` audit](docs/audits/2026-07-12-square-crossover.md) for
commands, versions, accuracy, and caveats. The broader publication boundary is
kept in [docs/claims.md](docs/claims.md).

### Where COB has verified wins

The recorded wins cover large balanced FP32 multiplications: contiguous
row-major inputs with every dimension a multiple of 64, a largest-to-smallest
dimension ratio of at most 4:3, and sizes from `6144` (exact squares from
`5632`). Those inputs route through one level of Strassen on top of AMX
kernels that run at about 96% of the measured single-core matrix-unit
ceiling, so there is little room left above them.

Huge dense multiplies like these are the expensive core of real workloads:
covariance and Gram matrices in scientific computing, dense layers in
CPU-side ML experiments, and blocked solvers that reduce to GEMM. When such a
multiplication runs on one core of the audited Apple M5 Max, the July 12 COB
source finished it faster than the other implementations tested there, while
leaving the GPU and the remaining cores free for other work. The code is also
a compact single-file reference for combining AMX, SME2.1, and Strassen
techniques behind one dispatcher.

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
configurations. On Apple Silicon, the suite currently checks 648 shapes in each
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
- Apple Silicon uses a `32x32` AMX kernel and hardware-assisted packing.
- SME2.1 routes reduce packing work and reuse `B` data on selected shapes.
- Large balanced inputs use one level of Strassen, reducing eight half-size
  products to seven.

Eligible exact square matrices switch to Strassen at `5632`. Other eligible
balanced matrices switch at `6144`. The route requires contiguous `A` and `B`,
dimensions divisible by 64, and a largest-to-smallest dimension ratio no
greater than `4:3`. Everything else stays on the classical AMX/SME scheduler.

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
multiplication implementation inside this narrow contract. The current next
step is deeper M5-specific SME kernel schedule exploration, backed by paired
measurements rather than isolated peak numbers.

## License

MIT. See [LICENSE](LICENSE).
