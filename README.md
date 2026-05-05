# consequence_of_boredom_and_curiosity

An experimental single-threaded matrix multiplication project.

The first target is intentionally narrow: FP32 row-major GEMM with no transpose
and `C = A * B`. The implementation is built around a packed-`B` path, because
that gives us a clean route to beat general BLAS calls when the right-hand
matrix is reused.

Goal: become the fastest open-source single-threaded matrix multiplication
software in this repo's chosen scope.

Current audited evidence: fastest among the tested licensed/open-source
baselines for single-thread FP32 row-major SGEMM on Apple Silicon, in the routed
shape ranges. See [docs/claims.md](docs/claims.md) for the audit recipe and
[docs/optimization-design-rules.md](docs/optimization-design-rules.md) for the
measurement rules and exclusions.

## Current Scope

- CPU only
- Single-threaded
- FP32 GEMM first
- Row-major only
- `alpha = 1`, `beta = 0`
- Public packed-`B` API
- Apple Silicon AMX `32x32` FP32 microkernel when available
- Apple Silicon SME2.1 packed-`B` path for larger reused-`B` cases when available
- Apple Silicon SME2.1 direct-`B` one-shot path for selected medium contiguous cases
- ARM64 NEON `8x8` microkernel when available
- Scalar reference and scalar edge path

On Apple Silicon, the AMX path is enabled by default and can be disabled with
`-DCOB_DISABLE_APPLE_AMX=1`. The AMX kernel uses a safe buffered store for
arbitrary output matrices and a faster direct-store path when `C` is 128-byte
aligned with a row stride that is a multiple of 32 floats.
Packed `B` uses a native `32`-column AMX layout on Apple Silicon and the
portable `8`-column layout elsewhere. Aligned Apple Silicon inputs use AMX for
the `A` panel transpose/pack step and direct AMX stores for both 128-byte and
64-byte aligned output strides. Larger AMX problems use an `MC` block to reuse
packed `B` panels across multiple `A` panels. On Apple Silicon with SME2.1, the
packed-`B` API can use a `16x64` SME kernel for larger reused-`B` problems, and
the one-shot path can reuse that kernel for the `512`-wide direct-`B` conflict
case. The one-shot path also has a guarded SME direct-`B` route for selected
medium contiguous cases where avoiding `B` packing is faster.

## Build

```sh
make
make test
```

Or with CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Benchmark

```sh
make bench
./build/cob_gemm_bench 128 256 512
./build/cob_gemm_bench 832x960x896
```

On macOS the benchmark also builds an Apple Accelerate comparison when the
framework is available. The benchmark allocates aligned matrices so it measures
the fastest AMX output path. Set `COB_BENCH_REPEATS` to use more benchmark
repeats for noisier large shapes, for example `COB_BENCH_REPEATS=11`.
Arguments can be square sizes (`512`) or rectangular `MxNxK` shapes
(`832x960x896`).
Set `COB_BENCH_PACK_SETUP=1` to also print the one-time packed-`B` setup cost.
Set `COB_BENCH_CSV=1` to print machine-readable benchmark rows for grid sweeps
and plotting. Set `COB_BENCH_ROUTE=1` to add the benchmark's current COB route
label to CSV or text output.

For repeated boundary sweeps, use the CSV wrapper:

```sh
sh tools/bench_grid.sh 832 896 960
COB_GRID_M="64 96 128" COB_GRID_N="1024 2048" COB_GRID_K="512 1024" sh tools/bench_grid.sh
COB_BENCH_ROUTE=1 sh tools/bench_grid.sh 832 896 960 | python3 tools/bench_gap_report.py
COB_BENCH_ROUTE=1 sh tools/bench_grid.sh 832 896 960 > /tmp/cob-grid.csv
python3 tools/bench_heatmap.py /tmp/cob-grid.csv --output /tmp/cob-grid.png
```

## Comparison Status

Current evidence is scoped to the benchmarked shape set and this repo's narrow
single-threaded FP32 row-major contract. In the routed shape ranges, COB beats
the tested licensed/open-source baselines so far: BLIS, OpenBLAS, BLASFEO,
Eigen, Rust `matrixmultiply`, `coral-aarch64`, LIBXSMM, `tract-linalg`, and
KleidiAI's comparable public one-shot path. Accelerate is still reported
separately and still leads on some small or pack-overhead-heavy cases.
Source-available projects without a clear license, such as MpGEMM, are tracked
as calibration targets rather than folded into the open-source claim.

To compare against an open-source CBLAS implementation, build the separate
external CBLAS benchmark target. For example, with a local BLIS build:

```sh
make bench-cblas \
  CBLAS_NAME=blis_apple \
  CBLAS_HEADER=blis.h \
  CBLAS_CFLAGS=-isystem/path/to/blis/include/aaplmx \
  CBLAS_LDFLAGS=/path/to/blis/lib/aaplmx/libblis.a
```

The benchmark pins common BLAS thread environment variables to `1`.

For libraries that export the standard Fortran `sgemm_` symbol, use the
Fortran-BLAS target. For example, with a local BLASFEO build:

```sh
make bench-fortran-blas \
  FORTRAN_BLAS_NAME=blasfeo \
  FORTRAN_BLAS_LDFLAGS=/path/to/blasfeo/lib/libblasfeo.a
```

## API

```c
void cob_sgemm_rowmajor(
    int m, int n, int k,
    const float* a, int lda,
    const float* b, int ldb,
    float* c, int ldc);

int cob_sgemm_pack_b(
    cob_packed_b_f32* packed,
    int k, int n,
    const float* b, int ldb);

void cob_sgemm_rowmajor_packed_b(
    int m, int n, int k,
    const float* a, int lda,
    const cob_packed_b_f32* packed_b,
    float* c, int ldc);
```

## Next Work

- Add `4x8`, `12x8`, and `16x4` NEON kernels and shape dispatch.
- Keep expanding route-aware grid sweeps before adding new dispatch gates.
- Add hardware-counter profiling when local tooling is available.
- Replace scalar edges with vector edge kernels.
- Use the CBLAS and Fortran-BLAS comparison targets to track external results.

## License

MIT. See [LICENSE](LICENSE).
