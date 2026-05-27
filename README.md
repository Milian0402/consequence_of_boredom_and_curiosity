# consequence_of_boredom_and_curiosity

An experimental single-threaded matrix multiplication project.

The first target is intentionally narrow: FP32 row-major GEMM with no transpose
and `C = A * B`. The implementation is built around a packed-`B` path, because
that gives us a clean route to beat general BLAS calls when the right-hand
matrix is reused.

Goal: become the fastest open-source single-threaded matrix multiplication
software in the world

Current audited evidence: fastest among the tested licensed/open-source
baselines for single-thread FP32 row-major SGEMM on Apple Silicon, in the routed
shape ranges. See [docs/claims.md](docs/claims.md) for the audit recipe and
[docs/optimization-design-rules.md](docs/optimization-design-rules.md) for the
measurement rules and exclusions. For the final state of the long optimization
session and the next serious work items, see
[docs/project-status.md](docs/project-status.md).

## Current Scope

- CPU only
- Single-threaded
- FP32 GEMM first
- Row-major only
- `alpha = 1`, `beta = 0`
- Public packed-`B` API
- Public packed-`A` + packed-`B` API for repeated-use workloads
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
Set `COB_BENCH_ITERS` to force multiple GEMM calls per timed sample when short
shapes are too noisy for single-call timing.
Arguments can be square sizes (`512`) or rectangular `MxNxK` shapes
(`832x960x896`).
Set `COB_BENCH_PACK_SETUP=1` to also print the one-time packed-`B` setup cost.
Set `COB_BENCH_CSV=1` to print machine-readable benchmark rows for grid sweeps
and plotting. Set `COB_BENCH_ROUTE=1` to add the benchmark's current COB route
label to CSV or text output. Set `COB_BENCH_ONLY=one-shot`, `packed`,
`packed-AB`, `pack-setup`, or `accelerate` to isolate one benchmark row, which
is useful for hardware-counter runs; `pack-setup` still requires
`COB_BENCH_PACK_SETUP=1`.
Set `COB_BENCH_HOT_SECONDS=N` with `COB_BENCH_ONLY` and one shape to run a
single row in a hot loop for profiler attachment; normal benchmark behavior is
unchanged when it is unset.

For repeated boundary sweeps, use the CSV wrapper:

```sh
sh tools/bench_grid.sh 832 896 960
COB_GRID_M="64 96 128" COB_GRID_N="1024 2048" COB_GRID_K="512 1024" sh tools/bench_grid.sh
COB_BENCH_ROUTE=1 sh tools/bench_grid.sh 832 896 960 | python3 tools/bench_gap_report.py
COB_BENCH_ROUTE=1 sh tools/bench_grid.sh 832 896 960 > /tmp/cob-grid.csv
python3 tools/bench_heatmap.py /tmp/cob-grid.csv --output /tmp/cob-grid.png
```

For candidate-vs-baseline source changes, use the paired A/B harness. It
defaults to the one-shot API; set `COB_AB_MODE=packed` for packed-`B`, or
`COB_AB_MODE=packed-AB` for the fully prepacked API:

```sh
COB_AB_REPEATS=61 COB_AB_MAX_REPEATS=101 \
  sh tools/paired_ab_bench.sh baseline/src/gemm.c src/gemm.c 512
COB_AB_MODE=packed-AB COB_AB_REPEATS=61 \
  sh tools/paired_ab_bench.sh baseline/src/gemm.c src/gemm.c 512
```

For Apple Silicon hardware-counter probes, build `mperf-stat` from
<https://github.com/tmcgilchrist/mperf>. On M5 machines, force the local `as5`
PMC database; if an upstream `mperf-stat` build cannot load that database, use
a build that honors `MPERF_KPEP_DB`:

```sh
sudo env MPERF_KPEP_DB=as5 COB_COUNTER_ONLY=one-shot sh tools/counter_probe.sh 64x4096x7168
sudo env MPERF_KPEP_DB=as5 COB_COUNTER_ONLY=packed sh tools/counter_probe.sh 512x1280x1536
```

The counter helper defaults to the remaining structural probe shapes and uses
`COB_BENCH_ONLY` so counters are not polluted by Accelerate or the other COB
benchmark rows.

If root-backed counters are unavailable, the hot-loop mode can keep one process
alive long enough for macOS `sample` from a normal terminal:

```sh
COB_BENCH_ONLY=one-shot COB_BENCH_ROUTE=1 COB_BENCH_HOT_SECONDS=20 \
  build/cob_gemm_bench 512x1024x1536
```

For MpGEMM calibration, keep the checkout outside this repo and point the helper
at it:

```sh
git clone https://github.com/MpGEMM/mpgemm /private/tmp/mpgemm_latest
sh tools/mpgemm_calibration.sh
```

Set `COB_MPGEMM_RUN_MPGEMM=0` to skip rebuilding/running MpGEMM and only
refresh COB's matching route-aware CSV.

## Comparison Status

Current evidence is scoped to the benchmarked shape set and this repo's narrow
single-threaded FP32 row-major contract. In the routed shape ranges, COB beats
the tested licensed/open-source baselines so far: BLIS, OpenBLAS, BLASFEO,
Eigen, Rust `matrixmultiply`, `coral-aarch64`, LIBXSMM, `tract-linalg`, and
KleidiAI's comparable public one-shot path. COB also has a fully prepacked
`A`+`B` path for repeated-use cases, so tract's packed-both mode is no longer a
stronger-contract gap on the sampled square calibration. Accelerate is still
reported separately and still leads on some small or pack-overhead-heavy cases.
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

int cob_sgemm_pack_a(
    cob_packed_a_f32* packed,
    int m, int k,
    const float* a, int lda);

void cob_sgemm_rowmajor_packed_b(
    int m, int n, int k,
    const float* a, int lda,
    const cob_packed_b_f32* packed_b,
    float* c, int ldc);

void cob_sgemm_rowmajor_packed_ab(
    int m, int n, int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c, int ldc);
```

## Next Work

- Re-run m64 large-`K` calibration after the latest exact `n = 2560` route
  changes, then write fixed-shape SME only for the gaps that remain.
- Use hardware counters before accepting more exact dispatch gates.
- Re-audit external baselines before publishing any broader fastest claim.
- Keep MpGEMM separate unless its license becomes clear, then rerun
  same-contract benchmarks.
- Treat any public packed-B layout change as a versioned ABI change.

## License

MIT. See [LICENSE](LICENSE).
