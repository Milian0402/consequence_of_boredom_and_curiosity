# consequence_of_boredom_and_curiosity

An experimental single-threaded matrix multiplication project.

The first target is intentionally narrow: FP32 row-major GEMM with no transpose
and `C = A * B`. The implementation is built around a packed-`B` path, because
that gives us a clean route to beat general BLAS calls when the right-hand
matrix is reused.

## Current Scope

- CPU only
- Single-threaded
- FP32 GEMM first
- Row-major only
- `alpha = 1`, `beta = 0`
- Public packed-`B` API
- Apple Silicon AMX `32x32` FP32 microkernel when available
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
packed `B` panels across multiple `A` panels.

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
```

On macOS the benchmark also builds an Apple Accelerate comparison when the
framework is available. The benchmark allocates aligned matrices so it measures
the fastest AMX output path.

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
- Tune the AMX `MC` threshold and add `KC`/`NC` blocking.
- Replace scalar edges with vector edge kernels.
- Use the CBLAS and Fortran-BLAS comparison targets to track external results.
