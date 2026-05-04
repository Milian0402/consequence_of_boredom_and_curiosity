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
- ARM64 NEON `8x8` microkernel when available
- Scalar reference and scalar edge path

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
framework is available.

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

- Benchmark packed-`B` reuse separately from one-shot packing.
- Add `4x8`, `12x8`, and `16x4` NEON kernels and shape dispatch.
- Add packed-`A` panels and tune `MC`, `NC`, and `KC`.
- Replace scalar edges with vector edge kernels.
- Add BLIS/OpenBLAS/BLASFEO comparison harnesses.
