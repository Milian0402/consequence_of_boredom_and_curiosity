# COB GEMM

COB computes **C = A × B** for row-major FP32 matrices on one CPU thread. It accepts independent row strides, overwrites C, and supports both one-shot calls and reuse of packed inputs. It has native kernels for Apple Silicon, Arm64 and x86-64.

## Quick start

Build with a C11 compiler and run the correctness tests and benchmark.

```sh
make
make test
./build/cob_gemm_bench 512
./build/cob_gemm_bench 832x960x896
```

The rectangular arguments are output rows, output columns and reduction depth. For 832 × 960 × 896, A is 832 × 896 and B is 896 × 960.

### CMake

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

CMake builds the static library target `cob_matmul`. To install its archive and public header, run `cmake --install build-release --prefix /path/to/prefix`.

## Using the API

```c
#include "cob_gemm.h"

int main(void) {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {7, 8, 9, 10, 11, 12};
    float c[4];

    cob_sgemm_rowmajor(2, 2, 3, a, 3, b, 2, c, 2);
    return c[0] == 58 && c[1] == 64 && c[2] == 139 && c[3] == 154 ? 0 : 1;
}
```

The three leading dimensions are row strides in FP32 elements. The operation uses untransposed A and B with `alpha = 1` and `beta = 0`. If B stays fixed, pack it once and reuse it across calls.

```c
cob_packed_b_f32 packed_b = {0};
if (cob_sgemm_pack_b(&packed_b, k, n, b, ldb) == 0) {
    cob_sgemm_rowmajor_packed_b(m, n, k, a, lda, &packed_b, c, ldc);
    cob_sgemm_free_packed_b(&packed_b);
}
```

Callers reusing both operands can use `cob_sgemm_pack_a` and `cob_sgemm_rowmajor_packed_ab`. Release each packed object with its matching free function. [Public C API](include/cob_gemm.h)

## Measured performance

COB wins some measured workloads on Apple, AMD and Arm. The fastest implementation changes with the CPU, matrix dimensions, input values and buffer reuse. The benchmark reports below give the tested revisions, competitors, complete-call times and numerical checks.

| Workload | What the measurement established |
|---|---|
| Neoverse N2 and Intel Xeon 6975P-C, three fresh large irregular products | Standalone HK15 reached **0.989× on Arm and 0.799× on Intel** against the fastest competing entry per shape. On Arm it was 1.103× as fast as earlier standalone COB and 1.028× as fast as the fastest independent one-shot library. A released COB configuration backed by OpenBLAS led two of the three fresh Arm shapes. [Full field and paired timings](docs/audits/2026-11-25-ranking/leaf-confirmation/README.md) |
| Apple M5 Max, six large balanced products | COB led all six original target medians against Accelerate, MpGEMM and MaxMulSK. MaxMulSK led the 4096-square control and one target on a later integrated build. [Square results](docs/audits/2026-11-25-strassen-sme.md) |
| Apple M5 Max, large irregular products with biased inputs | The guarded opt-in HK15 kernel delivered **1.20–1.30×** its former throughput across eight comparisons. All 266,354,000 exactly representable outputs matched and 469,839,504 further outputs met the 0.002 absolute tolerance. The guard reduced throughput to **0.916×** on four ordinary signed layouts. [Accuracy, timings and counterexamples](docs/audits/2026-11-25-ranking/hk15-stability/README.md) |
| Intel Xeon and AMD EPYC, five wide rectangles | Packing one strip of B at a time made standalone COB **1.12×** as fast as its predecessor on both tested CPUs. An independent library led three Intel layouts and all five AMD layouts. [Per-CPU comparison](docs/audits/2026-11-25-ranking/README.md#medium-rectangular-products) |

On the M5 Max, splitting each FP32 input into FP16 components added more work than the hardware recovered. All four variants lost every paired timing against the fastest accuracy-qualified FP32 control on six layouts. [FP16 accuracy and complete-call times](docs/audits/2026-11-25-ranking/fp16-components/README.md)

Warm one-shot timings include work performed inside the multiplication, including packing and temporary allocation. Prepared calls and first-call timings have separate results. Reported throughput uses the classical-equivalent count `2 × M × N × K` divided by elapsed time. [All results by CPU and layout](docs/audits/2026-11-25-ranking/README.md) · [Benchmarking guide](docs/benchmarking.md)

## Choosing a kernel

The default build dispatches by CPU and shape. Apple builds probe for the optional MaxMulSK-derived SME leaf. Linux builds offer opt-in Strassen, Winograd and single-thread OpenBLAS, oneMKL or ArmPL leaves. Both Hopcroft–Kerr options default to off. The build reference covers their eligibility, workspace, numerical fallbacks and tests. [Kernels and build options](docs/kernels.md)

## License

COB uses the MIT license. The optional MaxMulSK-derived leaf uses Apache-2.0, and the Hopcroft–Kerr coefficient table uses BSD-2-Clause. [COB license](LICENSE) · [MaxMulSK license and notice](third_party/maxmulsk) · [Coefficient table and license](third_party/fast-matmul)
