# Single-thread FP32 benchmark results

COB's regular public entry with sequential OpenBLAS subproducts won both registered odd layouts on an Arm Neoverse N2. Direct oneMKL beat every released build on both Intel Xeon 6975P-C holdouts. The same library can win on one CPU and lose on another for identical matrices.

| CPU | Output rows × columns × inner dimension | COB, ms | Fastest independent one-shot entry, ms | COB paired wins |
| --- | --- | ---: | --- | ---: |
| Arm Neoverse N2 | 2039 × 2969 × 3367 | 666.75 | FastMMW 700.01 | 6/6 |
| Arm Neoverse N2 | 1973 × 3319 × 2897 | 621.91 | FastMMW 647.45 | 6/6 |
| Intel Xeon 6975P-C | 2039 × 2969 × 3367 | 233.14 | oneMKL 221.63 | 0/6 |
| Intel Xeon 6975P-C | 1973 × 3319 × 2897 | 209.39 | oneMKL 208.76 | 2/6 |

The two layouts use eleven and seven padding floats per row, respectively. Times are medians of six warm calls to multiply ordinary FP32 arrays, including packing, allocation and recombination. COB used the provider chosen before the run, OpenBLAS on Arm and oneMKL on Intel. The full comparison includes a previous layout, a divisible control, earlier prototypes, direct vendor libraries and separate results for a retained oneDNN plan. The benchmark checked 4,415,300,232 output values against FP64. [Complete Linux results and offline verifier](linux/README.md) · [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45682774110)

## Apple and AMD measurements

The release history includes standalone kernels for small Apple matrices and large AMD rectangles. The Apple small-kernel study compared the public call against both a cached LIBXSMM entry and a direct call to its prepared kernel. The AMD studies compared the standalone fifteen-product schedule and BLAS-backed schedules separately. Each record identifies the version, competitors and control shapes. [Earlier measured results](earlier/README.md)

A subsequent M5 Max experiment improved the 49 × 35 × 39 case from 393.9 ns to 325.7 ns. Cached LIBXSMM took 353.9 ns. One mixed-scale fixture exceeded the registered 0.002 absolute-error limit in both the candidate and its predecessor, with identical output bits. The experiment did not qualify for release. [Results, patch and failing fixture](m5-experiment/README.md)

Checking the input magnitudes and switching to double accumulation fixed that failing value, from 0.00283442 to 0.00088130 absolute error. The successor passed all 350 stronger numerical fixtures. It improved on the released kernel by 1.175× across eight fresh ordinary layouts, but beat both the fastest independent public entry and a direct prepared LIBXSMM call on only four. The performance rule required six. [Follow-up results, numerical criteria and replay](m5-numerical/README.md)

## Medium rectangular products

A standalone candidate packed A across output rows and read B in place. It lost all five fresh warm-call comparisons against the release on each of Intel Xeon Platinum 8573C and Arm Neoverse N2. Its geometric-mean speed relative to the release was 0.775× on Intel and 0.980× on Arm. All 576 full-output validations passed. On the Intel 49 × 2861 × 157 layout with eleven floats of row padding, the released implementation reached 0.679× oneMKL's speed. [Timings, checks and offline verifier](linux-medium/README.md) · [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45686443679)

## Next large-matrix experiment

An experimental patch writes two terminal Winograd products directly into their output quadrants, removing fourteen separate addition or subtraction passes across a two-level multiply. The patch passed local Linux Arm64 correctness checks with serial OpenBLAS. Its speed has yet to be measured. The registered performance comparison covers two previously losing Intel layouts and three follow-up layouts against both the released code and direct oneMKL. [Research patch, correctness checks and timing protocol](signed-winograd/README.md)

## Reproduction

The production checkout for these Linux results is [b218bef](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b218beff1eaf8fd606d0934405cdc1e28ecdfe15). Its library code matches [518e7d7](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/518e7d77ea3d82d029f31dc10a3a705b42a7638b). The only changes between them correct two test fixtures. All ten host/build combinations produced byte-identical libraries before and after the corrections. The benchmark definitions, worker sources, vendor recipes and workflow are pinned at [7885da1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank).

To build the aligned OpenBLAS configuration on native Linux with OpenBLAS development files installed, run these commands from the production checkout. The hosted runs also selected CPU-specific compiler tuning and used a pinned OpenBLAS build. [Exact build commands](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank/release-public/build-native.sh)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON \
  -DCOB_ENABLE_ALIGNED_WINOGRAD=ON -DCOB_BLAS_LEAF=OPENBLAS
cmake --build build --parallel
OPENBLAS_NUM_THREADS=1 ctest --test-dir build --output-on-failure
```

For the measured Intel provider, use `-DCOB_BLAS_LEAF=MKL` with sequential LP64 oneMKL. [Provider discovery and other build options](../../../README.md)
