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

On a Xeon 6975P-C, packing one strip of B and using it across the output rows improved standalone COB on all five previously untimed layouts and all 30 paired observations. The five-layout geometric mean was 1.120× the previous implementation. COB beat the fastest independent library at 108 × 1728 × 320 and 137 × 2179 × 347. The fastest independent library beat COB on the other three. The experiment also checked AVX2 and AVX-512 builds separately and compared 69,104,464 output elements with FP64. [Per-layout times, direct competitors and verifier](intel-stream-b/README.md)

A separate test delegated whole products or all but the last few rows to a sequential BLAS. Over eight fresh eligible layouts, the whole oneMKL wrapper reached 1.100× the speed of released standalone COB and 1.002× the speed of calling the same oneMKL library directly. Moving the last one to four rows back into COB slowed down both providers on average. [Matched comparisons and single-thread checks](intel-vendor-tail/README.md)

## Removing fourteen Winograd output updates

A two-level variant uses serial BLAS to accumulate fourteen terminal products directly into C. On Neoverse N2 the OpenBLAS-backed variant had the lowest median of all eleven methods on each of five layouts. It beat the strongest independent method, FastMMW, in all twelve pairs on the two fresh odd layouts. Most of the advantage was already present in released COB.

| CPU and leaf provider | Speed relative to released COB, deep / wide | Speed relative to fastest independent entry, deep / wide |
|---|---:|---:|
| Intel Xeon 6975P-C, oneMKL | 1.0044× / 1.0048× | 0.989× / 1.015× |
| Arm Neoverse N2, OpenBLAS | 1.0016× / 1.0049× | 1.056× / 1.048× |

The two new layouts are 2083 × 2887 × 3433 with nine padding floats per row and 1933 × 3407 × 2947 with five. The Intel acceptance rule required more than 2% on both. All 480 full-output checks passed. The patch remains experimental. [Native results, raw measurements and offline verifier](signed-winograd/results-linux/README.md) · [Research patch](signed-winograd/README.md)

A separately frozen rule used two older layouts to choose the schedule for these two layouts and a compact control. It retained released COB on both CPUs. [Calibration results](host-selection/README.md)

## Testing a formula with fewer multiplications

Two published formulas partition each matrix into 4 × 4 blocks. The first computes 48 block products and needs 201 additions, subtractions or halvings. The comparison formula uses 49 block products and 156 other operations. The benchmark verified every one of the 4,096 bilinear coefficients in each formula over exact rationals. Both formulas used the same implementation for forming the operands and combining the results. [Published schedules](https://github.com/jgdumas/plinopt/tree/a539243938203fd1d2104a820930e8902df4aad7/data) · [Benchmark implementation](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/af33d7d4266ff27525839b6ed644929792720c76/tools/world-rank/rank48)

On Arm Neoverse N2, the 48-product formula beat both direct OpenBLAS and ArmPL on three of four layouts. Its paired speedups over the faster library were 1.075×, 1.093× and 1.121×. It lost the 2048-square case and lost all four comparisons against the fastest direct Intel library. The Arm screen did not include released COB or FastMMW. [Linux timings, exact identities and offline replay](rank48-linux/README.md)

On the M5 Max, the 48-product formula passed all sixteen accuracy fixtures. At 2048 square it took 36.13 ms, against 8.92 ms for released COB and 9.46 ms for single-thread Accelerate. Both rivals beat the 48-product formula in all 24 paired comparisons. The 49-product formula exceeded the accuracy limit on the cancellation inputs at both square sizes. [Apple timings, error measurements and replay](rank48-m5/README.md)

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
