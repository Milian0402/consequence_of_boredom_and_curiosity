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

## Apple M5 Max wide rectangles

A new opt-in SME kernel packs A once and computes the first 16 rows while packing each panel of B. The broad prototype improved on released COB across eight fresh wide layouts by 2.773×, with a geometric mean of 0.978× the strongest external entry. A subsequent nine-layout confirmation supported a narrower range of 24–55 rows. [Broad test and raw timings](m5-wide-experiment/README.md)

On those nine layouts, the installed library with `COB_ENABLE_APPLE_MEDIUM_WIDE=ON` delivered **4.807× the speed of the same build with the option off** and 4.672× the prior release. The weakest gain over the prior release was 1.940×. The new implementation averaged 0.972× the fastest measured external entry per layout, including prepared LIBXSMM. Its lead exceeded 1% on two layouts. MaxMulSK and MpGEMM were called directly on compact layouts in this study.

A clean reconstruction repeated all nine wins against the prior release, averaging 3.494× its speed. The fastest-peer geometric mean fell to 0.950×, and the 39 × 2304 × 160 case reached only 0.782× MpGEMM. That fell below the prespecified 0.85× floor. Two layouts cleared a 1% lead over the fastest peer in the first installed run and three different layouts did so in the rebuild. None repeated that lead. [First per-layout results](m5-wide-release/production-summary.csv) · [Clean rebuild results](m5-wide-release/clean-rebuild/production-summary.csv)

The option remains experimental and off by default. On all six padded thin holdouts, COB also beat MaxMulSK and MpGEMM when the adapters compacted the inputs and copied the result back inside every call. The paired geometric means were 3.079× against adapted MaxMulSK and 2.769× against adapted MpGEMM. [Whole-call compact adapters](m5-compact-adapters/README.md)

The installed option reproduced the default output bit for bit on 108 stress fixtures. Both configurations exceeded the 0.002 absolute-error threshold on some high-magnitude inputs. The speed result applies to the normalized benchmark inputs. The release option requires Apple SME2.1, 1024–4096 output columns and reduction depth 96–384. [Installed timings, accuracy limits and offline replay](m5-wide-release/README.md)

## Packing adjacent B panels on x86

Two Intel Xeon 6975P-C jobs tested packing four adjacent B panels before doing the same arithmetic as the released kernel. Grouped packing reached 97.1% and 96.8% of the incumbent's throughput across eight new rectangles. At 64 × 2304 × 224, the released kernel itself took 382 and 377 µs, while direct oneMKL took 315 and 313 µs. The four-panel implementation stays in the benchmark branch. BLASFEO qualified on nine of the twelve layouts per job and exited on the other three. [Per-layout timings, eligibility and replay](grouped-b-x86/README.md) · [Native benchmark](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45701814927)

## Medium rectangular products

A standalone candidate packed A across output rows and read B in place. It lost all five fresh warm-call comparisons against the release on each of Intel Xeon Platinum 8573C and Arm Neoverse N2. Its geometric-mean speed relative to the release was 0.775× on Intel and 0.980× on Arm. All 576 full-output validations passed. On the Intel 49 × 2861 × 157 layout with eleven floats of row padding, the released implementation reached 0.679× oneMKL's speed. [Timings, checks and offline verifier](linux-medium/README.md) · [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45686443679)

Packing one strip of B and reusing it across the output rows accelerated standalone COB on all five preregistered wide rectangles, on both a Xeon 6975P-C and an AMD EPYC 9V45. The same patch beat matched prior COB in all 60 paired observations across the two processors. Both native binaries selected AVX-512. The separately forced AVX2 path passed the improvement threshold on each machine as well.

| CPU | Geometric mean vs prior COB | Geometric mean vs fastest independent one-shot | Layout wins vs fastest independent one-shot |
|---|---:|---:|---:|
| Intel Xeon 6975P-C | 1.1200× | 0.935× | 2/5 |
| AMD EPYC 9V45 | 1.1201× | 0.958× | 0/5 |

The two Intel wins were at 108 × 1728 × 320 and 137 × 2179 × 347. AOCL led the independent field on all five AMD layouts. Each native comparison checked 69,104,464 output elements against FP64. [Cross-CPU comparison and verifier](stream-b/README.md) · [Intel timings](stream-b/intel/README.md) · [AMD timings](stream-b/amd/README.md)

The installed public library passed both acceptance tests on an Intel Xeon 6975P-C. With conservative x86-64 compiler settings, switching on the streamed-B option yielded **1.328×** the speed of the installed default across the five repeated wide rectangles (30/30 paired wins). With both builds tuned to the host, the gain was **1.128×**. The generic opt-in beat the fastest tested direct BLAS on two layouts, by 2.2% and 6.3%. Direct BLAS won the other three. Every one of the 112 method/layout combinations passed full-output checks. The AMD result above is for the earlier native build of the kernel. [Public-archive measurements, compiler flags and offline verifier](stream-b-release/README.md)

A separate test delegated whole products or all but the last few rows to a sequential BLAS. Across eight eligible layouts (four shared with the streamed-B panel and four additional), the whole oneMKL wrapper reached 1.100× the speed of prior standalone COB on Intel and 0.464× on AMD. It reached 1.002× the speed of calling the same oneMKL library directly on both hosts. Moving the last one to four rows into COB produced no consistently winning split across providers and CPUs. [Matched comparisons and single-thread checks](vendor-tail/README.md)

A further candidate packed A in addition to B. It lost every one of the 96 paired comparisons against the installed streamed-B kernel on eight fresh layouts across two separate Xeon 6975P-C runners, at 0.896× and 0.893× the incumbent's speed. The fastest independent method by median was the oneMKL direct-call interface on all eight layouts on both runners. At 79 × 1793 × 223 the incumbent took 361–362 µs and oneMKL took 303 µs. The tested field included verified LIBXSMM JIT kernels, both direct and retained oneDNN interfaces, OpenBLAS, AOCL and BLASFEO. A separately declared third run landed on AMD EPYC 9V45. Packed A lost all 48 paired calls against the released kernel there as well (0.901×). Among independent one-shot implementations, LIBXSMM dispatch led two fresh layouts and AOCL led six. [Full x86 comparison, including the AMD transfer](packed-a-x86/README.md)

Rounding selected row counts up to a multiple of eight before calling the host's BLAS also slowed down every genuinely padded fresh comparison on the two Xeon runners. [Matched direct calls and actual provider dimensions](row-padding/README.md)

## Cropped matrices on Intel

Two previously reported wins against direct libraries reproduced with the original aligned layouts and reversed on cropped views of the same mathematical inputs. Cropping changed the three starting addresses and leading dimensions, with all guard regions preserved.

| M × N × K | COB speed relative to fastest direct peer, aligned | COB speed relative to fastest direct peer, cropped |
|---|---:|---:|
| 108 × 1728 × 320 | 1.038× | 0.928× |
| 137 × 2179 × 347 | 1.067× | 0.936× |

On the six cropped cases, oneMKL beat the installed streamed-B kernel in all 36 paired comparisons. COB still beat its own default in all 36. On the two rows above, COB's elapsed times changed by +1.0% and +0.6% from aligned to cropped. The oneMKL direct-call interface took 9.7% and 13.4% less time. This comparison changes offsets and strides together. [Native records, views, exact routes and full FP64 replay](cropped-views/README.md)

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

A follow-up on N2 added released COB with sequential OpenBLAS and two FastMMW cutoffs. Released COB led all seven measured methods on both previously untimed odd layouts. At 2333 × 3119 × 3413 it took 810.65 ms versus 848.75 ms for FastMMW-768 and 871.89 ms for rank 48. At 3019 × 2657 × 3347 the times were 873.43, 918.60 and 925.05 ms. Both COB and FastMMW-768 beat rank 48 in all twelve paired calls. FastMMW-768 took first place on the compact control. [Arm confirmation, raw data and verifier](rank48-arm-confirmation/README.md)

On the M5 Max, the 48-product formula passed all sixteen accuracy fixtures. At 2048 square it took 36.13 ms, against 8.92 ms for released COB and 9.46 ms for single-thread Accelerate. Both rivals beat the 48-product formula in all 24 paired comparisons. The 49-product formula exceeded the accuracy limit on the cancellation inputs at both square sizes. [Apple timings, error measurements and replay](rank48-m5/README.md)

## Reproduction

The [public x86 kernel](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/commit/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6) is opt-in and uses COB's own arithmetic throughout. It dispatches to AVX2/FMA or AVX-512 at runtime. To build the static library and run the correctness tests on Linux x86-64:

```sh
cmake -S . -B build-stream -DCMAKE_BUILD_TYPE=Release -DCOB_ENABLE_X86_STREAM_B=ON
cmake --build build-stream --parallel
ctest --test-dir build-stream --output-on-failure
```

The production checkout for the large rectangular Winograd results is [b218bef](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b218beff1eaf8fd606d0934405cdc1e28ecdfe15). Its library code matches [518e7d7](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/518e7d77ea3d82d029f31dc10a3a705b42a7638b). The only changes between them correct two test fixtures. All ten host/build combinations produced byte-identical libraries before and after the corrections. The benchmark definitions, worker sources, vendor recipes and workflow are pinned at [7885da1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank).

To build the aligned OpenBLAS configuration on native Linux with OpenBLAS development files installed, run these commands from the production checkout. The hosted runs also selected CPU-specific compiler tuning and used a pinned OpenBLAS build. [Exact build commands](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank/release-public/build-native.sh)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON \
  -DCOB_ENABLE_ALIGNED_WINOGRAD=ON -DCOB_BLAS_LEAF=OPENBLAS
cmake --build build --parallel
OPENBLAS_NUM_THREADS=1 ctest --test-dir build --output-on-failure
```

For the measured Intel provider, use `-DCOB_BLAS_LEAF=MKL` with sequential LP64 oneMKL. [Provider discovery and other build options](../../../README.md)
