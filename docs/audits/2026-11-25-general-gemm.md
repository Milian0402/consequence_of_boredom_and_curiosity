# Generalizing single-thread FP32 GEMM

COB beat its previous best on all 20 large rectangular, irregular and strided holdout cases across three CPU families. The number of distinct Linux layouts is seven, each measured on two CPU architectures. The Apple study used six different large layouts. The largest Apple gains were 1.50× on an odd shape and 1.47× on a padded wide matrix.

| Native confirmation host | Eligible large cases faster than prior COB | Geometric mean over prior COB | Against single-thread library |
|---|---:|---:|---:|
| Apple M5 Max | 6/6 | 1.198× | 1.158× Accelerate |
| AMD EPYC 9V45 | 7/7 | 1.127× | 0.914× OpenBLAS |
| Arm Neoverse-N2 | 7/7 | 1.152× | 1.090× OpenBLAS |

Apple also led MaxMulSK v5.1 on the five compatible compact large rectangles by a 1.020× geometric mean and MpGEMM by 1.148×. MaxMulSK led COB on the 4096-square control. Accelerate led on three of the four eligible short-row cases.

GitHub assigned the screening run to an Intel Xeon 6975P-C and the large-matrix confirmation run to an AMD EPYC 9V45. The Arm screening and confirmation runs both reported Neoverse-N2. This distinction matters when attributing the x86 gains. [Discovery run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45635844429) · [Large-matrix confirmation](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45636752652)

## Two optional Linux improvements

| Variant | AMD over standalone one-level COB | AMD over one OpenBLAS call | Arm over standalone one-level COB | Arm over one OpenBLAS call |
|---|---:|---:|---:|---:|
| Seven single-thread OpenBLAS subproducts, four new rectangles | 1.113× | 1.042× | 1.015× | 1.114× |
| Two-level Strassen–Winograd with native COB kernels, two new squares | 1.094× | 1.006× | 1.117× | 1.218× |

The OpenBLAS-backed version beat an ordinary call to the same installed OpenBLAS on all four held-out rectangles on both Linux CPUs. The smallest AMD margin was 0.8%. Every timing includes the seven calls, workspace and recombination. Its result is a gain from organizing work around an existing optimized library. To use it, enable both `COB_ENABLE_PORTABLE_STRASSEN=ON` and `COB_ENABLE_OPENBLAS_LEAF=ON`, then configure OpenBLAS to use one thread. COB reads that setting and uses its native kernel when the count differs. The normal library has no OpenBLAS build dependency.

The deeper schedule reduces the work to 49 quarter-size native products. It beat one-level COB on both 3584 and 5120 squares on both Linux hosts. It beat OpenBLAS on both Arm shapes and on the 5120 AMD shape. This option requires both `COB_ENABLE_PORTABLE_STRASSEN=ON` and `COB_ENABLE_RECURSIVE_STRASSEN=ON`. It adds 25% to the outer Strassen workspace and had the largest observed FP64 error at 0.000932. The two optional configurations were timed separately.

The result is a cross-platform implementation advance for the stated operation. A universal speed ranking, a new matrix-multiplication identity and a better asymptotic exponent are outside what the measurements establish.

## Measurement scope

Each per-shape ratio is the median of six paired full-call timings in two reversed process blocks. Each family mean weights its shapes equally. The previous build was `dd73beb` on M5 Max and `377c477` on Linux, with portable Strassen enabled on both sides of the Linux comparison. Independent FP64 products checked all output elements and stride canaries. The 0.002 maximum absolute error criterion applies to the bounded input distributions used.

The addressable operation is single-thread FP32 row-major C = A × B using the one-shot entry point, including allocations and data movement. Packed-input, multithreaded and GPU operations have separate performance characteristics. The public benchmark branch contains the two Linux experiments and the build/dispatch verification. [Final build and short-row verification](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45640437598)

## Short-row replication and accuracy

On four eligible M5 inputs, the narrowed kernel averaged **1.622×** the preceding COB. On the AVX2-only AMD EPYC 7763, three eligible inputs averaged **16.409×** previous COB and **1.770×** OpenBLAS. The preceding x86 dispatcher bypassed its packed vector path for two through five rows. The 16× gain is specific to eliminating that expensive fallback. Six fresh Neoverse-N2 shapes averaged **3.318×** previous COB and **1.341×** OpenBLAS. The repeated Arm 8 × 1024 × 3072 case gained 2.027× and is shown separately. Different CPUs have different row limits, so these short-row means cover different eligible inputs. [Final native build and short-row verification](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45640437598)

All three Linux runs succeeded on both architectures. Make and CMake built the OpenBLAS option both on and off on native AVX2 and Neoverse-N2 hosts. A call-count fixture verified exactly seven OpenBLAS subproducts in a one-level eligible case at one thread and zero when configured for two threads. Every observed candidate error in the confirmation and final verification runs stayed below 0.002. Across the Apple validation, the Linux primary, two optional variants and the short-row rerun, the largest absolute error was **0.000932**. All tested canaries passed. The complete Apple Make suites passed 667 shapes in each of three configurations, and all four CMake tests passed.
