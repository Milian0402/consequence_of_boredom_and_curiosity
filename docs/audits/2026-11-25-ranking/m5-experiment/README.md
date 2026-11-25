# M5 Max FP32 GEMM experiment with 49–64 rows

**The candidate missed its accuracy requirement.** At 49 × 35 × 39, it brought a warm single-thread public call from 393.9 ns to 325.7 ns. Cached LIBXSMM 2.1.0 took 353.9 ns. Across eight previously untimed shapes, the candidate beat the incumbent on eight, the fastest independent public entry on seven and a direct prepared LIBXSMM call with four rotating operand sets on six. Geometric-mean speedups were 1.264×, 1.277× and 1.295×, respectively. These comparisons cover one Apple M5 Max.

The research patch adds one complete 16-row panel before the final partial panel in the Apple SME leaf. Its experimental range covers 49–64 output rows, 17–48 columns and reduction depths 16–48. Shapes where all three dimensions are multiples of 16 retain their existing route.

## Reproduce the accuracy result

On an SME2-capable Apple Silicon Mac, use CMake, Git and Apple Clang with SME2 support. From the directory containing this README, run the following against a checkout containing commit `b218beff1eaf8fd606d0934405cdc1e28ecdfe15`.

```sh
bash reproduce.sh "$(git rev-parse --show-toplevel)"
```

The script extracts the recorded baseline into a temporary directory, patches a separate copy, and builds each with the release CMake settings. It calls the ordinary row-major FP32 entry with A and B sharing the same input. The fixture is 64 × 25 × 29, input stride 32 and output stride 31, with deterministic mixed-scale input from seed 40096813.

The expected exit code is **1**, reflecting the numerical gate. Both builds produced the same 1,600 FP32 values. At zero-based row 44, column 12, the FP64 reference was 17,846.3838843 and the absolute error was 0.00283442336513. The predeclared limit was 0.002. The fixture reports a different status if the observed failure differs. The full study found this one failing value in 31,425,639 FP64 comparisons, passed four repository tests and passed all 84 protected-page checks. The experimental patch was withheld. [Failure measurements](failure.csv)

## Measurements

The warm timings covered whole row-major calls with unpacked FP32 operands and overwritten output. Each public observation timed 32,768 calls, with five rotated method orders for each of two seeds. The public field included Accelerate, OpenBLAS, ArmPL and LIBXSMM with and without last-kernel caching. Compact layouts also ran MaxMulSK and MpGEMM. The separate prepared study called the LIBXSMM JIT pointer directly, with either one or four rotating A/B/C sets. Independent thread probes observed one thread per process. The public log retains all observations, including 73 of 480 candidate, incumbent and cached-LIBXSMM observations where caller CPU time differed from wall time by over 2%. The summaries report median nanoseconds and median paired ratios. Raw timing units are seconds.

| Evidence | Public entry | Prepared LIBXSMM |
|---|---|---|
| Paired summaries | [Public summary](public-summary.csv) | [Prepared summary](prepared-summary.csv) |
| All observations | [Public timings](public-raw.csv) | [Prepared timings](prepared-raw.csv) |

The shape list and qualification thresholds were frozen before performance measurement. [Panel](panel.csv) · [Protocol](protocol.md) · [Thread checks](threads.csv) · [Experimental patch](candidate.patch)
