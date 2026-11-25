# Earlier single-thread FP32 results

COB beat the fastest tested independent implementation on six fresh tiny layouts on Apple M5 Max and on three Hopcroft–Kerr rectangles on AMD EPYC 9V74. A later Winograd experiment also won both fresh odd layouts on AMD. The results depend on CPU, geometry and build. A 49-row Apple control and a later Intel release replay both favored independent libraries.

## Apple M5 Max

All dimensions below are output rows × output columns × reduction depth. Padding is measured in FP32 elements per row. Times cover warmed public calls on one thread. Speedups are medians of matched peer-time/COB-time ratios. The macOS scheduler chose the core.

| Kernel and measured revision | Observation | Fastest relevant comparison |
| --- | --- | --- |
| SME, 17–32 rows. Patch on `6abc921`, integrated at `8863a0e` | At the pre-existing 32³ target, COB took 65.6 ns versus 76.2 ns for its predecessor. | Cached LIBXSMM 2.1.0 took 87.4 ns. In a separate four-buffer test, COB took 64.5 ns versus 82.9 ns for a direct prepared LIBXSMM kernel. |
| SME, 33–48 rows. Patch on `8863a0e`, integrated at `b3be729` | Across six prospectively selected neighboring layouts, geometric-mean speedups were 1.44× versus its predecessor and 1.36× versus cached LIBXSMM. | COB won 10/10 paired rounds on each held-out layout against each comparator, including direct prepared LIBXSMM with four rotating buffer sets. The 49 × 35 × 39, +1 control took 363.9 ns versus 331.6 ns for cached LIBXSMM, a loss in all ten rounds. |
| One-level fused square, `2209681` | At the preregistered compact 9344³ square, COB led five independent implementations by paired median. | Speedups were 1.174× versus MaxMulSK v5.1 (6/6 pairs), 1.253× versus Accelerate (5/6), 1.322× versus MpGEMM (5/6), 1.821× versus serial ArmPL (6/6) and 18.591× versus OpenBLAS (6/6). |

The two small panels each used two seeds, five rounds per seed and 32,768 calls per observation. The first kernel passed 5,375,540 FP64 output checks and the second passed 22,642,778. The large comparison ran six individual calls per method pairing across two seeds. Each pairing ran in separate processes, so the COB median varies by peer. All six implementations passed full-output FP64 checks on both seeds at 9344³, with maximum absolute error 0.00083324 against a 0.002 limit. [Small results, including all controls](m5-small.csv) · [Large results by peer](m5-large.csv)

## Native Linux runs

- **Standalone Hopcroft–Kerr, integrated public entry.** With COB computing its own leaf products, the integrated 15-product algorithm at `9df006c` beat the fastest independent library on one repeated and two new rectangles in 18/18 paired rounds per CPU. Geometric-mean speedups were 1.120× on AMD EPYC 9V74 and 1.164× on Arm Neoverse-N2. On AMD at 2112 × 3216 × 3144, +1, COB took 337.5 ms versus 387.9 ms for OpenBLAS. [Public-entry benchmark](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45659244461)
- **Two-level aligned Winograd, experimental public-entry build.** On AMD EPYC 7763, OpenBLAS-backed COB won at 2093 × 3211 × 3127, +9 (404.0 ms versus 444.4 ms for direct OpenBLAS). oneMKL-backed COB won at 1879 × 3437 × 3041, +3 (375.8 ms versus 417.0 ms for direct AOCL). A fixed oneMKL-backed choice selected on the earlier shapes also beat the best independent peer in 6/6 pairs on each of the three new layouts, including the divisible control. [Experimental build and results](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45672997916)
- **Ordinary linked release.** A separate replay linked the public CMake archive at `518e7d7` directly into the timed executable. The predefined aligned OpenBLAS build on Neoverse-N2 beat the best independent peer on two new odd layouts by 1.051× and 1.038× (6/6 each). The predefined aligned oneMKL build on Intel Xeon 6975P-C recorded 0.954× (0/6) and 0.996× (2/6). This release replay ran on Intel and Arm. The AMD Winograd timings above come from the earlier experimental build. [Linked release, tests and timings](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45682774110)
