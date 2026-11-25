# Large ragged HK15 on native Linux

The research patch extends one level of the 15-product Hopcroft–Kerr algorithm to large matrices with dimensions that leave remainders. Five frozen layouts compared the ordinary standalone implementation with the shipped standalone algorithm and the independently qualified single-thread field. [Benchmark run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45709409988) · [Immutable benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/37ead44b9f3a144dada8a92b17bd447b138e6689/tools/world-rank/ragged-hk) · [Production source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/22d925ed5a31f7246eb2269675e9a3e85d502410)

On the Neoverse N2, ragged HK15 was **1.108×** as fast as the released standalone routine and **1.042×** as fast as the fastest admitted independent rival on the three fresh layouts. On the Intel Xeon 6975P-C, the corresponding ratios were **1.004×** and **0.888×**. Four missing vendor-backed configurations limit the N2 ranking.

## Intel(R) Xeon(R) 6975P-C

The standalone promotion rule **did not pass**. On the three fresh layouts the paired geometric mean versus the matched release was **1.004×**, with 2 of 3 layout medians above 1.00×. The slower seed-block geometric mean was 0.998×, and the lowest fresh layout median was 0.978×. The divisible control was 1.005× with identical FP32 outputs. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45709409988/job/139539760136)

| Geometry, M × N × K | Role | Ragged HK15, ms | Speedup vs release | Fastest independent rival, ms | Speedup vs that peer | Fastest overall | Overall win count vs fastest other |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| 1920 × 2880 × 2880 | repeat | 151.16 | 1.005× | oneMKL (146.09) | 0.967× | release-mkl-aligned-two | 0/6 |
| 2039 × 2969 × 3367 | repeat | 221.09 | 0.950× | oneMKL (189.88) | 0.859× | oneMKL | 0/6 |
| 2024 × 3081 × 3005 | fresh | 199.62 | 0.978× | oneMKL (171.88) | 0.859× | oneMKL | 0/6 |
| 2153 × 3007 × 3159 | fresh | 209.40 | 1.022× | oneMKL (191.80) | 0.920× | oneMKL | 0/6 |
| 1949 × 3227 × 3023 | fresh | 199.95 | 1.013× | oneMKL direct-call/JIT (177.56) | 0.885× | oneMKL direct-call/JIT | 0/6 |

The candidate won 0 of 18 paired timings against the fastest admitted rival on each fresh layout, with a geometric mean of **0.888×**. The independent one-shot field rule did not pass. The registered complete-field lead remained unestablished. Both OpenBLAS-backed COB builds were unavailable on all five layouts. BLASFEO exited on a segmentation fault on the fresh row-and-column and all-fringes layouts. Sequential oneMKL and its separate JIT interface took the three fresh wins.

Qualification admitted 73 of 85 method–layout combinations. The workers checked 1,783,915,764 output cells in the four accuracy fixtures and 2,675,873,646 output cells after timed calls. Maximum admitted absolute error was 0.00046685294 for ragged HK15 and 0.0004716591 for released HK15. The acceptance threshold was 0.002. The timed thread ran on logical CPU 0.

## Neoverse-N2

The standalone promotion rule **passed**. On the three fresh layouts the paired geometric mean versus the matched release was **1.108×**, with 3 of 3 layout medians above 1.00×. The slower seed-block geometric mean was 1.108×, and the lowest fresh layout median was 1.105×. The divisible control was 1.002× with identical FP32 outputs. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45709409988/job/139539760172)

| Geometry, M × N × K | Role | Ragged HK15, ms | Speedup vs release | Fastest independent rival, ms | Speedup vs that peer | Fastest overall | Overall win count vs fastest other |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| 1920 × 2880 × 2880 | repeat | 505.32 | 1.002× | FastMMW 768 (519.89) | 1.028× | Released standalone HK15 | 4/6 |
| 2039 × 2969 × 3367 | repeat | 689.09 | 1.078× | FastMMW 768 (707.16) | 1.027× | Ragged HK15 | 6/6 |
| 2024 × 3081 × 3005 | fresh | 620.66 | 1.105× | FastMMW 768 (649.14) | 1.046× | Ragged HK15 | 6/6 |
| 2153 × 3007 × 3159 | fresh | 677.40 | 1.108× | FastMMW 768 (698.94) | 1.033× | Ragged HK15 | 6/6 |
| 1949 × 3227 × 3023 | fresh | 628.18 | 1.112× | FastMMW 768 (660.48) | 1.049× | Ragged HK15 | 6/6 |

The candidate won 18 of 18 paired timings against the fastest admitted rival on each fresh layout, with a geometric mean of **1.042×**. The independent one-shot field rule passed. The registered complete-field lead remained unestablished. All four registered vendor-backed COB builds were unavailable on all five layouts. The candidate ranked first on all three fresh layouts among nine eligible implementations, including FastMMW-768, the fastest independent rival on each.

Qualification admitted 45 of 65 method–layout combinations. The workers checked 1,100,981,844 output cells in the four accuracy fixtures and 1,651,472,766 output cells after timed calls. Maximum admitted absolute error was 0.00046685294 for ragged HK15 and 0.0004716591 for released HK15. The acceptance threshold was 0.002. The timed thread ran on logical CPU 0.

## Released vendor-backed comparison coverage

The two public COB+OpenBLAS configurations passed CMake tests and installation on each host, then stopped before recording the worker identity. The measured post-install check requires an unreferenced Fortran symbol, `sgemm_`, for a static CBLAS path that calls `cblas_sgemm` and `openblas_get_num_threads`. A retained OpenBLAS 0.3.34 executable confirms that the static linker can omit the Fortran entry. The native logs end before identifying the first failed check. On Arm, the two COB+ArmPL configurations also reached installation and lack qualified workers. Direct OpenBLAS, direct ArmPL on Arm and both stock FastMMW configurations have separately recorded results.

The companion archive contains a tested post-study OpenBLAS linkage-check correction and its evidence under `forward-fix/`. The measured source, published archive hashes, missing field entries and scores retain their original identities.

## Reproduce the scores

The archive includes the exact benchmarked candidate patch, raw first-call and warm timings, full-matrix error records, route observations, and reference and executable identities. From its root, run `python3 -B score.py aarch64` and `python3 -B score.py x86_64`. To reproduce the independent calculation, run `python3 -B independent/replay.py aarch64 aarch64-replay.json` and repeat for `x86_64`. The two original GitHub artifact SHA-256 digests and independent reports are included.

One-thread OpenBLAS DGEMM generated 20 FP64 references on each host. Each reference passed 23 scalar spot checks. The four finite qualification fixtures and all six same-seed, same-round warm comparisons checked the whole output and its surroundings. Every timed process ran one complete warm-up and one timed call. The scope is row-major FP32 multiplication on the listed CPUs, inputs, strides and measured libraries. [Registered criteria](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/37ead44b9f3a144dada8a92b17bd447b138e6689/tools/world-rank/ragged-hk/protocol.md)

The initial run reached all 30 warm shape–round markers on each host, then failed the route-digest integrity check and artifact upload. GitHub retained zero timing archives for that run. All reported timings come from the authenticated recovery run, with the same production source, candidate patch, inputs, schedule and scoring rules. [Initial run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45706678234)
