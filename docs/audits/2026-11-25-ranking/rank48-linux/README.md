# Published 48-product and 49-product matrix multiplication schedules

**Rank 48 beat both direct Arm libraries on three of four layouts on Neoverse-N2. The best direct Intel library beat it on every layout on Xeon 6975P-C.** Both Linux processors met the narrower three-layout screen against rank 49. Both schedules multiply a 4 × 4 array of matrix blocks. They share the same experimental FP32 implementation for input transforms, workspace, leaf products, output recombination and handling of edges. Each complete call uses one CPU thread. The code is a benchmark prototype. [Measured source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/af33d7d4266ff27525839b6ed644929792720c76/tools/world-rank/rank48)

## Comparison by CPU

- **Intel(R) Xeon(R) 6975P-C.** Rank 48 passed all four numerical fixtures on every layout. Rank 48 beat rank 49 by paired median on 3 of 3 numerically comparable layouts (4 registered). This met the three-win threshold for independent confirmation. It won 0/24 paired observations against the fastest direct library on each layout. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015/job/139470869823)
- **Neoverse-N2.** Rank 48 passed all four numerical fixtures on every layout. Rank 48 beat rank 49 by paired median on 3 of 3 numerically comparable layouts (4 registered). This met the three-win threshold for independent confirmation. It won 18/24 paired observations against the fastest direct library on each layout. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015/job/139470869789)

Times are warm medians in milliseconds. Speedups are the median of the six matched peer-time/rank-48-time ratios. A dash marks a comparison excluded by numerical qualification. M × N × K denotes output rows × output columns × reduction depth, followed by row padding in floats.

| CPU | M × N × K, pad | Rank 48 ms | Rank 49 ms | Best direct method, ms | Vs rank 49 (wins/6) | Vs best direct (wins/6) |
|---|---|---:|---:|---|---:|---:|
| Intel(R) Xeon(R) 6975P-C | 2048 × 2048 × 2048, +0 | 128.92 | 129.90 | mkl-jit · 79.90 | 1.013× (3/6) | 0.620× (0/6) |
| Intel(R) Xeon(R) 6975P-C | 2039 × 2969 × 3367, +11 | 267.93 | 268.97 | mkl-jit · 189.13 | 1.004× (4/6) | 0.702× (0/6) |
| Intel(R) Xeon(R) 6975P-C | 2869 × 2423 × 3209, +3 | 291.86 | 294.07 | mkl-jit · 207.52 | 1.010× (3/6) | 0.712× (0/6) |
| Intel(R) Xeon(R) 6975P-C | 3328 × 3328 × 3328, +0 | 452.03 | — | mkl · 341.35 | — | 0.757× (0/6) |
| Neoverse-N2 | 2048 × 2048 × 2048, +0 | 442.08 | 452.88 | openblas · 332.91 | 1.025× (6/6) | 0.752× (0/6) |
| Neoverse-N2 | 2039 × 2969 × 3367, +11 | 737.67 | 751.11 | openblas · 790.99 | 1.019× (6/6) | 1.075× (6/6) |
| Neoverse-N2 | 2869 × 2423 × 3209, +3 | 804.27 | 818.67 | openblas · 875.24 | 1.023× (6/6) | 1.093× (6/6) |
| Neoverse-N2 | 3328 × 3328 × 3328, +0 | 1267.83 | — | openblas · 1421.02 | — | 1.121× (6/6) |

## Accuracy and scope

For each method and layout the worker compared every output with FP64 for two random seeds, a cancellation input and a scaled input. The absolute-error ceiling was 0.002. The separate integer fixture counted the actual vendor calls, checked every output, tested padding and checked that A and B retained their original bytes. Each algorithm made exactly 48 or 49 core calls, plus two additional vendor calls on each odd rectangle. Remaining depth terms use a direct accumulation into C.

For scalar inputs bounded by one, the largest coefficient-based upper bound on the sum of absolute product contributions to any output is 474 for rank 48 and 324 for rank 49. This describes the scale of terms that can enter recombination. The table gives the observed FP64 errors.

| CPU | Layout | Method | Fixture | Seed | Maximum absolute error |
|---|---|---|---|---:|---:|
| Intel(R) Xeon(R) 6975P-C | large-square | form49 | cancel | 35485721 | 0.00418976 |
| Neoverse-N2 | large-square | form49 | cancel | 35485721 | 0.00403069 |

Only rank 49 exceeded the measured error ceiling here, despite its smaller coefficient bound.

The native runs covered 160 full-output FP64 fixtures and 1,131,010,640 logical output comparisons. Each registered kernel used oneMKL leaves on x86-64 and OpenBLAS leaves on Arm. The direct x86 field contained oneMKL, oneMKL JIT, OpenBLAS and AOCL. The Arm field contained OpenBLAS and ArmPL. The 2039 × 2969 × 3367 case repeated an earlier release-study layout. The finite-rank screen measures only this field and these implementations. A direct-library lead required at least five wins in six matched warm rounds.

The schedules are PLinOpt’s `4x4x4_48_201-balanced` and `4x4x4_49_156`. Exact rational arithmetic independently verified all 4,096 bilinear coefficients in each. The rank-48 programs contain 197 additions/subtractions and four divisions by two across the three transforms. Rank 49 uses 156 additions/subtractions. Both implementations materialize 40 linear combinations of each input. The time includes these transforms, all products and reconstruction of C. [Upstream schedules and authors](https://github.com/jgdumas/plinopt/tree/a539243938203fd1d2104a820930e8902df4aad7/data) · [Native workflow](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015)

## Separate Apple M5 Max screen

Both block schedules also ran with single-thread Accelerate leaves on an M5 Max using the same four layouts. Rank 48 passed all 16 numerical fixtures but lost 24/24 pairs to direct Accelerate and 24/24 to the ordinary COB release at `b218bef`. Rank 49 failed cancellation at 2048³ and 3328³. Rank 48 won one of the two qualified rectangular comparisons, below the three-layout threshold. [Separate M5 measurements](m5-rank48-results.md)

[Paired comparisons](comparisons.csv) · [Warm observations](timings.csv) · [FP64 fixtures](accuracy.csv) · [Dispatch checks](routes.csv) · [Official artifact digests](provenance.json)

For an offline check from this directory, run `python3 -B replay.py`. It reproduces the paired scores from the native records and verifies both exact rational identities. [M5 full evidence](../rank48-m5/README.md)
