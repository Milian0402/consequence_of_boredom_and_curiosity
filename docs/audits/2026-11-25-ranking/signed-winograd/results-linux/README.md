# Signed updates in single-threaded Winograd SGEMM

**The signed update missed its preregistered oneMKL speed criterion on the Intel Xeon 6975P-C.**

The signed COB build calls its serial BLAS provider to add seven products and subtract seven others directly in the destination blocks. This removes fourteen merge passes from a two-level Winograd multiplication. Each result times the complete row-major FP32 operation, including workspace allocation and recombination. [Registered protocol](../protocol.md) · [Linux run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015)

## The two new odd layouts

On each layout, the registered result requires a median original/signed time ratio above **1.02×** and at least **5 of 6** paired wins against the original COB schedule with the same provider. The two prior layouts and the new near-square layout provide separate comparisons. [Full decisions](decision.csv)

| CPU | Leaf provider | Test | Deep odd, ratio (wins) | Wide odd, ratio (wins) | Meets both thresholds |
|---|---|---|---:|---:|---|
| Intel(R) Xeon(R) 6975P-C | oneMKL | Registered | 1.004× (5/6) | 1.005× (6/6) | Missed |
| Intel(R) Xeon(R) 6975P-C | OpenBLAS | Transfer | 1.008× (5/6) | 1.011× (5/6) | Missed |
| Neoverse-N2 | ArmPL | Transfer | 0.999× (2/6) | 1.003× (5/6) | Missed |
| Neoverse-N2 | OpenBLAS | Transfer | 1.002× (5/6) | 1.005× (6/6) | Missed |

Raw milliseconds are medians of six warm complete calls. Each speedup is the median of six peer/signed time ratios paired by seed and round. A ratio above 1 favors signed COB. These two ways of taking a median can yield different quotients. The five rows below follow the registered order. [Warm measurements](timings.csv)

## Intel(R) Xeon(R) 6975P-C · oneMKL leaves

| Layout (M × N × K / pad) | Signed ms | Original ms | vs original | vs direct oneMKL | vs MKL-JIT | Best independent one-shot · ms | vs best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2039 × 2969 × 3367 / +11 | 195.43 | 194.55 | 1.000× (3/6) | 0.974× | 0.990× | openblas · 190.38 | 0.978× (0/6) |
| 1973 × 3319 × 2897 / +7 | 174.14 | 174.48 | 1.006× (5/6) | 1.017× | 1.016× | mkl-jit · 176.80 | 1.016× (4/6) |
| 2083 × 2887 × 3433 / +9 | 191.96 | 192.59 | 1.004× (5/6) | 0.990× | 0.989× | mkl-jit · 190.02 | 0.989× (0/6) |
| 1933 × 3407 × 2947 / +5 | 177.07 | 177.73 | 1.005× (6/6) | 1.015× | 1.017× | mkl · 179.82 | 1.015× (4/6) |
| 2688 × 2816 × 2752 / +0 | 180.09 | 181.08 | 1.006× (6/6) | 1.082× | 1.080× | mkl · 193.31 | 1.082× (6/6) |

Intel’s `mkl-jit` entry measures the sequential oneMKL direct-call JIT interface. All five layouts exceed its small-kernel domain and use the ordinary SGEMM fallback. [Per-method medians and first calls](method-summary.csv) · [All signed-entry comparisons](pairs.csv)

The fastest qualified one-shot entries across the full field were openblas (1/5), signed-mkl (3/5), mkl-jit (1/5). Allowing retained plans gave the same independent winners on all five layouts. [Per-layout winners](summary.csv)

The signed oneMKL build won 10/10 first calls against its original. The geometric mean of the individual first-call ratios was 1.007×. [Per-seed first calls](method-summary.csv)

## Neoverse-N2 · ArmPL leaves

| Layout (M × N × K / pad) | Signed ms | Original ms | vs original | vs direct ArmPL | Best independent one-shot · ms | vs best |
| --- | --- | --- | --- | --- | --- | --- |
| 2039 × 2969 × 3367 / +11 | 700.43 | 700.69 | 1.001× (4/6) | 1.177× | fastmmw-768 · 716.63 | 1.022× (6/6) |
| 1973 × 3319 × 2897 / +7 | 639.10 | 637.84 | 1.001× (4/6) | 1.203× | fastmmw-768 · 659.25 | 1.031× (6/6) |
| 2083 × 2887 × 3433 / +9 | 688.39 | 687.44 | 0.999× (2/6) | 1.197× | fastmmw-768 · 713.95 | 1.039× (6/6) |
| 1933 × 3407 × 2947 / +5 | 645.80 | 647.82 | 1.003× (5/6) | 1.205× | fastmmw-768 · 666.16 | 1.029× (6/6) |
| 2688 × 2816 × 2752 / +0 | 671.91 | 673.08 | 1.003× (6/6) | 1.250× | fastmmw-768 · 671.43 | 1.001× (4/6) |

ArmPL supplies the native direct comparison on this processor. [Per-method medians and first calls](method-summary.csv) · [All signed-entry comparisons](pairs.csv)

The fastest qualified one-shot entries across the full field were signed-openblas (5/5). Allowing retained plans gave the same independent winners on all five layouts. [Per-layout winners](summary.csv)

On the 5 layouts led by signed COB with OpenBLAS leaves, its paired speedup over the fastest independent one-shot implementation ranged from 0.58% to 5.98%, with 30/30 paired wins. [Signed/OpenBLAS comparisons](pairs.csv)

The signed ArmPL build won 7/10 first calls against its original. The geometric mean of the individual first-call ratios was 1.002×. [Per-seed first calls](method-summary.csv)

## Correctness and measurement

The workers used one pinned logical CPU and one process thread. Each layout had two random input seeds, 31048579 and 196173023, with three counterbalanced warm rounds each. A first call is the first multiplication in a fresh worker after library initialization. FastMMW uses external Winograd with sequential OpenBLAS. The oneDNN one-shot entry constructs its primitive for each call. Its retained-plan entry reuses a primitive. [Method roles](methods.csv) · [Per-layout coverage](coverage.csv)

Four fixtures on each layout compare every output to FP64 for the two random inputs, cancellation and scaling. Qualification requires absolute error at most 0.002 and checks finite values, input preservation and output guards. Traced builds verify the seven positive and seven negative core updates. Separate seeded output comparisons check bitwise parity between pristine COB and the build with signed updates disabled. [Signed routes](routes.csv) · [FastMMW routes](fastmmw-routes.csv) · [Pristine parity](parity.csv)

| Processor | Pinned CPU | Warm measurements | Output cells checked across field | Signed residual range | Signed input / guard changes | FP64 records failing qualification |
|---|---:|---:|---:|---:|---:|---:|
| Intel(R) Xeon(R) 6975P-C | 0 | 390 | 1,704,088,776 | -0.00087116 to 0.00069019 | 0 / 0 | 0 |
| Neoverse-N2 | 0 | 330 | 1,441,921,272 | -0.0011137 to 0.00091517 | 0 / 0 | 0 |

Run `python3 verify.py` in this directory to verify the published files and recompute the results. The run, job and artifact identifiers and their SHA-256 digests bind these measurements to the public source. [Accuracy records](accuracy.csv) · [Linked builds](builds.csv) · [Provenance](provenance.json) · [Verifier](verify.py)
