# Large ragged matrix multiplication against restored single-thread libraries

Three new matrix views and three repeated views tested the previously observed Arm lead of standalone Hopcroft–Kerr SGEMM. Two released standalones share the same native compiler flags. Four COB Winograd builds use serial BLAS leaves. Two experimental HK15 variants send each of the fifteen core products to a serial vendor library. Independently developed libraries and retained plans compete on the same inputs. [Native jobs](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45715020807) · [Frozen experiment](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/7f413444ef2c907a265275b749de4fc49e1daf96/tools/world-rank/leaf-confirm) · [Released source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/f322a9ab58454762563b97323fdc5f29016cc39f)

**Neoverse-N2** missed the registered full-field confirmation rule for standalone HK15. Its paired geometric mean against the fastest other admitted method was **0.994×** on repeats and **0.989×** on new views.

**Intel(R) Xeon(R) 6975P-C** missed the registered full-field confirmation rule for standalone HK15. Its paired geometric mean against the fastest other admitted method was **0.802×** on repeats and **0.799×** on new views.

On N2, standalone HK15 was 1.106× as fast as its earlier standalone implementation on repeats and 1.103× on fresh views. With the missing methods restored, OpenBLAS-backed Winograd led 5 of six target views. Standalone HK15 led 1.

## Neoverse-N2

CPU 0 ran each complete multiplication on one pinned Linux task. The recorded numerical and route checks admitted 112 of 119 layout–method combinations from 17 enrolled implementations. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45715020807/job/139558833248)

| M × N × K | Cohort | Standalone HK15, ms | Fastest admitted method, ms | Fastest rival for standalone | Standalone/rival speed ratio | Pair wins |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 2024 × 3081 × 3005 | repeat | 632.79 | COB Winograd + OpenBLAS (unaligned) (624.90) | COB Winograd + OpenBLAS (unaligned) | 0.989× | 0/6 |
| 2153 × 3007 × 3159 | repeat | 688.16 | COB Winograd + OpenBLAS (aligned) (679.52) | COB Winograd + OpenBLAS (aligned) | 0.992× | 2/6 |
| 1949 × 3227 × 3023 | repeat | 637.09 | COB Winograd + OpenBLAS (aligned) (635.65) | COB Winograd + OpenBLAS (aligned) | 1.002× | 3/6 |
| 2039 × 2969 × 3367 | anchor | 696.41 | COB Winograd + OpenBLAS (aligned) (676.95) | COB Winograd + OpenBLAS (aligned) | 0.973× | 0/6 |
| 2018 × 3094 × 2953 | fresh | 615.63 | Standalone HK15 (615.63) | COB Winograd + OpenBLAS (aligned) | 1.002× | 3/6 |
| 2227 × 3213 × 3278 | fresh | 793.47 | COB Winograd + OpenBLAS (aligned) (777.46) | COB Winograd + OpenBLAS (aligned) | 0.984× | 0/6 |
| 1863 × 3488 × 2928 | fresh | 640.40 | COB Winograd + OpenBLAS (unaligned) (628.53) | COB Winograd + OpenBLAS (unaligned) | 0.981× | 0/6 |

| Candidate | Cohort | vs earlier standalone | vs released vendor-backed Winograd | vs direct libraries | vs fastest independent one-shot | vs entire qualified field | Registered full-field gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Standalone HK15 | repeat | 1.106× | 0.994× | 1.162× | 1.031× | 0.994× | missed |
| Standalone HK15 | fresh | 1.103× | 0.989× | 1.165× | 1.028× | 0.989× | missed |
| HK15 + OpenBLAS | repeat | 1.065× | 0.956× | 1.120× | 0.992× | 0.956× | missed |
| HK15 + OpenBLAS | fresh | 1.059× | 0.949× | 1.120× | 0.987× | 0.949× | missed |
| HK15 + ArmPL | repeat | 1.057× | 0.949× | 1.112× | 0.985× | 0.949× | missed |
| HK15 + ArmPL | fresh | 1.057× | 0.947× | 1.119× | 0.986× | 0.948× | missed |

Maximum admitted absolute errors against serial FP64 were Standalone HK15 0.000466853, HK15 + OpenBLAS 0.000237961, HK15 + ArmPL 0.000912878. The acceptance limit was 0.002. Exclusions are recorded for Arm Compute Library on all seven views.

Standalone HK15 beat the fastest independent one-shot rival, FastMMW 768, in all 36 paired calls across the two cohorts. One fresh view led by 1.014×, below the 1.02× per-view hurdle. The restored COB Winograd configurations led five of the six raw target medians.

Arm Compute Library passed its installation check. On the hosted worker, initialization left four Linux tasks despite reporting a serial scheduler. All seven Arm Compute Library rows failed the one-task criterion. BLIS completed all seven views.

The separately compiled magnitude guard reproduced the complete standalone output in 36 of 36 warm pairs. Unguarded/guarded elapsed-time ratios were 0.992× on repeats and 0.991× on the fresh cohort.

| Exact-input audit on 1931 × 2903 × 2881 | Tested patterns | Exact on every cell | Patterns exceeding 0.002 | Nonfinite outputs |
| --- | ---: | ---: | ---: | ---: |
| Direct OpenBLAS | 19 | 19 | 0 | 0 |
| Earlier standalone | 19 | 6 | 7 | 5,600,860 |
| Standalone HK15 | 19 | 7 | 8 | 5,595,068 |
| COB Winograd + OpenBLAS (unaligned) | 19 | 11 | 3 | 5,241,750 |
| COB Winograd + OpenBLAS (aligned) | 19 | 11 | 3 | 5,184,000 |
| COB Winograd + ArmPL (unaligned) | 19 | 10 | 6 | 5,241,750 |
| COB Winograd + ArmPL (aligned) | 19 | 10 | 6 | 5,184,000 |
| HK15 + OpenBLAS | 19 | 8 | 5 | 3,732,620 |
| HK15 + ArmPL | 19 | 7 | 8 | 3,732,620 |
| HK15 with trial guard | 19 | 19 | 0 | 0 |

The trial guard and direct OpenBLAS returned exact answers on these 19 patterns. The unguarded HK15 exceeded the 0.002 threshold on eight patterns and produced nonfinite outputs on some finite inputs. These observations limit the earlier four-fixture performance claims. Separate studies found additional finite input patterns outside this guard’s coverage.

## Intel(R) Xeon(R) 6975P-C

CPU 0 ran each complete multiplication on one pinned Linux task. The recorded numerical and route checks admitted 145 of 147 layout–method combinations from 21 enrolled implementations. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45715020807/job/139558833328)

| M × N × K | Cohort | Standalone HK15, ms | Fastest admitted method, ms | Fastest rival for standalone | Standalone/rival speed ratio | Pair wins |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 2024 × 3081 × 3005 | repeat | 228.25 | OpenBLAS (176.51) | OpenBLAS | 0.777× | 0/6 |
| 2153 × 3007 × 3159 | repeat | 234.97 | OpenBLAS (193.63) | OpenBLAS | 0.824× | 0/6 |
| 1949 × 3227 × 3023 | repeat | 227.46 | OpenBLAS (182.48) | OpenBLAS | 0.806× | 0/6 |
| 2039 × 2969 × 3367 | anchor | 252.64 | OpenBLAS (194.51) | OpenBLAS | 0.772× | 0/6 |
| 2018 × 3094 × 2953 | fresh | 223.23 | OpenBLAS (175.68) | OpenBLAS | 0.782× | 0/6 |
| 2227 × 3213 × 3278 | fresh | 285.95 | OpenBLAS (224.41) | OpenBLAS | 0.797× | 0/6 |
| 1863 × 3488 × 2928 | fresh | 220.20 | oneMKL explicit retained (178.00) | oneMKL explicit retained | 0.818× | 0/6 |

| Candidate | Cohort | vs earlier standalone | vs released vendor-backed Winograd | vs direct libraries | vs fastest independent one-shot | vs entire qualified field | Registered full-field gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Standalone HK15 | repeat | 0.997× | 0.875× | 0.802× | 0.802× | 0.802× | missed |
| Standalone HK15 | fresh | 0.982× | 0.864× | 0.798× | 0.797× | 0.799× | missed |
| HK15 + OpenBLAS | repeat | 1.118× | 0.974× | 0.895× | 0.895× | 0.895× | missed |
| HK15 + OpenBLAS | fresh | 1.101× | 0.968× | 0.892× | 0.894× | 0.893× | missed |
| HK15 + oneMKL | repeat | 1.141× | 0.998× | 0.917× | 0.917× | 0.917× | missed |
| HK15 + oneMKL | fresh | 1.128× | 0.991× | 0.919× | 0.917× | 0.920× | missed |

Maximum admitted absolute errors against serial FP64 were Standalone HK15 0.000466853, HK15 + OpenBLAS 0.000241172, HK15 + oneMKL 0.000517715. The acceptance limit was 0.002. Exclusions are recorded for BLASFEO on repeat-row-and-column/repeat-all-fringes.

BLASFEO exited on a segmentation fault for the repeated row-and-column and all-fringes views. It competed on the other five.

The successful oneMKL explicit handle calls recorded their actual execution modes as oneMKL explicit one-shot (explicit-fallback), oneMKL explicit retained (explicit-fallback). The native status distinguishes generated kernels from oneMKL's supplied callable fallback.

The separately compiled magnitude guard reproduced the complete standalone output in 36 of 36 warm pairs. Unguarded/guarded elapsed-time ratios were 0.977× on repeats and 0.975× on the fresh cohort.

| Exact-input audit on 1931 × 2903 × 2881 | Tested patterns | Exact on every cell | Patterns exceeding 0.002 | Nonfinite outputs |
| --- | ---: | ---: | ---: | ---: |
| Direct OpenBLAS | 19 | 19 | 0 | 0 |
| Earlier standalone | 19 | 6 | 7 | 5,600,860 |
| Standalone HK15 | 19 | 7 | 8 | 5,595,068 |
| COB Winograd + OpenBLAS (unaligned) | 19 | 11 | 3 | 5,241,750 |
| COB Winograd + OpenBLAS (aligned) | 19 | 11 | 3 | 5,184,000 |
| COB Winograd + oneMKL (unaligned) | 19 | 11 | 5 | 5,241,750 |
| COB Winograd + oneMKL (aligned) | 19 | 11 | 5 | 5,184,000 |
| HK15 + OpenBLAS | 19 | 8 | 5 | 3,732,620 |
| HK15 + oneMKL | 19 | 8 | 5 | 3,732,620 |
| HK15 with trial guard | 19 | 19 | 0 | 0 |

The trial guard and direct OpenBLAS returned exact answers on these 19 patterns. The unguarded HK15 exceeded the 0.002 threshold on eight patterns and produced nonfinite outputs on some finite inputs. These observations limit the earlier four-fixture performance claims. Separate studies found additional finite input patterns outside this guard’s coverage.

## Reproduce the comparison

The native jobs used fresh processes, one full warm-up and one timed multiplication. Each of three rounds was run with each of two random seeds. The fastest rival is chosen by its median elapsed time on a view. Reported speed ratios are the medians of six same-round rival/contender times. Cohort ratios equally weight the three views. First-call timings, caller CPU time, all retained plans and numerical exclusions are in the raw CSVs. The older historical anchor carries zero weight in both cohorts.

Run `python3 -B score.py aarch64` and `python3 -B score.py x86_64` inside the extracted archive. A separately implemented verifier runs as `python3 -B independent/replay.py aarch64 arm-check.json`, with the same command for x86_64. The manifest covers the four-fixture and warm timing records, reference and binary hashes, detailed leaf proofs, post-score diagnostics and both independent reports.
