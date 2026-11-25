# Arm confirmation of 48-product matrix multiplication

**Rank 48 missed the prespecified confirmation rule on Neoverse-N2.** It passed 6 of the ten head-to-head tests on two fresh odd layouts. Each test required a faster paired median and at least five wins in six rounds. Both FastMMW thresholds, the ordinary COB release, direct OpenBLAS and ArmPL entered each layout. [Frozen protocol and measured source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/008b9af3617194e1119d0a09c8b69494627debc7/tools/world-rank/rank48-confirm) · [Native workflow](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45693475951/job/139484801959)

At 2333 × 3119 × 3413, COB release / OpenBLAS finished first in 810.65 ms, compared with 871.89 ms for rank 48 (0/6 paired wins for rank 48).

At 3019 × 2657 × 3347, COB release / OpenBLAS finished first in 873.43 ms, compared with 925.05 ms for rank 48 (0/6 paired wins for rank 48).

## Warm single-thread wall time

Each entry is the median of six complete FP32 calls in milliseconds. M × N × K denotes output rows × output columns × reduction depth. Padding adds the indicated number of FP32 positions to each leading dimension.

| Implementation | 2333 × 3119 × 3413, +7 | 3019 × 2657 × 3347, +9 | 3136 × 3200 × 3072, +0 |
|---|---:|---:|---:|
| Rank 48 | 871.89 | 925.05 | 1038.65 |
| Rank 49 | 887.26 | 942.81 | Excluded |
| COB release / OpenBLAS | 810.65 | 873.43 | 984.49 |
| FastMMW-768 | 848.75 | 918.60 | 981.69 |
| FastMMW-1536 | 911.78 | 981.38 | 1067.27 |
| OpenBLAS | 965.66 | 1044.84 | 1184.55 |
| ArmPL | 980.20 | 1066.69 | 1208.02 |

## Paired comparisons for rank 48

The ratios divide each peer time by the matching rank-48 time before taking the median. A value above one favors rank 48. The compact control was frozen separately from the two primary odd layouts.

| Peer | First fresh odd | Second fresh odd | Compact control |
|---|---:|---:|---:|
| Rank 49 | 1.019× (6/6) | 1.018× (6/6) | Excluded |
| COB release / OpenBLAS | 0.928× (0/6) | 0.944× (0/6) | 0.947× (0/6) |
| FastMMW-768 | 0.976× (0/6) | 0.992× (0/6) | 0.946× (0/6) |
| FastMMW-1536 | 1.048× (6/6) | 1.059× (6/6) | 1.027× (6/6) |
| OpenBLAS | 1.109× (6/6) | 1.129× (6/6) | 1.141× (6/6) |
| ArmPL | 1.132× (6/6) | 1.150× (6/6) | 1.163× (6/6) |

The strongest independent method by warm median was FastMMW-768 on all three layouts. A secondary comparison of the same matched calls put ordinary COB at 1.049× (6/6), 1.051× (6/6), 0.996× (1/6) versus FastMMW-768 on the two odd cases and the compact control, respectively.

## What was verified

The ordinary executable linked the CMake Release archive built from the original `b218bef` files with aligned two-level Winograd and a serial OpenBLAS leaf. A separate witness observed 49 actual core calls on all three layouts and the expected vendor calls for each nonempty fringe. Both independent FastMMW cutoffs passed the direct full-output fixture and native-leaf routing checks. The rank-48 and rank-49 programs and FP32 implementation matched their earlier measured revision `af33d7d`. Each produced its exact expected core/fringe leaf counts. Every registered method used one process thread pinned to logical CPU 0. [48- and 49-product schedules](https://github.com/jgdumas/plinopt/tree/a539243938203fd1d2104a820930e8902df4aad7/data)

Two random inputs, cancellation and scaled inputs were each checked against full-output FP64 per method and layout. The native run recorded 84 fixtures, 709,332,680 output comparisons and 126 warm calls. All input and padding checks passed. The maximum absolute-error limit was 0.002.

| Layout | Method | Fixture | Maximum absolute error |
|---|---|---|---:|
| compact-control | Rank 49 | cancel | 0.00273159 |

The earlier Arm screen compared rank 48 with rank 49, OpenBLAS and ArmPL. This confirmation adds a directly linked production COB and two independent FastMMW thresholds to the timed field. [Earlier native screen](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015)

[Rank-48 comparisons](scores.csv) · [Release comparisons](release-comparisons.csv) · [Individual warm calls](timings.csv) · [Full-output fixtures](accuracy.csv) · [Artifact receipt](provenance.json)

From this directory, run `python3 -B score.py .` to regenerate `scores.csv` and `decision.json` from the recorded numerical, routing and timing data.
