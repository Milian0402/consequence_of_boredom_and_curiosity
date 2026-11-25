# Rank-48 and rank-49 FP32 multiplication on Apple M5 Max

The 48-product schedule passed the 0.002 full-output accuracy limit on all four layouts. It lost every matched timing against both direct single-thread Accelerate and the ordinary COB release. The 49-product schedule failed cancellation checks on the two square layouts. On the two rectangles where both schedules qualified, rank 48 won one paired median and lost the other. The registered criterion required wins on at least three layouts and full qualification.

| M × N × K, row padding | Accelerate, ms | COB release, ms | Rank 48, ms | Qualified rank 49, ms | Rank 48 vs rank 49, median paired speedup and wins |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048 × 2048 × 2048, +0 | 9.46 | 8.92 | 36.13 | Excluded | Cancellation error 0.00222805649058 |
| 2039 × 2969 × 3367, +11 | 22.90 | 22.98 | 31.56 | 37.18 | 1.1019×, 4/6 |
| 2869 × 2423 × 3209, +3 | 24.82 | 25.89 | 33.80 | 33.74 | 0.9919×, 3/6 |
| 3328 × 3328 × 3328, +0 | 41.68 | 37.45 | 85.31 | Excluded | Cancellation error 0.00466119705869 |

The entries are six-call medians after initialization. Each speedup uses the median of the six matching per-call ratios. Accelerate won 24/24 pairs against rank 48, as did COB. The maximum observed error by method was 0.00100370374895 for rank 48, 0.00466119705869 for rank 49, 0.000282673558516 for Accelerate and 0.00011254210574 for COB.

## What the timing covered

Both block schedules used the unchanged input transforms, allocation, leaf multiplication, output recombination and odd-edge handling from benchmark commit `af33d7d4266ff27525839b6ed644929792720c76`. They used Accelerate as the single-thread leaf provider. The normal COB public entry was linked from a Release CMake build of `b218beff1eaf8fd606d0934405cdc1e28ecdfe15`, including its Apple SME leaf. The Mac port used the same four layouts, two seeds, four FP64 fixtures and counterbalanced six warm rounds as the Linux schedule experiment. [Source and numerical program provenance](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/af33d7d4266ff27525839b6ed644929792720c76/tools/world-rank/rank48)

The run performed 64 full-output fixtures, checked 452,404,256 output cells against FP64, and recorded 96 complete call timings. Both schedules passed separate exact-output, input-preservation and route checks on every layout. All 4,096 bilinear tensor coefficients were verified over the rationals for each schedule. Accelerate was explicitly set to single-thread mode. macOS process inspection reported one thread for each measured worker, and calling-thread CPU time ranged from 0.9979 to 1.0003 of wall time. The native macOS run had no fixed CPU affinity. These results apply to this M5 Max, macOS 27.2.1 and Apple Clang 21.0.0.

An initial execution identified the same two cancellation failures. The rejected worker could exit before the controller captured its thread count. The reported replay kept a rejected worker alive until the controller had collected its complete error and thread record, and corrected the external comparator label in the platform-specific summary. All programs, seeds and layouts were held fixed.

## Where the time went

A separate instrumented build split one complete multiplication into allocation, input transforms, core SGEMMs, output recombination and free/tail handling, using the previously known 2039 × 2969 × 3367 layout. The table shows medians across six warm calls for each program. The instrumentation used an exactly checkable integer fixture. Phase medians are computed separately and need not sum to the median of the whole call.

| Schedule | Input transforms, ms | Core SGEMMs, ms | Recombination, ms | Free and tails, ms | Whole profiled call, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 48 | 6.41 | 21.66 | 2.51 | 2.39 | 35.30 |
| Rank 49 | 6.41 | 23.49 | 2.43 | 2.44 | 35.18 |

The rank-48 transforms, recombination and tail handling cost about 11 ms on this fixture. Direct Accelerate completed the uninstrumented full product on the benchmark inputs in 22.90 ms.
