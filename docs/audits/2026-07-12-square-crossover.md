# 2026-07-12 Exact-Square Crossover Audit

This audit supports only the exact `5632x5632x5632` FP32 row-major one-shot
result on the local Apple M5 Max. It does not restore the broad fastest claim.

## Context

- Baseline COB commit: `d744f28`.
- Candidate change: exact-square Strassen floor `5632`; general floor `6144`.
- Hardware: Apple M5 Max, macOS Darwin 25.5.0, single benchmark thread.
- MpGEMM: commit `419d283`, MIT licensed.
- OpenBLAS: Homebrew `0.3.31_1`.
- Accelerate: system framework from the recorded macOS build.
- Correctness contract: full-output max absolute difference at most `0.002`.

## Source A/B Confirmation

Command shape:

```sh
COB_AB_REPEATS=15 COB_AB_MAX_REPEATS=15 \
  COB_AB_BOOTSTRAP_DRAWS=4000 COB_AB_COOLDOWN_US=750000 \
  COB_AB_B_FLAGS=-DCOB_SGEMM_STRASSEN1_MIN_DIM=5632 \
  sh tools/paired_ab_bench.sh src/gemm.c src/gemm.c \
  5504 5632 5760 5632x5120x5632 5120x5632x5632
```

At `5632^3`, candidate/baseline median was `1.1296x`, bootstrap95
`[1.0803x, 1.1813x]`, with 13/15 candidate wins. Split-half holdout median was
`1.1584x`. Maximum absolute error was `0.000667572`, RMS error
`0.0000586311`. `5760^3` also won at `1.0722x` median. The 5504 row and the two
rectangles were dispatch guards and did not justify broadening the lower route.

## Competitor Spot Checks

- Paired Accelerate, repeat 9 with 750 ms cooldown: COB median `1745.28 GF/s`,
  Accelerate median `1548.87 GF/s`; paired Accelerate/COB median `0.8826x`.
- OpenBLAS, repeat 7 with 750 ms cooldown in the route-aware benchmark: COB
  median `1769.42 GF/s`, OpenBLAS median `893.67 GF/s`.
- MpGEMM, seven isolated-process samples with a monotonic clock: median
  `1641.62 GF/s`, best `1649.07 GF/s`.

MpGEMM could not be safely interleaved in the same process. Its current
assembly clobbers callee-saved `d8-d15`, so the isolated harness preserved those
registers around each call. Treat its comparison as a spot check, not paired
proof.

## Decision

Accept the lower crossover only for exact contiguous squares from 5632. Keep
the general balanced-input crossover at 6144 and keep padded inputs on the
classical scheduler.
