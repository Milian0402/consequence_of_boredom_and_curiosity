# Claim Audit

This document defines the claim this repository is allowed to make and the
evidence needed to re-audit it.

## Claim

COB is fastest among the tested licensed/open-source baselines for
single-thread FP32 row-major SGEMM on Apple Silicon, in the routed shape ranges.

This is not a universal matrix-multiplication claim. It does not cover
multi-threaded GEMM, non-FP32 types, transposed operands, arbitrary
`alpha`/`beta`, GPU kernels, or shapes outside the routed ranges.

## Contract

- CPU only.
- Single-threaded execution.
- FP32 SGEMM.
- Row-major inputs and output.
- No transposition.
- `C = A * B`, equivalent to `alpha = 1` and `beta = 0`.
- Public one-shot API and public packed-`B` API are reported separately.

## Audit Hardware

Current local audit context:

- Machine: Apple M5 Max.
- Single-thread assumption: one P-core in one P-cluster.
- P-cluster L2: 8 MB.
- L1d: 64 KB.
- Page size: 16 KB.
- Cache line: 128 B.
- SME2.1 is available.

Different Apple Silicon generations need fresh measurements. Do not port route
thresholds or blocking constants from another chip without paired validation.

## Tested Baselines

Licensed/open-source baselines tested so far:

- BLIS.
- OpenBLAS.
- BLASFEO.
- Eigen.
- Rust `matrixmultiply`.
- `coral-aarch64`.
- LIBXSMM exact `beta = 0`.
- `tract-linalg` comparable public paths.
- KleidiAI comparable public one-shot path.

Tracked separately:

- Apple Accelerate, because it is a proprietary system framework and still wins
  some small or pack-overhead-heavy cases.
- MpGEMM, because the local checkout had no license file; it remains a
  source-available calibration target and still wins some `m = 64` one-shot
  shapes.
- Packages that wrap Accelerate or ship no inspectable source.

## Required Audit Recipe

1. Build from a clean tree:

```sh
make clean
make all
make test
```

2. Run a route-aware grid for the shape family being claimed:

```sh
COB_BENCH_ROUTE=1 COB_BENCH_CSV=1 sh tools/bench_grid.sh > /tmp/cob-grid.csv
python3 tools/bench_gap_report.py /tmp/cob-grid.csv
```

3. For any candidate source change, use paired A/B:

```sh
COB_AB_REPEATS=61 COB_AB_MAX_REPEATS=101 COB_AB_CV_TARGET=2 \
  sh tools/paired_ab_bench.sh baseline/src/gemm.c src/gemm.c SHAPE...
```

4. Accept a performance change only when the median ratio, bootstrap interval,
sign test, and split-half holdout agree. If implementation CV is high but the
paired ratio is tight, prefer the paired ratio but rerun cold for small deltas.

5. Re-run correctness and build checks:

```sh
make test
make all
git diff --check
```

## Current Evidence Summary

- Correctness suite currently covers 75 GEMM shapes on Apple Silicon.
- The paired A/B harness reports median ratio, mean-log speedup, bootstrap
  confidence interval, sign-test p-value, and split-half holdout.
- Route-aware benchmarking and `tools/bench_heatmap.py` make dispatcher
  boundaries auditable.
- Recent structural wins include skinny SME B-reuse generalization,
  `m = 64, k = 512` threshold lowering, B-pack prefetching, packed-B
  large-square blocking, wide `m = 64` K-chunk tuning, and medium SME direct
  routing for selected `n = 1280..1472` shapes.

## Claim Boundaries

- Say "fastest among tested licensed/open-source baselines on the routed shape
  ranges."
- Do not say "fastest matrix multiplication software" without the scope above.
- Do not fold MpGEMM into the open-source claim unless its licensing becomes
  clear and the same-contract benchmarks are re-run.
- Do not fold Accelerate into the open-source baseline set.
