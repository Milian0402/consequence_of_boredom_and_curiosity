---
license: mit
task_categories:
- tabular-classification
language:
- en
pretty_name: Apple Silicon Single-Thread SGEMM Benchmarks
tags:
- benchmarks
- linear-algebra
- matrix-multiplication
- apple-silicon
- sgemm
- amx
- sme
---

# Apple Silicon Single-Thread SGEMM Benchmarks

This dataset contains benchmark evidence from
`consequence_of_boredom_and_curiosity`, an experimental single-threaded FP32
row-major SGEMM project for Apple Silicon.

The most interesting additions are the fresh paired A/B runs, the packed-AB
headline rows near 1.9 TFLOP/s median, and the `512x1216x3072` source-route
win. I also included the less flattering Accelerate gaps and rejected probes so
the dataset shows the actual search, not just the wins.

The scoped contract is:

- CPU-only, single-threaded execution.
- FP32 GEMM.
- Row-major `A`, `B`, and `C`.
- No transposes.
- `C = A * B`, equivalent to `alpha = 1`, `beta = 0`.
- One-shot, packed-B, and packed-A+B APIs reported separately.

## Files

- `data/benchmark_results.csv`: route-aware benchmark rows from selected May
  2026 sweeps, including the fresh headline repeat-21/iters-8 sweep.
- `data/paired_confirmations.csv`: paired A/B and paired Accelerate
  confirmations for accepted and rejected route changes.
- `data/fresh_paired_runs.csv`: parsed fresh repeat-61/iters-16 paired logs for
  current-vs-repo-head and Accelerate-vs-COB direct one-shot checks.
- `data/validation_runs.csv`: correctness/test command results.
- `data/research_attempts.csv`: accepted, mixed, and rejected optimization
  attempts that are more useful than raw benchmark rows alone.
- `data/optimization_timeline.csv`: structured parse of the full optimization
  timeline, one row per dated entry, with result labels, shape scope, summary,
  metrics snippets, source line, and full markdown details.
- `data/optimization_timeline.md`: raw source timeline copied from the repo.
- `data/mpgemm_comparison.csv`: MpGEMM FP32 `row_sgemm` calibration rows when
  locally available.
- `data/hardware_context.json`: local machine context.
- `data/schema.json`: short file descriptions.

## Claim Boundary

The safe claim is:

> COB is fastest among the tested licensed/open-source baselines for
> single-thread FP32 row-major SGEMM on Apple Silicon, in the routed shape
> ranges.

This is not a universal matrix-multiplication claim. Apple Accelerate is a
proprietary system framework and is tracked separately. MpGEMM is treated as a
source-available calibration target unless its licensing becomes clear.

The fresh May 27 additions are intentionally honest: they include the
impressive packed-AB rows near 1.9 TFLOP/s median, the strong
`512x1216x3072` current-vs-head source win, the still-open direct one-shot
Accelerate gaps, and rejected probes that explain which obvious paths failed.

The timeline export is meant to preserve the research trail: accepted route
changes, rejected probes, mixed measurements, tooling changes, and external
calibration notes are browseable without reading the whole markdown file.

## Hardware

The current local audit context is an Apple M5 Max with SME2.1 available, 8 MB
P-cluster L2, 64 KB L1d, 16 KB pages, and 128 B cache lines.

Different Apple Silicon generations need fresh measurements. Do not port route
thresholds or blocking constants without paired validation.

## Source

Source repository: <https://github.com/Milian0402/consequence_of_boredom_and_curiosity>

License: MIT.
