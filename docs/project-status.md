# Project Status

Last updated: 2026-05-26

## Goal

The project goal is still ambitious: become the fastest open-source
single-threaded matrix multiplication software in this repo's chosen scope.

The chosen scope is intentionally narrow:

- CPU-only, single-threaded execution.
- FP32 GEMM.
- Row-major `A`, `B`, and `C`.
- No transposes.
- `C = A * B`, equivalent to `alpha = 1` and `beta = 0`.
- Apple Silicon first, with AMX and SME2.1 paths when available.

## Where It Landed

The current implementation is in a strong publishable state for the scoped
claim:

COB is fastest among the tested licensed/open-source baselines for
single-thread FP32 row-major SGEMM on Apple Silicon, in the routed shape
ranges.

That is the claim supported by the current audit docs, including the final
scoped audit note at `docs/audits/2026-05-10-claim-audit.md`. It is not a
universal "fastest matmul software" claim. Accelerate is proprietary and still
wins some small or pack-overhead-heavy cases. MpGEMM is source-available with
unclear licensing in the local scan and remains a calibration target; it still
wins a small number of one-shot `m = 64` large-`K` shapes.

## What Shipped

- One-shot row-major FP32 SGEMM.
- Public packed-`B` API for reused right-hand matrices.
- Public packed-`A` plus packed-`B` API for fully prepacked repeated-use
  workloads.
- Apple Silicon AMX kernels and route-specific packing paths.
- Apple Silicon SME2.1 direct-`B`, packed-`B`, and skinny/reuse routes.
- ARM64 NEON and scalar fallback paths.
- Correctness coverage for 135 GEMM shapes.
- A May 10 clean rebuild and claim-audit snapshot for the scoped routed suites.
- Route-aware benchmarks, grid sweeps, gap reports, and heatmap generation.
- A paired A/B benchmark harness with median ratio, bootstrap confidence
  interval, sign-test p-value, split-half holdout, checksum guards, and
  CV-driven repeat extension.
- Claim-audit tooling that records hardware context, route-aware CSVs, gap
  reports, sanity warnings, and summary output.

The most important performance wins came from:

- AMX packed-`B` routing and blocking.
- SME packed-`B` and direct-`B` tuple-load paths.
- Skinny SME `B`-reuse generalization for `m = 64/96/128`.
- `m = 64, k = 512` threshold lowering and mid-width extensions.
- `m = 64` high-`K` SME N-chunking for exact `n = 2560, k >= 12288` and
  `3584 <= n < 4096`.
- `m = 64` SME reuse and direct-NC routing for the `n = 2560..5120`
  gap band.
- B-pack prefetch tuning.
- Packed-B large-square and `m = 384` blocking.
- Small-A packed-B B-panel traversal.
- Medium SME direct routes in the `n = 1280..1472` band.
- Public packed-AB support and packed-AB traversal tuning.

## Current Limits

- The full "fastest fastest" goal is not proven.
- The remaining serious speed gap is narrower after the May 26 m64 route pass,
  but still not globally closed: some `m = 64` large-`K` shapes still need
  fresh calibration against MpGEMM-style runs and licensed/open-source baselines.
- Earlier C-level probes did not close that gap: B-panel-first traversal,
  prefetch locality changes, epilogue branch hoisting, broad compiler unrolling,
  `-mcpu=native`, K16 unroll pragmas, and prefetch-boundary branch splitting
  were neutral, noisy, or regressive.
- The next speed step is likely not another dispatch gate. It is probably an
  original fixed-shape SME kernel or hand-scheduled assembly for the `m = 64`
  large-`K` path.

## Future Work

1. Re-run the m64 large-`K` calibration grid after the May 26 route changes,
   then write an original fixed-shape SME kernel only for the gaps that remain.
   Keep `64x2112x7168` and the high-K `n = 1024` family on the short list.
2. Use hardware counters before accepting more exact gates. The prior counter
   evidence pointed at dispatch/scheduling pressure, not a simple memory-wait
   problem.
3. Re-audit external baselines when publishing a claim. Include BLIS, OpenBLAS,
   BLASFEO, Eigen, LIBXSMM, Rust `matrixmultiply`, `coral-aarch64`,
   `tract-linalg`, KleidiAI, and any newly available source-inspectable
   Apple Silicon packages.
4. Keep MpGEMM separate unless its license becomes clear. If it becomes
   clearly open-source, rerun same-contract benchmarks and treat the remaining
   `m = 64` gaps as release blockers.
5. Investigate 64-column contiguous public packed-B layout only as a versioned
   ABI change. It may enable tuple loads in more public packed-B paths, but it
   should not be mixed into a small tuning pass.
6. Keep the single-translation-unit layout unless the A/B harness is redesigned.
   The current methodology relies on compiling the same source twice with
   different flags and symbol names.
7. Compress old timeline material only after the reusable lessons are preserved
   in `docs/optimization-design-rules.md`.

## Useful Entry Points

- `README.md`: build, benchmark, API, and comparison overview.
- `docs/claims.md`: current claim boundary and audit recipe.
- `docs/optimization-design-rules.md`: accepted rules and rejected paths.
- `docs/optimization-timeline.md`: detailed chronological tuning record.
- `tools/paired_ab_bench.sh`: candidate-vs-baseline paired A/B harness.
- `tools/claim_audit.sh`: reproducible audit bundle generator.
