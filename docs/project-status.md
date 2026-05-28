# Project Status

Last updated: 2026-05-28

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
unclear licensing in the local scan and remains a calibration target; the
latest focused FP32 `row_sgemm` calibration found no same-contract gaps on its
stock skinny rows, while its FP16 and int8 rows remain out of scope.

## What Shipped

- One-shot row-major FP32 SGEMM.
- Public packed-`B` API for reused right-hand matrices.
- Public packed-`A` plus packed-`B` API for fully prepacked repeated-use
  workloads.
- Apple Silicon AMX kernels and route-specific packing paths.
- Apple Silicon SME2.1 direct-`B`, packed-`B`, and skinny/reuse routes.
- ARM64 NEON and scalar fallback paths.
- Correctness coverage for 636 GEMM test shapes.
- A May 10 clean rebuild and claim-audit snapshot for the scoped routed suites.
- Route-aware benchmarks, grid sweeps, gap reports, and heatmap generation.
- Benchmark controls for forced iterations, cooled repeats, route isolation,
  and macOS benchmark-thread QoS.
- A paired A/B benchmark harness with median ratio, bootstrap confidence
  interval, sign-test p-value, split-half holdout, checksum guards, and
  CV-driven repeat extension.
- A direct paired Accelerate harness for checking proprietary one-shot gaps
  without relying on separate-process medians.
- Xcode `xctrace` CPU Counter and Time Profiler XML summarizers for non-root
  profiling runs.
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
- One-shot SME source-`B` reuse routes for exact `512x1216x3072` and for
  selected `k = 4096`, `n = 512/768` rows up to `m = 1280`, plus exact
  `1536x512x4096`.
- Public packed-AB support and packed-AB traversal tuning.

## Current Limits

- The full "fastest fastest" goal is not proven.
- The May 27 focused MpGEMM FP32 `row_sgemm` calibration cleared the previous
  apparent `m = 64` stock-shape gaps when COB uses forced benchmark iterations.
  This improves the calibration story but does not prove the full "fastest
  fastest" goal across new shapes, libraries, hardware, or non-FP32 contracts.
- Earlier C-level probes did not close that gap: B-panel-first traversal,
  prefetch locality changes, epilogue branch hoisting, broad compiler unrolling,
  `-mcpu=native`, K16 unroll pragmas, and prefetch-boundary branch splitting
  were neutral, noisy, or regressive.
- The newest cooled target sweep still needs broad re-audit against
  Accelerate and source-available competitors. The exact `512x1216x3072` row is
  neutral in the latest paired Accelerate confirmation, not a current confirmed
  loss. The clearest remaining proprietary baseline target is
  `2048x512x4096`; after narrowing the high-K SME reuse route so this exact
  row falls back to AMX, a May 28 paired Accelerate rerun measured
  Accelerate/COB median `1.0695x` with bootstrap95 `[1.0688,1.0822]`. That is
  materially better than the earlier `1.1101x` gap, but still not closed.
  Exact `1536x768x4096` and `2048x768x4096` now also fall back to AMX because
  paired A/B favored that route over SME reuse.
- The next speed step is likely not another broad dispatch gate. It is probably
  a real kernel/layout change backed by Time Profiler or hardware-counter
  evidence. The latest `2048x512x4096` Time Profiler trace put most samples in
  `cob_sgemm_16x64_sme_from_packed_b64_tuple`, not the first source-B packing
  pass. Follow-up probes rejected scalar A packing, B-panel-major reuse
  traversal, weak/noisy high-K m-blocking, two-A-panel `16x32` reuse, full-B
  single-store packing, fixed-`KC=1024` specialization, and `restrict`-only
  codegen changes. The next useful candidate likely needs a deeper SME compute
  schedule or packed-layout change than these C-level probes. CPU Counter
  traces on `2048x512x4096` showed COB and Accelerate with similar CPU pipeline
  ratios, so this does not look like a simple front-end/codegen cleanup.

## Future Work

1. Re-audit the m64 large-`K` grid against licensed/open-source baselines and
   any newly source-inspectable Apple Silicon packages before publishing a
   broader claim. Keep `COB_BENCH_ITERS` available for short/noisy one-shot rows.
2. Use Time Profiler or hardware counters before accepting more exact gates.
   The latest target trace points at the reused packed-B SME compute helper,
   while cooled A/B probes rejected the obvious route changes.
3. Re-audit external baselines when publishing a claim. Include BLIS, OpenBLAS,
   BLASFEO, Eigen, LIBXSMM, Rust `matrixmultiply`, `coral-aarch64`,
   `tract-linalg`, KleidiAI, and any newly available source-inspectable
   Apple Silicon packages.
4. Keep MpGEMM separate unless its license becomes clear. If it becomes
   clearly open-source, rerun same-contract FP32 benchmarks with the focused
   calibration settings before treating any row as a release blocker.
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
