# Project Status

Last updated: 2026-07-12

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

The implementation is elite on selected routed shapes, but the previous broad
fastest claim is suspended. A July current-head re-audit found a confirmed
OpenBLAS loss at `64^3`, several proprietary Accelerate wins, and multiple
noisy but material gaps to current MIT-licensed MpGEMM on its stock `m = 64`
portfolio. The May claim audit remains useful historical evidence, not current
publication support.

## What Shipped

- One-shot row-major FP32 SGEMM.
- Public packed-`B` API for reused right-hand matrices.
- Public packed-`A` plus packed-`B` API for fully prepacked repeated-use
  workloads.
- Apple Silicon AMX kernels and route-specific packing paths.
- Apple Silicon SME2.1 direct-`B`, packed-`B`, and skinny/reuse routes.
- ARM64 NEON and scalar fallback paths.
- Correctness coverage for 645 GEMM test shapes, including test-only lowered
  Strassen crossover and a cancellation-heavy quadrant pattern.
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
- Broad one-shot SME pack-and-reuse for `M = 512..896`, `N = 1024..1280`, and
  `K >= 3072`. A 12-shape repeat-31 portfolio improved by `1.076x` geometric
  mean against the previous source, with every shape median winning; six
  representatives also beat Accelerate in 184/186 pairs.
- One-level Strassen for large balanced contiguous problems. It replaces eight
  half-size products with seven and delegates those products to the existing
  AMX scheduler. The final cooled repeat-5 portfolio improved by `1.181x`
  geometric mean across six shapes, with medians from `1.114x` to `1.233x`
  and every paired sample winning. The machine was thermally noisy, so the
  bootstrap lower bounds, all above `1.08x`, are more useful than any single
  peak ratio. Full-output maximum error stayed between `0.00066` and
  `0.00105`, inside the existing `0.002` correctness contract.
- A lower exact-square Strassen crossover at 5632. The cooled repeat-15 source
  A/B run measured `1.1296x` median with bootstrap95 `[1.0803x, 1.1813x]`;
  full-output max error was `0.000667572`. Same-contract spot checks measured
  COB at `1769 GF/s`, current MpGEMM at `1642 GF/s`, Accelerate at `1549 GF/s`,
  and OpenBLAS at `894 GF/s` on `5632^3`. MpGEMM was timed in an isolated
  process because its current assembly violates the platform ABI.
- Public packed-AB support and packed-AB traversal tuning.
- A guarded 64-row SME subpanel route for strided source-B views. It packs a
  192-512-column B slab once and reuses it across four 16-row groups, enabling
  the multi-threaded consumer to avoid duplicate external B reads. The SME
  capability cache is now race-free for concurrent first use.

## Current Limits

- The full "fastest fastest" goal is not proven.
- The narrower licensed/open-source claim is also not currently proven.
- MpGEMM is now MIT licensed and must be included in future open-source audits.
- A bounded hardware-guided search over 11 `16x64` and six native-layout
  `32x32` SME schedules found no stable winner on the exercised packed-B reuse
  routes. All variants were correct, but discovery gains either disappeared or
  regressed guard shapes; two other m64 blockers stayed on unchanged fallback
  code.
- AMX trapped with `SIGILL` in the tested SM+ZA-live state and with ZA left
  live after `SMSTOP` on this M5. AMX set/clear passed only after both states
  were disabled, closing the tested dual-engine overlap designs on the current
  hardware and OS.
- Pack-native Strassen removed intermediate row-major materialization and
  repacking while retaining the same three-buffer workspace, but was neutral at
  `5632^3` in cooled paired validation. Keep the current row-major
  linear-combination schedule.
- Three full-route m64 ownership redesigns were correct but substantially
  slower: no-copy AMX source-B panel reuse, direct-source-B SME `32x32`, and a
  fused-pack SME `32x32` one-panel scratch design. Keep the current `16x64`
  SME direct and packed-B-reuse routes.
- Same-process COB/MpGEMM interleaving crashed during the July 12 audit. Use
  isolated alternating processes until that interaction is understood.
- The May 27 focused MpGEMM calibration appeared to clear the old `m = 64`
  gaps, but the July current-head audit supersedes that result for publication.
- Earlier C-level probes did not close that gap: B-panel-first traversal,
  prefetch locality changes, epilogue branch hoisting, broad compiler unrolling,
  `-mcpu=native`, K16 unroll pragmas, and prefetch-boundary branch splitting
  were neutral, noisy, or regressive.
- The newest cooled target sweep still needs broad re-audit against
  Accelerate and source-available competitors. The exact `512x1216x3072` row is
  neutral in the latest paired Accelerate confirmation, not a current confirmed
  loss. A July 9 AMX source-B panel-reuse schedule closed the former
  `512x1152x2048` gap: paired source A/B measured `1.1613x`, and the follow-up
  Accelerate/COB median was `0.9598x`. The same schedule improved
  `512x896x2048` by `1.1315x` and `512x1280x2048` by `1.0864x` against the
  previous source. It also improved `384x896x2048` by `1.2294x`,
  `384x1280x2048` by `1.1598x`, and `768x896x2048` by `1.0943x`. The cleaner
  remaining high-K target is
  `2048x512x4096`; after narrowing the high-K SME reuse route so this exact
  row falls back to AMX, paired Accelerate reruns measured about `1.07x`
  Accelerate/COB, materially better than the earlier `1.1101x` gap but still
  not closed. Exact `512x512x2048` and `512x512x4096` moved the other
  direction: repeat-151 source A/B now favors the SME medium direct route at
  B/A medians `1.0752x` and `1.0767x`, so both rows are narrow direct-route
  exceptions. Exact `1536x768x4096` and `2048x768x4096` now also fall back to
  AMX because paired A/B favored that route over SME reuse.
- The July 9 medium high-K family shows that a broad route can still matter
  when it changes the packing/compute ownership instead of merely moving a
  threshold. Outside that family, the next speed step is probably a deeper
  kernel, layout, or algorithm change backed by Time Profiler or
  hardware-counter evidence. Earlier high-K SME traces put most samples in
  `cob_sgemm_16x64_sme_from_packed_b64_tuple`, not the first source-B packing
  pass. Follow-up probes rejected scalar A packing, B-panel-major reuse
  traversal, weak/noisy high-K m-blocking, two-A-panel `16x32` reuse, full-B
  single-store packing, fixed-`KC=1024` specialization, `restrict`-only codegen
  changes, AMX compute prefetching, m512 B-pack outside AMX mode, and another
  m512 384-column chunk retest. The next useful candidate likely needs a deeper
  compute schedule or packed-layout change than these C-level probes. A direct
  cost split now shows `512x1152x2048` packed-B compute near Accelerate while
  one-shot loses the B-pack setup cost; exact 288-column chunks,
  panel-immediate reuse, and fused AMX pack-plus-first-tile helpers did not
  convert that into a source win until the July 9 no-copy source-B panel reuse
  schedule changed the cache traversal and reversed the Accelerate comparison.
  Exact row-wise chunk packing, B-pack prefetch
  retunes, and a four-row unrolled panel packer also regressed or stayed
  neutral. CPU Counter traces on `2048x512x4096` showed COB and Accelerate with
  similar CPU pipeline ratios, so this does not look like a simple
  front-end/codegen cleanup.

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
4. Treat MpGEMM as an in-scope release blocker. Build an isolated-process paired
   harness and rerun its FP32 stock shapes before tuning exact routes.
5. Investigate 64-column contiguous public packed-B layout only as a versioned
   ABI change. It may enable tuple loads in more public packed-B paths, but it
   should not be mixed into a small tuning pass.
6. Keep the single-translation-unit layout unless the A/B harness is redesigned.
   The current methodology relies on compiling the same source twice with
   different flags and symbol names.
7. Compress old timeline material only after the reusable lessons are preserved
   in `docs/optimization-design-rules.md`.
8. Test recursive or lower-workspace Strassen only if the schedule reduces
   total traffic beyond direct packed-operand formation, which was neutral.
   Keep cancellation-heavy accuracy and memory pressure as hard gates.

## Useful Entry Points

- `README.md`: build, benchmark, API, and comparison overview.
- `docs/claims.md`: current claim boundary and audit recipe.
- `docs/optimization-design-rules.md`: accepted rules and rejected paths.
- `docs/optimization-timeline.md`: detailed chronological tuning record.
- `tools/paired_ab_bench.sh`: candidate-vs-baseline paired A/B harness.
- `tools/claim_audit.sh`: reproducible audit bundle generator.
