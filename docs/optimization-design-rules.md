# Optimization Design Rules

These rules summarize repeated findings from the optimization timeline. They are not permanent laws; revisit them only when the route, hardware, compiler, or benchmark harness changes enough to invalidate the old measurement.

## Measurement

- Prefer paired A/B runs over comparing medians from separate benchmark invocations.
- Treat runs with sample CV above 2% as noisy, even when the apparent median delta is large. Use `COB_AB_MAX_REPEATS` to let the paired harness collect more samples before deciding; paired speedup CV can stop a run early when common-mode noise is high but the ratio is stable.
- Check the split-half holdout line before shipping a small change. If the full-run result looks positive but the reporting half is neutral or reversed, rerun cold instead of tuning a gate from that sample.
- Prefer candidates whose bootstrap interval and sign-test p-value agree. If they disagree, rerun instead of shipping the change.
- Recheck any candidate win in a fresh process before committing a performance route.
- Use `COB_AB_A_FLAGS` / `COB_AB_B_FLAGS` for compile-time threshold or constant sweeps before forking source; it keeps A/B probes cheap and reproducible.
- Use `COB_AB_MODE=packed` for public packed-B probes and
  `COB_AB_MODE=packed-AB` for fully prepacked A+B probes; do not validate
  packed-AB route changes with separate-process medians when paired mode can
  compare them directly.
- For new dispatch gates, default to a higher acceptance bar: median speedup
  at least 3%, split-half holdout median at least 2%, and sign-test
  `p < 1e-10`. Treat smaller gates as exceptional and document why they are
  worth the added route complexity.
- Keep one-shot, packed-B, and packed-AB claims separate. More-prepacked results
  are useful ceilings but do not prove the less-prepacked public path is faster.
- Use `COB_BENCH_ONLY` or `tools/counter_probe.sh` for hardware-counter runs so
  the counter totals cover only one implementation row, not the whole comparison
  benchmark.
- On M5, split hardware-counter groups when needed: `INST_SME_ENGINE_ALU`
  cannot be mixed into the same pipeline event set.

## Dispatch

- Do not add new exact-shape gates based on a single benchmark run. Confirm with paired measurements and nearby shapes first.
- Smooth dispatch boundaries are more trustworthy than isolated winners. If a route only wins at one exact size, assume noise until proven otherwise.
- Avoid large dispatcher refactors while evaluating a hot-path change; refactor after the A/B harness can prove behavior and performance stayed stable.

## Rejected Paths

- Row-wise B traversal during packing has repeatedly reduced pack bandwidth and collapsed one-shot performance.
- Scalar or simpler A-pack fallbacks for small K have been hard regressions.
- Broad low-threshold direct SME routing has not held up. Exact 384 and 768 were accepted only after focused repeat runs.
- One-shot `n = 1024` via public packed-B or direct SME routes has generally
  not beaten the current path, except exact `512x1024x1536`, which should use
  SME direct-B.
- The low-height `m = 160/192/224/256/288/320/352/384` medium rectangles now
  use SME direct-`B` for `n = 1280/1344/1408/1472`, `768 <= k <= 1152`,
  alongside the older exact `384x1280x1536` exception. Do not broaden this
  below `m = 160`, below `k = 768`, or to `k = 2048/3072`; current guards were
  neutral/noisy or regressions.
- For `n = 1536`, use the same SME direct-`B` path only for
  `m = 160/192/224/256/288/320`, `832 <= k <= 1152`. Keep `m = 352/384`,
  `k = 768`, and `k = 1536` off this narrow edge; confirmation found mixed
  holdouts and a clear `352x1536x1024` regression, while the colder `k = 768`
  probe regressed `192..320`. Do not extend this to the upper medium
  `m = 416/448/480` band; the `n = 1536` probe regressed from `k = 1024`
  upward and was noisy even at the low-K edge.
- For `n = 1600`, use SME direct-`B` only for `m = 160/192/224/256`,
  `768 <= k <= 1152`. Keep `m >= 288`, `n = 1664`, `k < 768`, and `k = 1536`
  off this route; the narrowed repeat-301 confirmed the lower subset while
  those guards stayed neutral/noisy or behavior-identical.
- Exact `384x512x3072` may use SME direct for one-shot GEMM. Do not extend the
  rule to `n = 768` or `n = 1024` without new cold holdout proof; the current
  `n = 768` rerun lost its holdout and `n = 1024` was a hard regression.
- For exact one-shot `768x512x1536`, skip the SME packed-B conflict detour and
  let the AMX packed fallback consume the already-packed B panel. Nearby
  `n = 512` guards stayed neutral/noisy, so keep this gate exact.
- Increasing expression-level unrolling in the C SME packed-B kernel did not beat the compiler's current schedule.
- m=64 B-reuse changes are shape-sensitive; do not assume a NC/KC knob alone will close the MpGEMM gap.
- For wide `m = 64, k = 2048` B-reuse, use the tuple-prefetch pack helper.
  This improved the remaining `64x7168x2048` MpGEMM calibration gap and nearby
  `n = 4160..8192` rows. Keep the current `NC=512` and `WIDE_KC=1024`: probes
  at `WIDE_KC=768/1536/2048` and `NC=256/768` were neutral or regressive on
  the target/guard set, and the small exact `64x4160x1536` tight-NC win was too
  weak for another dispatch exception. Keep the shared pack-prefetch distance
  at `16`; distance `8` hurt the target and distance `32` hurt `8192` plus the
  existing `64x4096x7168` prefetch path.
- For wide `m = 64, n = 7168, k >= 8192` B-reuse, also use the tuple-prefetch
  pack helper. Do not broaden this to all high-K wide rows without fresh proof:
  `n = 6144` and `n = 8192` high-K guards stayed neutral/noisy.
- For exact wide `m = 64, n = 24576, k = 1536`, use the tuple-prefetch pack
  helper. Do not broaden this to the `n >= 24576` tail; `64x32768x1536`
  regressed hard under the broad rule.
- Do not add software prefetching to the packed-B reuse helper itself. A broad
  probe over m64 and 96/128-row reuse paths was neutral/noisy or worse; keep
  prefetching on selected source-B pack helpers only.
- SME streaming-B prefetch is route-specific. It helped `m = 64`, large-`K`
  skinny direct widths whose row stride is not a 512-float multiple, plus the
  exact `n = 4096` reuse path, but broad medium, m=96/128, and wide-`N`
  prefetch probes regressed or stayed noisy.
- Keep `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE=32` as the direct streaming-B
  default, but use `COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE=16` for the
  exact `n = 4096, k >= 7168` B-reuse tuple-prefetch path. Distance `64`
  regressed the target/guard set, and disabling the tuple-prefetch path also
  regressed.
- For `m = 64`, the SME skinny direct route is useful from `k >= 2048` through
  `n <= 4096`; at `k = 1536`, keep exact `n = 1024` plus the narrower
  `n >= 1408` gate because `n = 1088..1280` regressed or stayed noisy.
  Do not route exact `64x1024x1536` through one-shot packed AMX; it is a hard
  regression against the SME route.
- For `m = 64, k = 512`, the SME B-reuse route is useful from `n >= 4096` and
  at the mid-width `n = 2048/2560/3072/3584` multiples of 512. Do not lower the
  threshold for all `n >= 2048`; `n = 2112/2304` regressed hard in the broad
  threshold probe. Keep exact `n = 1024` on AMX; the SME exact probe had weak
  holdout and noisy behavior-identical guards.
- For `m = 64, k = 2048`, `KC=1024` is useful only in the low-width SME skinny
  band currently gated as `n = 1088..1280` plus `n = 1600`. Global
  `COB_SGEMM_SKINNY_SME_KC=384/768/1024` probes regressed important high-`K`
  and `n >= 2048` shapes, so keep cache-blocking changes route-specific.
- Hand-written K=2/K=4 unrolls in the prefetched m64 SME streaming-B C
  intrinsic kernel did not improve the remaining large-`K` gaps. Do not revisit
  that schedule without counter evidence or a dedicated assembly kernel.
- Splitting the m64 prefetched SME streaming-B loops into an unconditional
  prefetch main loop plus a no-prefetch tail did not improve the remaining
  large-`K` gaps. Keep the compact in-loop prefetch guard.
- For `64x2112x7168` and `64x4096x7168`, repeat counter profiles showed high
  map/dispatch stalls, near-zero `LDST_UNIT_WAITING_SME_ENGINE_MEM_DATA`, and
  no SME cross-page events. Treat the remaining gap as dispatch/scheduling
  pressure, not a simple B-memory-wait problem.
- Do not reduce m64 large-K prefetch issue rate to every fourth K iteration; it
  regressed. K4 unrolls of both prefetched kernels and from-packed B64 tuple
  kernels were neutral/noisy.
- Keep the exact `n = 4096` large-K reuse path enabled; disabling it
  hard-regressed `64x4096x7168` and `64x4096x8192`.
- Keep `64x2112x7168` on the direct streaming-B route; forcing B reuse there
  hard-regressed the exact gap shape.
- Keep `n = 1024` high-K m64 shapes on the existing direct streaming-B route
  with `KC=1024`. Exact direct-NC chunking, high-K B-reuse, and a narrowed
  `KC=512` probe either failed cold validation or regressed adjacent guards.
- Do not force exact `64x2112x7168` to the larger `KC=1024` direct route; it
  regressed the target and nearby high-K points.
- For m64 direct streaming-B, chunk the N loop at 1024 columns for
  exact `n = 2560, k >= 12288`, exact `n = 3072, k >= 4096`, and for the
  high-width, high-K band `3584 <= n < 4096, k >= 7168`. This keeps the
  current B slab hot across the four 16-row A panels without paying B-pack
  cost. Do not broaden the `n = 2560` rule to lower K: `k = 5120` had weak
  full-run support, `k = 6144` was inconsistent, and `k = 7168` missed the
  holdout-median bar. Below `k = 4096` at `n = 3072`, keep only the
  separately validated reuse route at `k = 1024`; the `k = 3072` reuse probe
  was positive but did not clear the usual holdout bar. At `n = 4096,
  k >= 7168`, keep the existing B-reuse path with `KC=512`, `NC=512`, and the
  narrower B-pack prefetch distance; direct-NC forcing regressed hard.
- For m64 one-shot B-reuse, use the reuse path for `k >= 1024, n >= 4160`;
  for `n = 4096` only below high-K (`1024 <= k < 7168`); for exact
  `64x2560x3072`; for `3072 <= n < 4096` only at `k = 1024`; and for exact
  `n = 3584` below high-K (`1024 <= k < 7168`). Do not broaden the exact
  `2560x3072` reuse gate: neighboring `2304/2432/2496/2624/2816` widths
  regressed hard. Do not route the tempting `3712x4096/5120` exact cases or
  `3712x6144`; mixed validation was unstable or regressive even after earlier
  positive samples.
- Do not route exact `64x1024x512` through the AMX skinny chunk packer; the
  existing AMX strided-B route is much faster. Do not force exact
  `64x4096x2048` onto one-shot packed AMX; B-pack cost overwhelms the compute
  win. Do not prefetch exact `64x1024x1536`; it regressed hard.
- Full-K chunks are not a substitute for a real single-store epilogue. They
  helped only tiny isolated low-width `k = 2048` cases and badly regressed
  high-`K` m64 SME skinny shapes.
- For the public packed-B API, AMX beats the current SME packed-B kernel at
  `k >= 4096`, at `n = 1152, k >= 2048`, and at exact `n = 1152, k = 1536`.
  Keep those shapes on AMX unless a future packed-B layout or kernel change is
  validated directly.
- For public packed-B `n = 1024`, use AMX at exact `m = 512, k = 1536`, at
  exact `k = 2048`, and from `k >= 3072`. The `k = 2048` `m = 512` holdout
  sign test was weaker than the higher-row results, but median and bootstrap
  holdout stayed positive.
- For public packed-B `n = 768`, use AMX at exact `k = 2048` and `k = 3072`.
  The `k = 2048` sibling needed a focused repeat-201 rerun before shipping, and
  the `k = 4096` path is already on AMX; do not broaden this without fresh
  paired evidence.
- For public packed-B `n = 512, k = 3072`, use AMX only when `m >= 1024`.
  Lower `m = 512/768` neighbors were neutral/noisy and should stay on SME.
- For public packed-B AMX, keep the `m = 384, n >= 2048, k >= 1024` case on
  the 384-row large-block schedule. The wider 512-row packed-B block makes this
  shape miss large-block B-panel reuse entirely. Do not extend this rule to
  `k = 512` or broader `m >= 512` shapes without fresh paired evidence.
- For public packed-B AMX, use the 512-row block at exact `512x1280x2048`.
  Neighboring `512x1280x1536` and `512x1344x2048` regressed under the same row
  block, so keep the exception exact.
- For public packed-B AMX with small A, use B-panel-outer traversal only in the
  proven wide regions: `m = 64, n >= 2048, k >= 1536`, and
  `m = 96/128, n >= 4096, k >= 1024`. Do not include `64x1024x1536` or the
  `m = 96/128, k = 512` low edge; those stayed neutral or regressed under the
  same traversal.
- For AMX packed full-width shapes, including one-shot and public packed-B, use
  the large-block schedule below `n = 1152` only when `n >= 768` and
  `k >= 3072`, plus the high-row `n = 512, m >= 1024, k >= 4096` band and
  exact `768x512x4096`. The lower-`k` `n = 768, k = 2048` rows,
  `n = 512, k = 3072`, `512x512x4096`, and `768x512x8192` large-block guards
  were behavior-identical/noisy or regressions and should not drive the rule.
- For one-shot AMX packed large-block high-`K` medium cases at `m >= 768`, use
  a 256-row A block for `n = 768, k >= 3072` and `n = 1024, k >= 4096`. A
  global 512-row block regressed, while the narrow 256-row block improved the
  validated audit gaps without changing public packed-B blocking.
- The one-shot 256-row AMX A-block rule also covers the audited high-`K`
  medium bands where local paired sweeps stayed positive: `k >= 4096,
  m >= 768, 512 <= n <= 1280`; `k >= 3072, m = 512, 768 <= n <= 1280`;
  `k >= 3072, m >= 1024, 1152 <= n <= 1280`; plus exact
  `768x1024x3072`. Keep it capped to the audited range.
- In one-shot AMX medium routes, high-`K` strided-B loses to packing B even
  after pack cost. Keep `m >= 512, k >= 4096` on the packed path, plus
  `n = 1152, k >= 3072`, and exact `n = 1152, k = 2048` from `m >= 512`.
- The same packed-path rule has a lower-height exception at `m = 384`: use the
  packed path for `n >= 1152, k >= 4096` and for exact `n = 1152, k >= 3072`.
  Guard `m = 384` widths below `1152` were neutral/noisy, so do not broaden it.
- For one-shot `512x1280x1536`, use the SME direct-B route. It avoids B-pack
  setup and beat the packed AMX path, while nearby `k = 1024/2048`, `m = 768`,
  `n = 1216`, and `n = 1344` guards stayed neutral/noisy.
- For one-shot `m = 512, k = 2048`, use the chunked AMX B-pack path only at
  exact `n = 896`, `1024`, `1152`, and `1280`. Neighbor widths `832`, `960`,
  `1088`, and `1216` regressed under the same chunked path, so keep this as an
  explicit-width rule unless a smoother kernel replaces it.
- For one-shot `n = 1216`, use the packed path from `k >= 3072`, and also at
  exact `k = 2048` when `m >= 768`. The `m = 512, k = 2048` neighbor and
  `k = 1536` guards are neutral/noisy; `k >= 4096` is covered by the high-`K`
  rule.
- For the one-shot `n = 512` conflict path, keep SME packed-B at `k <= 1024`,
  but use packed AMX at `k >= 2048`. The `k = 1024` AMX fallback probe was a
  regression/noise, while `k = 2048/3072` was consistently positive.
- Exception: exact one-shot `512x512x3072` should use the SME direct-B route,
  not packed AMX. Broader SME-direct probes at nearby medium shapes regressed
  or stayed noisy.
- Exception: exact one-shot `512x1024x1536` should also use SME direct-B.
  Its `k = 1024/2048`, `m = 768`, `n = 960`, and `n = 1088` guards stayed
  neutral/noisy.
- Do not use SME direct-B for high-`K` medium `n = 512/768/1024` audit gaps.
  Exact probes at `768x512x4096`, `768x768x4096`, `1024x768x4096`, and
  `768x1024x4096` regressed hard.
- Do not port cache-blocking constants from other Apple Silicon generations. On this M5 Max, per-P-cluster L2 is 8 MB, page size is 16 KB, and route-specific cache-fit probes still need paired A/B proof.
- Do not prioritize fused inline B-packing rewrites unless a profiler shows packing is the bottleneck on an in-scope licensed-baseline gap.

## External Baselines

- Treat MpGEMM as a calibration target, not a source to copy from. Its checkout has no license file in the local scan.
- Treat KleidiAI FP32 SME2 compute-only results as an upper-bound/reference. Its direct one-shot path includes both LHS and RHS packing and was not faster on the tested public one-shot shapes.
- Treat packages that wrap Accelerate or ship no inspectable source separately from licensed open-source GEMM implementations.
- If a source-available project has unclear licensing, document its numbers separately from the "open-source" claim.
- State the speed claim as "fastest among tested licensed/open-source baselines on the routed shape ranges" unless a fresh source scan and benchmark run cover a broader claim.
