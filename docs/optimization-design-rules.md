# Optimization Design Rules

These rules summarize repeated findings from the optimization timeline. They are not permanent laws; revisit them only when the route, hardware, compiler, or benchmark harness changes enough to invalidate the old measurement.

## Measurement

- Prefer paired A/B runs over comparing medians from separate benchmark invocations.
- Treat runs with sample CV above 2% as noisy, even when the apparent median delta is large. Use `COB_AB_MAX_REPEATS` to let the paired harness collect more samples before deciding; paired speedup CV can stop a run early when common-mode noise is high but the ratio is stable.
- Check the split-half holdout line before shipping a small change. If the full-run result looks positive but the reporting half is neutral or reversed, rerun cold instead of tuning a gate from that sample.
- Prefer candidates whose bootstrap interval and sign-test p-value agree. If they disagree, rerun instead of shipping the change.
- Recheck any candidate win in a fresh process before committing a performance route.
- Use `COB_AB_A_FLAGS` / `COB_AB_B_FLAGS` for compile-time threshold or constant sweeps before forking source; it keeps A/B probes cheap and reproducible.
- Keep one-shot and packed-B claims separate. Packed-B compute-only results are useful ceilings but do not prove the public one-shot path is faster.

## Dispatch

- Do not add new exact-shape gates based on a single benchmark run. Confirm with paired measurements and nearby shapes first.
- Smooth dispatch boundaries are more trustworthy than isolated winners. If a route only wins at one exact size, assume noise until proven otherwise.
- Avoid large dispatcher refactors while evaluating a hot-path change; refactor after the A/B harness can prove behavior and performance stayed stable.

## Rejected Paths

- Row-wise B traversal during packing has repeatedly reduced pack bandwidth and collapsed one-shot performance.
- Scalar or simpler A-pack fallbacks for small K have been hard regressions.
- Broad low-threshold direct SME routing has not held up. Exact 384 and 768 were accepted only after focused repeat runs.
- One-shot 1024 via public packed-B or direct SME routes has not beaten the current path.
- Increasing expression-level unrolling in the C SME packed-B kernel did not beat the compiler's current schedule.
- m=64 B-reuse changes are shape-sensitive; do not assume a NC/KC knob alone will close the MpGEMM gap.
- SME streaming-B prefetch is route-specific. It helped `m = 64`, large-`K`
  skinny direct widths whose row stride is not a 512-float multiple, plus the
  exact `n = 4096` reuse path, but broad medium, m=96/128, and wide-`N`
  prefetch probes regressed or stayed noisy.
- Keep `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE=32` as the default. Distance
  `64` helps a few high-`K` widths but the boundary is discontinuous; do not
  add exact-width distance gates without counter evidence or a smoother rule.
- For `m = 64`, the SME skinny direct route is useful from `k >= 2048` through
  `n <= 4096`; at `k = 1536`, keep the narrower `n >= 1408` gate because
  lower widths regressed.
- For `m = 64, k = 2048`, `KC=1024` is useful only in the low-width SME skinny
  band currently gated as `n = 1088..1280` plus `n = 1600`. Global
  `COB_SGEMM_SKINNY_SME_KC=384/768/1024` probes regressed important high-`K`
  and `n >= 2048` shapes, so keep cache-blocking changes route-specific.
- Hand-written K=2/K=4 unrolls in the prefetched m64 SME streaming-B C
  intrinsic kernel did not improve the remaining large-`K` gaps. Do not revisit
  that schedule without counter evidence or a dedicated assembly kernel.
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
- For AMX packed full-width shapes, including one-shot and public packed-B, use
  the large-block schedule below `n = 1152` only when `n >= 768` and
  `k >= 3072`, plus the high-row `n = 512, m >= 1024, k >= 4096` band and
  exact `768x512x4096`. The lower-`k` `n = 768, k = 2048` rows,
  `n = 512, k = 3072`, `512x512x4096`, and `768x512x8192` large-block guards
  were behavior-identical/noisy or regressions and should not drive the rule.
- In one-shot AMX medium routes, high-`K` strided-B loses to packing B even
  after pack cost. Keep `m >= 512, k >= 4096` on the packed path, plus
  `n = 1152, k >= 3072`, and exact `n = 1152, k = 2048` from `m >= 512`.
- The same packed-path rule has a lower-height exception at `m = 384`: use the
  packed path for `n >= 1152, k >= 4096` and for exact `n = 1152, k >= 3072`.
  Guard `m = 384` widths below `1152` were neutral/noisy, so do not broaden it.
- For one-shot `512x1280x1536`, use the SME direct-B route. It avoids B-pack
  setup and beat the packed AMX path, while nearby `k = 1024/2048`, `m = 768`,
  `n = 1216`, and `n = 1344` guards stayed neutral/noisy.
- For one-shot `n = 1216`, use the packed path from `k >= 3072`, and also at
  exact `k = 2048` when `m >= 768`. The `m = 512, k = 2048` neighbor and
  `k = 1536` guards are neutral/noisy; `k >= 4096` is covered by the high-`K`
  rule.
- For the one-shot `n = 512` conflict path, keep SME packed-B at `k <= 1024`,
  but use packed AMX at `k >= 2048`. The `k = 1024` AMX fallback probe was a
  regression/noise, while `k = 2048/3072` was consistently positive.
- Do not port cache-blocking constants from other Apple Silicon generations. On this M5 Max, per-P-cluster L2 is 8 MB, page size is 16 KB, and route-specific cache-fit probes still need paired A/B proof.
- Do not prioritize fused inline B-packing rewrites unless a profiler shows packing is the bottleneck on an in-scope licensed-baseline gap.

## External Baselines

- Treat MpGEMM as a calibration target, not a source to copy from. Its checkout has no license file in the local scan.
- Treat KleidiAI FP32 SME2 compute-only results as an upper-bound/reference. Its direct one-shot path includes both LHS and RHS packing and was not faster on the tested public one-shot shapes.
- Treat packages that wrap Accelerate or ship no inspectable source separately from licensed open-source GEMM implementations.
- If a source-available project has unclear licensing, document its numbers separately from the "open-source" claim.
- State the speed claim as "fastest among tested licensed/open-source baselines on the routed shape ranges" unless a fresh source scan and benchmark run cover a broader claim.
