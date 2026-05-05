# Optimization Design Rules

These rules summarize repeated findings from the optimization timeline. They are not permanent laws; revisit them only when the route, hardware, compiler, or benchmark harness changes enough to invalidate the old measurement.

## Measurement

- Prefer paired A/B runs over comparing medians from separate benchmark invocations.
- Treat runs with sample CV above 2% as noisy, even when the apparent median delta is large. Use `COB_AB_MAX_REPEATS` to let the paired harness collect more samples before deciding.
- Recheck any candidate win in a fresh process before committing a performance route.
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

## External Baselines

- Treat MpGEMM as a calibration target, not a source to copy from. Its checkout has no license file in the local scan.
- Treat KleidiAI FP32 SME2 compute-only results as an upper-bound/reference. Its direct one-shot path includes both LHS and RHS packing and was not faster on the tested public one-shot shapes.
- If a source-available project has unclear licensing, document its numbers separately from the "open-source" claim.
