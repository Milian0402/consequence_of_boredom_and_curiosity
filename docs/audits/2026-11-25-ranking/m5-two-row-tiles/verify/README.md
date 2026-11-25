# Compact 24–32-row public route on M5 Max

A total of 98 protected-memory fixtures checked every one of 5,217,900 output cells in the incumbent, narrow candidate and trace candidate. All three produced identical float values byte for byte. The counter recorded 74 qualifying compact calls and 24 bypasses. Both guard placements passed for the compact 33- and 39-row bypasses, each independently padded leading dimension, and all four edges of the N and K bounds.

The maximum normalized error was 0.0000242733, the largest reciprocal-scale error was 0.0000260541, and the dyadic outputs were exact. Both public implementations exceeded 0.002 absolute error on 14,804 correlated outputs at scale 16 and 112,377 at scale 32. A correctly rounded FP32 result would also exceed that tolerance for 4,886 of the scale-32 outputs. The largest error was 0.1021805. No threshold crossings were introduced or removed.

The separate allocation-failure probe verified all 24,600 outputs after one injected failure and again on an ordinary subsequent call. Nine invalid/no-op calls left the full output unchanged. Four overlapping-output controls bypassed the route and matched the incumbent. Overlap checks preserve the behavior of prior code without asserting that the public API defines in-place multiplication.

Run `python3 summarize.py checks.csv fallbacks.csv summary.json` to independently recalculate the summary. The shell script rebuilds the verifier and the fault-injected copy from the six paths listed in its usage. The timed kernel and ordinary installed baseline are unmodified by the checks.
