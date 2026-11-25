# Public M5 compact-route correctness

On Apple M5 Max, the candidate, the incumbent with its medium-wide option enabled at commit `602a9dddce7e940a282b0716a7b2ca40bd2ad35e`, and a separately instrumented candidate produced identical bits across 7,183,072 outputs in 94 protected-memory test cases. The trace confirmed 74 selected public calls and 20 bypasses. All 15 additional allocation-failure, retry, invalid-argument, and overlapping-output checks passed.

The largest normalized absolute error was 0.0000295414. In the correlated fixtures scaled by 16 and 32, both implementations exceeded absolute error 0.002 on 16,361 and 177,437 outputs respectively. At scale 32, 4,641 outputs exceeded 0.002 even with correctly rounded FP32. The largest observed absolute error at that scale was 0.102181. Exact dyadics produced the exact reference, and the reciprocal-scale fixture using exponents from −80 to +80 reached 0.0000278597 maximum absolute error.

The matrices touch inaccessible pages alternately on their leading and trailing boundaries. Input pages are read-only and output padding carries finite canaries. All output cells are checked against an independent FP64 reference. The leading/trailing placements are separate, freshly seeded calls. Each of the nine null, stride and zero-dimension checks verifies the entire output buffer stays untouched. The four overlapping-output checks compare the whole backing buffer to the incumbent. The public API does not specify an in-place matrix product.

To recompute the summary from saved records, run `python3 summarize.py checks.csv fallbacks.csv summary.json`. To rerun the public, traced and fault-injected checks natively, use the following command with the frozen study's built artifacts.

```sh
sh test.sh BASE_ON.dylib direct.dylib direct-trace.dylib direct-trace-leaf.c SHIPPED_DRIVER.o COB_INCLUDE_DIR
```

All native checks ran after the timed study had finished.
