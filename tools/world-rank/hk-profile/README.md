# HK15 parent and child phase timing

Run this diagnostic after the ranking jobs have completed on the same machine and CPU.

```sh
bash tools/world-rank/hk-profile/run.sh REPO NEW_OUTPUT_DIRECTORY CPU
```

The runner archives source revision `0cff6ec089a43e65eeb3bfefd12a1fc68f205be4` and builds two implementations from it with the Linux study flags. The 1024 control uses standard HK15 dispatch. The 960 parent uses its frozen opt-in flag. Each has an ordinary build and a separately instrumented build. The diagnostic covers 1920 × 2880 × 2880 compact and 2048 × 2880 × 3264 with padding three. It warms both implementations, alternates first place across four pairs, and checks one CPU, one thread, every logical output against an exact FP64 pattern, input images, and padding. Source instrumentation is applied in disposable snapshots. The results use one fixed input with randomized, bounded dyadic dot products that repeat across output rows and columns. Results from `smoke` mode cover small rescaled fixtures only.

`raw.csv` contains inclusive times and calls. `summary.json` reports medians of per-round exclusive shares. The outer frame splits into operand formation, output combinations, child calls and residual allocation/dispatch. Child-time residual subtracts the fused and classical bodies. Each body is instrumented after its panel workspace allocation, includes its free, and is partitioned into A packing, B packing, packed kernel application and a residual. The fused body also times its four output merges and odd edges separately. Short apply means a K panel of at most 64 FP32 elements. A packed apply includes its microkernel, SIMD tails and output writes. For each observation, the reported exclusive times sum to the measured whole call. Median shares across observations need not sum to 100%. Raw hardware cache events and DRAM bandwidth are outside this diagnostic.

The paired whole-call ratio quantifies instrumentation impact. The summary flags a median ratio outside 0.95–1.05. Treat a profile outside that range as approximate. Direct rankings and speedups should continue to use the uninstrumented study. Child body counts must be exactly 15 per whole call, all classical at cutoff 1024 and all fused for the 960 parent on these two shapes.
