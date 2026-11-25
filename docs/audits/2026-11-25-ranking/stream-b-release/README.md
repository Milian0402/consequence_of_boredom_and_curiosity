# Installed x86 SGEMM release on Intel(R) Xeon(R) 6975P-C

The generic CMake library with streamed B panels passed the registered acceptance gate. On five previously tested wide FP32 layouts, its geometric-mean paired speedup over the otherwise identical installed default was **1.328×**. The option won 5/5 layout medians and 30/30 warm pairs. Against the fastest of the independently built direct libraries on each layout, the installed generic library won 2/5 medians (0.934× geometric mean).

The separately built, native-tuned release passed the parity gate against the exact previously measured research binary rebuilt on this host. The geometric-mean research-time/release-time ratio was 1.008× and the minimum was 0.996×. The gate required at least 0.98× overall and 0.95× on each layout. In the matched tuned production opt-in/off comparison the gain was 1.128×, with 5/5 faster medians. The independently rebuilt research candidate achieved 1.118× versus its own frozen incumbent on these same inputs.

The table lists two repeated shapes, the five predeclared evaluation shapes, and the outside-width control. Each ratio is the median of six matching peer-to-candidate time ratios. Values above 1 favor the enabled production binary. Times are warm-call medians.

| M × N × K, padding | Generic off, µs | Generic on, µs | Generic gain (wins) | Native on vs research | Generic on vs fastest direct (wins) |
| --- | ---: | ---: | ---: | ---: | --- |
| 49 × 2861 × 157, +11 | 376.4 | 277.6 | 1.368× (6/6) | 1.019× | mkl-jit, 0.857× (0/6) |
| 112 × 1744 × 357, +0 | 911.9 | 770.3 | 1.176× (6/6) | 1.000× | mkl-jit, 0.879× (0/6) |
| 37 × 2387 × 119, +5 | 180.5 | 119.8 | 1.508× (6/6) | 1.009× | mkl-jit, 0.897× (0/6) |
| 61 × 3451 × 191, +13 | 619.3 | 481.6 | 1.286× (6/6) | 0.996× | mkl-jit, 0.829× (0/6) |
| 89 × 1933 × 257, +0 | 660.6 | 500.3 | 1.320× (6/6) | 1.019× | mkl-jit, 0.881× (0/6) |
| 108 × 1728 × 320, +4 | 823.2 | 646.8 | 1.273× (6/6) | 1.010× | mkl-jit, 1.022× (6/6) |
| 137 × 2179 × 347, +7 | 1426.8 | 1120.5 | 1.269× (6/6) | 1.006× | openblas, 1.063× (6/6) |
| 39 × 959 × 193, +2 | 114.7 | 113.5 | 1.013× (4/6) | 1.014× | mkl-jit, 0.631× (0/6) |

## What the release test established

The binaries used one thread pinned to logical CPU 0 on Intel(R) Xeon(R) 6975P-C. The native route witness selected 64-column B panels on the seven eligible layouts, verified output and input preservation, and bypassed the outside-width control. The ordinary timed executables link the public function from the exact archive under test. The route witnesses wrap individual kernels in separate, untimed executables linked to the same archive.

CMake configured conservative x86-64 and native builds. Its actual installed archive and public header were checked byte-for-byte against the corresponding build outputs and were the only project artifacts used when linking drivers. Make produced its normal objects, which the acceptance job assembled into an audit archive for the same public call. Opt-in CMake and Make passed tests under both assembly dialects. The conservative SGEMM translation unit loads CPU/OS support at runtime, and only the two private kernel objects require AVX2/FMA or AVX-512F. The option defaults to off.

The raw evidence covers 112 method/layout combinations and 448 full-output FP64 fixtures, including cancellation and scaled data. The workers checked 74,420,192 numerical outputs, and the 672 warm batches covered 21,504 complete products. The maximum observed absolute error was 2.39592641051e-05 for the generic enabled library and 2.39592641051e-05 for its default comparison, within the 0.002 limit. The raw files report first calls, individual seeds, dependencies, and exact binary identities.

Direct comparisons cover OpenBLAS, oneMKL, the oneMKL sequential direct-call/JIT-capable interface and AMD AOCL-BLAS. The independent OpenBLAS comes from the installed Ubuntu package. The earlier research experiment also tested oneDNN and BLASFEO across Intel and AMD. The production comparison reused those shapes, so it tests integration fidelity and supplies an additional full-call measurement.

## Reproduce the audit

Run `python3 verify.py raw` for the independent offline verifier. The included original study scorer also recomputes the registered decisions with `python3 score.py raw`. The raw directory retains the source inventory, compile targets, linked objects, admission evidence, per-method routes, all numerical checks and raw timings. The GitHub artifact checksum and public job identity are in provenance.json.

The production commit is `a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6` and the benchmark commit is `bcc56044efeb37f2ab67bd45c6dae7a42623c292`. [Public release source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6) · [Frozen acceptance source and criteria](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/bcc56044efeb37f2ab67bd45c6dae7a42623c292/tools/world-rank/stream-release) · [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45695245069)
