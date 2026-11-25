# M5 Max wide-rectangle field test

A single-thread FP32 kernel that packs each B panel while computing its first output tile improved on the released COB implementation on **all eight fresh layouts on both seeds**. Its geometric-mean speedup was **2.773×** versus release and **0.978×** versus the fastest external method per layout. It cleared the external 1% win margin on **2 of eight**. The broad 24–160-row treatment failed the four-win acceptance rule and remains an experiment.

The table gives paired median whole-call speedups over six warm rounds on two seeds. A value above one favors the candidate. All ordinary implementations include public entry, workspace, packing and output. Prepared LIBXSMM invokes its verified generated machine-code kernel directly, with JIT setup recorded separately. MaxMulSK and MpGEMM accept compact inputs. The other approaches support the padded cases. The three repeated shapes, below-width control, and Apple-specific controls are separate from the eight fresh cases.

| Layout (M × N × K) | Pad | Cohort | vs release | Leading external | vs leader |
|---|---:|---|---:|---|---:|
| 37 × 2387 × 119 | 5 | repeat-thin-loss | 13.155× | prepared-LIBXSMM | 1.005× |
| 61 × 3451 × 191 | 13 | repeat-wide-loss | 9.188× | Accelerate | 1.081× |
| 137 × 2179 × 347 | 7 | repeat-wide-win | 1.138× | prepared-LIBXSMM | 0.958× |
| 25 × 1087 × 97 | 3 | fresh-lower-fringe | 14.979× | Accelerate | 0.854× |
| 41 × 2377 × 139 | 11 | fresh-irregular | 11.930× | prepared-LIBXSMM | 0.953× |
| 55 × 3072 × 184 | 1 | fresh-exact-width | 8.853× | Accelerate | 1.116× |
| 79 × 1793 × 223 | 0 | fresh-single-column-tail | 1.174× | prepared-LIBXSMM | 0.955× |
| 96 × 2304 × 288 | 0 | fresh-full-tiles | 1.185× | MpGEMM | 0.963× |
| 73 × 4079 × 337 | 9 | fresh-near-max-width | 1.151× | prepared-LIBXSMM | 1.094× |
| 91 × 1217 × 377 | 5 | fresh-high-depth | 1.163× | prepared-LIBXSMM | 0.934× |
| 83 × 2459 × 307 | 7 | fresh-long-strided | 1.188× | prepared-LIBXSMM | 0.981× |
| 45 × 997 × 173 | 9 | control-below-width | 1.005× | Accelerate | 0.085× |
| 23 × 1537 × 157 | 3 | apple-lower-row-control | 0.994× | Accelerate | 0.095× |
| 160 × 4096 × 384 | 0 | apple-upper-corner | 1.056× | MpGEMM | 0.988× |
| 64 × 4097 × 257 | 7 | apple-above-width-control | 1.003× | prepared-LIBXSMM | 0.897× |

Across the eight fresh layouts, the candidate averaged 1.011× explicitly single-thread Apple Accelerate and 1.009× direct prepared LIBXSMM. 576 timed results passed independent FP64 checks, including the first invocation and every timed batch, for 153,771,264 checked cells. Maximum absolute error was 0.00002878 on normalized inputs. Mach reported one thread before and after every timed batch. No core affinity is available in this macOS protocol. Process-first observations in the broad panel may follow another participant on shared hardware.

The public candidate also completed 57 guard-page cases and 8,530,530 output comparisons, with read-only A and B, protected row ends, and input fixtures covering normalized random data, cancellation and mixed scales. A separate trace-build repeat observed the intended branch for selected cases and bypass outside its guard, including overlapping outputs. The archive contains the frozen patch against a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6, cohort, raw observations, scorer and harness. Run `python3 analyze.py` from the evidence directory to regenerate the scores. These M5 results apply to one tested processor and a bounded workload.
