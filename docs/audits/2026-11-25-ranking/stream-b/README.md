# Reproducing the streamed-B result on Intel Xeon and AMD EPYC

The opt-in research kernel beat the previous COB entry in **all 60 matched timings** across five previously unseen wide FP32 layouts and two native x86 CPUs. Its geometric-mean paired speedup was **1.1200× on an Intel Xeon 6975P-C** and **1.1201× on an AMD EPYC 9V45**. Both passed the frozen improvement criteria at native width, at forced AVX2 and at forced AVX-512.

Against the fastest qualified independent one-shot interface, it won **2/5 fresh layouts on Intel** and **0/5 on AMD**. The corresponding geometric means were 0.935× and 0.958×. Including a retained oneDNN plan, the geometric means were 0.932× and 0.949×. Each host used one pinned thread. These measurements cover the registered wide-layout domain, benchmark source and two CPU models.

| Fresh shape | Intel vs COB | AMD vs COB | Intel vs fastest one-shot | AMD vs fastest one-shot |
| --- | ---: | ---: | --- | --- |
| 37 × 2387 × 119, +5 | 1.338× (6/6) | 1.314× (6/6) | mkl, 0.899× (0/6) | aocl, 0.939× (1/6) |
| 61 × 3451 × 191, +13 | 1.165× (6/6) | 1.192× (6/6) | mkl-jit, 0.844× (0/6) | aocl, 0.944× (1/6) |
| 89 × 1933 × 257, +0 | 1.029× (6/6) | 1.039× (6/6) | mkl-jit, 0.882× (0/6) | aocl, 0.980× (1/6) |
| 108 × 1728 × 320, +4 | 1.041× (6/6) | 1.035× (6/6) | mkl-jit, 1.027× (6/6) | aocl, 0.951× (0/6) |
| 137 × 2179 × 347, +7 | 1.055× (6/6) | 1.047× (6/6) | openblas, 1.037× (6/6) | aocl, 0.976× (0/6) |

Times are scored as paired peer-to-candidate ratios. A value above 1 favors the candidate. Each cell shows the median of six paired observations across two random seeds, followed by the number won. The two runs are summarized separately. The AMD host also tested oneMKL on every layout, and AOCL was the fastest one-shot method on all five. On Intel the compact layout was a 1.013× candidate win against the retained oneDNN plan.

| Five-layout gate | Intel | AMD |
| --- | ---: | ---: |
| Native build | 1.1200× | 1.1201× |
| Forced AVX2 build | 1.1310× | 1.0871× |
| Forced AVX-512 build | 1.1152× | 1.1209× |
| Lowest native speedup on a fresh layout | 1.0294× | 1.0350× |

Both fresh-runner executions used the same source, registered panel, two seeds, and timing order. The exact original and candidate binaries were compiled on each machine with `-march=native`. Each run admitted all 104 method/layout combinations, checked 69,104,464 output cells across 416 FP64 fixtures, and recorded 624 warm batches comprising 19,968 whole multiplications. The maximum native-candidate absolute error was 0.0000239592641051 on both hosts. The retained oneDNN engine binary matched across the hosts. The same OpenBLAS source selected different CPU targets.

The control shape was outside the stated width range. Both trace builds verified bypass, and the native candidate-to-incumbent ratio for the original dispatch was 1.005× on Intel and 0.992× on AMD. The two repeated layouts, all eight layouts' relative and absolute timings, every first call, every individual error and route counts are included under each host. The release integrated after the experiment has its own build and follows a separate acceptance process.

## Verify offline

Run `python3 verify_repeat.py intel amd --host-receipt vm-comparison.json` and compare the resulting JSON to repeat-analysis.json. To bind the public evidence back to the two unedited archived ZIPs, replace the receipt option with `--archives ATTEMPT_1.zip ATTEMPT_2.zip`. The verifier checks both original artifact SHA-256 values, source and method coverage, independent host identity, exact numerical qualification and all paired timings. Each attempt's files include source, build, binary and route receipts.

The benchmark SHA was `b2cdc6a2985726cb1658dc7cc730b5a1c4b99680` and its baseline was `b218beff1eaf8fd606d0934405cdc1e28ecdfe15`. [Frozen protocol and candidate](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b2cdc6a2985726cb1658dc7cc730b5a1c4b99680/tools/world-rank/stream) · [Intel attempt](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697/attempts/1) · [AMD attempt](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697/attempts/2)
