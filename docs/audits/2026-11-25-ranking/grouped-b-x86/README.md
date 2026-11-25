# Grouped B-panel packing on native x86

The grouped-B candidate cleared the registered improvement threshold on 0 of 2 hosted jobs. Each job compared completed public FP32 multiplications on eight previously unmeasured wide rectangles. The remaining four layouts cover two historical anchors and two fallback controls.
Across the registered field, BLASFEO failed qualification on 3 distinct layouts. The results below compare every qualified implementation. The raw tables preserve all 6 failed job/layout/method combinations.

| Native CPU and job | Grouped B / incumbent | Fresh medians won | Pairs won | Grouped B / fastest one-shot | Grouped B / fastest including retained plans |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Intel(R) Xeon(R) 6975P-C, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45701814927/job/139513654760) | 0.971× | 0/8 | 3/48 | 0.858× | 0.858× |
| [Intel(R) Xeon(R) 6975P-C, replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45701814927/job/139513654784) | 0.968× | 1/8 | 4/48 | 0.855× | 0.855× |

The standalone candidate packs up to four full-depth B panels together and reuses its existing register tile. The installed baseline packs one panel at a time. Both ordinary executables call COB’s public overwrite operation and use the same native compiler options, with the grouped-B option enabled in the candidate. Its temporary B buffer is four times as large on the active path.

Both native jobs used Intel(R) Xeon(R) 6975P-C. The experiment supplies repeat measurements on one CPU model.

An entry above 1.000× favors grouped B. For each geometry we divided measured peer time by grouped-B time within each of six seed/round pairs and took the median. Each aggregate is the geometric mean of the eight layout medians. Each independent leader is a real measured implementation with the lowest warm median on that host and layout. A stable lead requires a ratio greater than 1.02× and five of six paired wins on all eight previously unmeasured layouts, with the entire independent tier qualified.

## Intel(R) Xeon(R) 6975P-C, replica 1

Grouped B cleared 0 of eight incumbent medians, with a worst layout of 0.929×. Its separate seed-block ratios were 0.966× and 0.974×. Leaving out each layout in turn gave 0.967–0.978×. The improvement threshold was 1.03× overall, six case wins, both seeds above one and every case at or above 0.97×.

The candidate beat the fastest one-shot competitor in 0/48 matched observations and reached the stable-lead criterion on 0/8 individual layouts. With retained plans eligible, those counts were 0/48 and 0/8.

| M × N × K | Physical layout | Grouped B, warm µs | COB incumbent, warm µs | vs incumbent, wins | Fastest independent one-shot, warm µs | Fastest independent incl. retained, warm µs |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 137 × 2179 × 347 (repeat) | origin | 1141.75 | 1112.49 | 0.975×, 0/6 | OpenBLAS, 1163.99 | OpenBLAS, 1163.99 |
| 79 × 1793 × 223 (repeat) | compact | 375.18 | 358.80 | 0.957×, 0/6 | oneMKL direct-call, 302.47 | oneMKL direct-call, 302.47 |
| 28 × 1153 × 107 (fresh) | cropped | 44.32 | 41.49 | 0.929×, 0/6 | oneMKL, 36.64 | oneMKL, 36.64 |
| 47 × 2051 × 153 (fresh) | cropped | 175.31 | 168.73 | 0.959×, 0/6 | oneMKL, 150.34 | oneMKL, 150.34 |
| 64 × 2304 × 224 (fresh) | compact | 392.84 | 381.86 | 0.972×, 0/6 | oneMKL, 315.14 | oneMKL, 315.14 |
| 75 × 2815 × 263 (fresh) | b-shift | 663.41 | 663.46 | 1.000×, 3/6 | oneMKL direct-call, 548.26 | oneMKL direct-call, 548.26 |
| 99 × 1537 × 311 (fresh) | c-shift | 571.11 | 558.74 | 0.974×, 0/6 | AMD AOCL, 506.08 | AMD AOCL, 506.08 |
| 118 × 3329 × 359 (fresh) | a-shift | 1570.88 | 1526.97 | 0.974×, 0/6 | oneMKL direct-call, 1378.23 | oneMKL direct-call, 1378.23 |
| 143 × 2473 × 229 (fresh) | cropped | 873.15 | 858.33 | 0.983×, 0/6 | oneMKL direct-call, 806.74 | oneMKL direct-call, 806.74 |
| 157 × 3889 × 307 (fresh) | cropped | 2023.97 | 1989.88 | 0.982×, 0/6 | oneMKL direct-call, 1777.68 | oneMKL direct-call, 1777.68 |
| 61 × 983 × 177 (control) | cropped | 141.56 | 141.23 | 0.997×, 2/6 | oneMKL, 102.28 | oneMKL, 102.28 |
| 161 × 2631 × 283 (control) | cropped | 1391.20 | 1389.90 | 0.999×, 2/6 | oneMKL, 1227.34 | oneMKL, 1227.34 |

The lowest full-field warm median belonged to grouped B on 0/8 fresh layouts. All tables use FP32 inputs and outputs. The independent FP64 check examined 307,893,260 logical output cells across four dedicated fixtures and six separate post-batch checks per qualified method/layout. The four fixtures account for 123,157,304 of those checks. Maximum absolute error among observed rows was 3.00649391e-05. The checker examined 18,220,020 locations outside the logical output, with read-only input parents and inaccessible end pages. It timed 24,768 complete calls across 774 batches on logical CPU 0.

Both installed executables passed twenty-four ordinary/trace route checks in total, two allocation-failure checks and eight ordinary-output bit comparisons across the two geometric bypasses. 129/132 possible method and layout combinations qualified.

Excluded method and layout pairs follow. A worker exit of −11 is SIGSEGV on Linux. Its cause has not been determined from this study.

- 137 × 2179 × 347 (repeat-win-ragged), BLASFEO. worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0.
- 157 × 3889 × 307 (fresh-near-row-limit), BLASFEO. worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0.
- 161 × 2631 × 283 (control-above-rows), BLASFEO. worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0.

The first-call observations for the two standalone methods span 49.7–2326.0 µs across all listed layouts. The CSV records each first invocation in a fresh worker after library initialization and before its 32-call batch. Warm data reuse inputs and output storage within the batch. oneDNN one-shot creates a plan per call, whereas its retained variant reuses the plan after creation on that worker’s first call. LIBXSMM dispatch looks up a generated kernel on every call, and its retained variant reuses a kernel pointer. These first-call observations include runtime and page warm-up effects.

## Intel(R) Xeon(R) 6975P-C, replica 2

Grouped B cleared 1 of eight incumbent medians, with a worst layout of 0.902×. Its separate seed-block ratios were 0.964× and 0.973×. Leaving out each layout in turn gave 0.964–0.978×. The improvement threshold was 1.03× overall, six case wins, both seeds above one and every case at or above 0.97×.

The candidate beat the fastest one-shot competitor in 0/48 matched observations and reached the stable-lead criterion on 0/8 individual layouts. With retained plans eligible, those counts were 0/48 and 0/8.

| M × N × K | Physical layout | Grouped B, warm µs | COB incumbent, warm µs | vs incumbent, wins | Fastest independent one-shot, warm µs | Fastest independent incl. retained, warm µs |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 137 × 2179 × 347 (repeat) | origin | 1131.81 | 1109.85 | 0.981×, 0/6 | OpenBLAS, 1161.10 | OpenBLAS, 1161.10 |
| 79 × 1793 × 223 (repeat) | compact | 371.34 | 357.14 | 0.959×, 0/6 | oneMKL direct-call, 304.10 | oneMKL direct-call, 304.10 |
| 28 × 1153 × 107 (fresh) | cropped | 47.27 | 41.73 | 0.902×, 0/6 | oneMKL direct-call, 36.66 | oneMKL direct-call, 36.66 |
| 47 × 2051 × 153 (fresh) | cropped | 177.63 | 167.59 | 0.950×, 0/6 | oneMKL direct-call, 150.41 | oneMKL direct-call, 150.41 |
| 64 × 2304 × 224 (fresh) | compact | 389.39 | 377.36 | 0.966×, 0/6 | oneMKL direct-call, 312.70 | oneMKL direct-call, 312.70 |
| 75 × 2815 × 263 (fresh) | b-shift | 656.20 | 657.26 | 1.001×, 4/6 | oneMKL direct-call, 547.40 | oneMKL direct-call, 547.40 |
| 99 × 1537 × 311 (fresh) | c-shift | 560.44 | 550.59 | 0.980×, 0/6 | AMD AOCL, 495.73 | AMD AOCL, 495.73 |
| 118 × 3329 × 359 (fresh) | a-shift | 1547.46 | 1514.74 | 0.978×, 0/6 | oneMKL direct-call, 1370.05 | oneMKL direct-call, 1370.05 |
| 143 × 2473 × 229 (fresh) | cropped | 861.86 | 848.27 | 0.985×, 0/6 | oneMKL direct-call, 797.50 | oneMKL direct-call, 797.50 |
| 157 × 3889 × 307 (fresh) | cropped | 2000.14 | 1975.95 | 0.988×, 0/6 | oneMKL direct-call, 1767.45 | oneMKL direct-call, 1767.45 |
| 61 × 983 × 177 (control) | cropped | 140.57 | 141.51 | 1.006×, 3/6 | oneMKL direct-call, 101.74 | oneMKL direct-call, 101.74 |
| 161 × 2631 × 283 (control) | cropped | 1377.39 | 1376.46 | 0.999×, 2/6 | oneMKL direct-call, 1219.99 | oneMKL direct-call, 1219.99 |

The lowest full-field warm median belonged to grouped B on 0/8 fresh layouts. All tables use FP32 inputs and outputs. The independent FP64 check examined 307,893,260 logical output cells across four dedicated fixtures and six separate post-batch checks per qualified method/layout. The four fixtures account for 123,157,304 of those checks. Maximum absolute error among observed rows was 3.00649391e-05. The checker examined 18,220,020 locations outside the logical output, with read-only input parents and inaccessible end pages. It timed 24,768 complete calls across 774 batches on logical CPU 0.

Both installed executables passed twenty-four ordinary/trace route checks in total, two allocation-failure checks and eight ordinary-output bit comparisons across the two geometric bypasses. 129/132 possible method and layout combinations qualified.

Excluded method and layout pairs follow. A worker exit of −11 is SIGSEGV on Linux. Its cause has not been determined from this study.

- 137 × 2179 × 347 (repeat-win-ragged), BLASFEO. worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0.
- 157 × 3889 × 307 (fresh-near-row-limit), BLASFEO. worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0.
- 161 × 2631 × 283 (control-above-rows), BLASFEO. worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0.

The first-call observations for the two standalone methods span 48.9–2219.0 µs across all listed layouts. The CSV records each first invocation in a fresh worker after library initialization and before its 32-call batch. Warm data reuse inputs and output storage within the batch. oneDNN one-shot creates a plan per call, whereas its retained variant reuses the plan after creation on that worker’s first call. LIBXSMM dispatch looks up a generated kernel on every call, and its retained variant reuses a kernel pointer. These first-call observations include runtime and page warm-up effects.

## Reproduction

Run `python3 -B packet.py` in the extracted evidence directory to validate the package, compare selected raw members against the original authenticated GitHub ZIPs when included, and recompute every layout, peer, route and per-host decision. The per-host tables retain raw timings, all first-call observations, full physical strides, pointer residues, methods, qualifications and exclusions. [Frozen benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/31a8fd88e74182d984386596585f2f1002d05ce4/tools/world-rank/grouped) · [Installed source release](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6)
