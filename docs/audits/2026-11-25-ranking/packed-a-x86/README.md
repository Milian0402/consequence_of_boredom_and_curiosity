# Supplemental packed-A SGEMM measurement

The single additional job ran on AMD EPYC 9V45 96-Core Processor. Across the same eight fresh-at-registration layouts, packed A ran 0.901× as fast as the installed COB release and won 0/48 matched batches. The fixed incumbent criterion failed.
Packed A had the lowest warm median against the fastest independent library, including retained plans, on 0/8 layouts. Its geometric-mean speedup against those per-layout winners was 0.827×.

The lowest-numbered complete registered replica supplies the model’s primary result, regardless of completion time. The two original registered jobs had both landed on Intel(R) Xeon(R) 6975P-C, with paired speedups of 0.896× and 0.893×. After those outcomes were known, one additional VM was requested to test transfer. The commit, panel, inputs, method cohorts, timing procedure and decision thresholds were held fixed. The supplemental result was retained regardless of the VM’s CPU or measured performance. [Registered follow-up decision](registered/supplemental.md)

The third job supplies a cross-vendor measurement on EPYC 9V45. The original Intel result failed its incumbent criterion, so the candidate did not qualify for a cross-vendor improvement claim.

| Execution | CPU | Packed A / release | Fresh medians won | Fastest one-shot beaten | Fastest incl. retained beaten |
| --- | --- | ---: | ---: | ---: | ---: |
| [Registered replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45697897911/job/139500093897) | Xeon 6975P-C | 0.896× | 0/8 | 0/8 | 0/8 |
| [Registered replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45697897911/job/139500093862) | Xeon 6975P-C | 0.893× | 0/8 | 0/8 | 0/8 |
| [Supplemental job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45697897911/job/139503993618) | EPYC 9V45 | 0.901× | 0/8 | 0/8 | 0/8 |

## EPYC 9V45, replica 1

The incumbent criterion failed on this CPU. Across the eight fresh layouts, packed A ran 0.901× as fast as the installed release, with 0/8 higher paired medians and 0/48 individual wins. Its two seed-block speedups were 0.900× and 0.901×. The slowest layout reached 0.863×. Omitting one layout at a time gave 0.894–0.906×.

Packed A had the lowest time against the one-shot independent field on 0/8 fresh layouts and against the independent field including retained plans on 0/8. Its geometric-mean speedups against the respective per-layout leaders were 0.835× and 0.827×. The registered all-layout test required a paired median above 1.02× and at least five of six wins on each fresh layout. It failed against one-shot methods and failed against the field including retained plans.

The installed release beat all independent methods on 0/8 fresh layouts.

Times are warm per-call medians in microseconds. The speedup columns use six matched ratios. Parentheses give the number of individual wins.

| M × N × K, stride padding | Packed A, µs | vs release | Best independent one-shot | Best independent incl. retained | Fastest overall, µs |
| --- | ---: | ---: | --- | --- | --- |
| 37 × 2387 × 119, +5 (repeat) | 104.9 | 0.926× (0/6) | LIBXSMM dispatch, 88.7 · 0.838× (0/6) | LIBXSMM dispatch, 88.7 · 0.838× (0/6) | LIBXSMM dispatch, 88.7 |
| 61 × 3451 × 191, +13 (repeat) | 371.9 | 0.932× (0/6) | AOCL, 327.5 · 0.875× (0/6) | AOCL, 327.5 · 0.875× (0/6) | AOCL, 327.5 |
| 137 × 2179 × 347, +7 (repeat) | 836.3 | 1.003× (5/6) | AOCL, 806.1 · 0.967× (1/6) | oneDNN retained, 795.5 · 0.959× (1/6) | oneDNN retained, 795.5 |
| 25 × 1087 × 97, +3 (fresh) | 27.7 | 0.897× (0/6) | LIBXSMM dispatch, 20.0 · 0.722× (0/6) | LIBXSMM retained, 19.6 · 0.707× (0/6) | LIBXSMM retained, 19.6 |
| 41 × 2377 × 139, +11 (fresh) | 135.6 | 0.888× (0/6) | LIBXSMM dispatch, 114.5 · 0.842× (0/6) | oneDNN retained, 111.7 · 0.829× (0/6) | oneDNN retained, 111.7 |
| 55 × 3072 × 184, +1 (fresh) | 323.1 | 0.863× (0/6) | AOCL, 278.1 · 0.851× (0/6) | AOCL, 278.1 · 0.851× (0/6) | Host direct (forwarded to AOCL), 276.9 |
| 79 × 1793 × 223, +0 (fresh) | 285.6 | 0.946× (0/6) | AOCL, 256.1 · 0.897× (0/6) | AOCL, 256.1 · 0.897× (0/6) | Host direct (forwarded to AOCL), 254.1 |
| 96 × 2304 × 288, +0 (fresh) | 563.3 | 0.897× (0/6) | AOCL, 469.7 · 0.829× (0/6) | AOCL, 469.7 · 0.829× (0/6) | Host pad 8 (forwarded to AOCL), 463.0 |
| 73 × 4079 × 337, +9 (fresh) | 967.7 | 0.896× (0/6) | AOCL, 802.5 · 0.830× (0/6) | AOCL, 802.5 · 0.830× (0/6) | Host direct (forwarded to AOCL), 795.5 |
| 91 × 1217 × 377, +5 (fresh) | 393.3 | 0.908× (0/6) | AOCL, 325.4 · 0.837× (0/6) | AOCL, 325.4 · 0.837× (0/6) | AOCL, 325.4 |
| 83 × 2459 × 307, +7 (fresh) | 560.6 | 0.914× (0/6) | AOCL, 494.4 · 0.882× (0/6) | oneDNN retained, 476.9 · 0.851× (0/6) | oneDNN retained, 476.9 |
| 45 × 997 × 173, +9 (control) | 79.0 | 1.002× (3/6) | LIBXSMM dispatch, 60.2 · 0.740× (0/6) | LIBXSMM dispatch, 60.2 · 0.740× (0/6) | LIBXSMM dispatch, 60.2 |

The wrapper selected AOCL. The padded wrapper’s geometric-mean speedup over calling that library directly was 0.971× on the 8 qualified fresh layouts, with 2/8 higher paired medians. A vendor-assisted method had the lowest time in the full field on 4/8. The unpadded wrapper reached 0.997× against the same library. The raw-median winner Host direct on 55 × 3072 × 184, Host direct on 79 × 1793 × 223, Host pad 8 on 96 × 2304 × 288, Host direct on 73 × 4079 × 337 forwarded the original matrices to AOCL. Its timing advantage belongs to a repeated call to that same provider.

The padding rule added rows on 5/8 fresh layouts and forwarded 3/8 unchanged. On the padded subset it reached 0.954× against the provider and 0.954× against the forwarding wrapper. The respective minimum paired medians were 0.909× and 0.902×. 0 padded layouts cleared both 1.02× and five-of-six win requirements. The separate padding criterion failed.

On the first call in a fresh worker, packed A had the lowest two-seed median among eligible methods on 0/8 fresh layouts. Its first-call time was 1.12–1.74× its warm time across the qualified fresh layouts. First-call comparisons use two observations per layout.

Qualification covered 156 admitted method/layout combinations, 624 FP64 fixtures, 99,385,104 output values and 936 timed batches. Packed A’s largest observed absolute error was 2.66617e-05.

## Verify the evidence

Extract the bundle and run `python3 -B verify_bundle.py`. Each of the three original jobs has a separate archive receipt, native numerical and route records, and individual warm and cold timings. `layout-comparisons.csv` gives every paired ratio by job, including both assisted wrappers against their selected library. The timing samples are kept separate for each VM.

Each warm batch contains 32 complete FP32 row-major SGEMMs. The six batch observations use two seeds and three rounds per seed. Ratios above one favor the candidate. The incumbent and candidate are standalone, host-direct and host-pad8 forward to the CPUID-selected vendor, and the reported independent field includes one-shot and separately identified retained-plan variants. [Pinned benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/668d794ea6485bb77c4178e8bffc848b7e187a19/tools/world-rank/frontier) · [Installed release](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6)
