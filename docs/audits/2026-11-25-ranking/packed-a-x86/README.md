# Standalone packed-A SGEMM on x86-64

Packing A once per call delivered 0.896× on Xeon 6975P-C against the installed COB kernel on eight previously unseen wide FP32 layouts.
All 2 independently scheduled replicas ran on that same CPU model. The per-replica ratios were 0.896×, 0.893×. Packed A won 0/96 matched batches across their separate runs. Cross-vendor transfer requires measurements on another CPU vendor.
Against the fastest qualified independent library on each layout, including retained plans, packed A had the lowest warm median on 0/8 on Xeon 6975P-C. A claim of superiority across the registered panel failed.

The protocol fixed all twelve layouts before the run. Eight fresh cases determine the score. Two repeat earlier losses, one repeats an earlier Intel win at 137 rows, and one checks the lower-width boundary. The latter two use the fallback paths. [Registered panel, source and decision rule](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/668d794ea6485bb77c4178e8bffc848b7e187a19/tools/world-rank/frontier)

The standalone candidate packs A for reuse across output columns. LIBXSMM dispatch requests a kernel in every call and may find it in its internal registry. Both standalone builds use COB’s public overwrite operation on untransposed row-major arrays and share the same installed base release. The host-selected wrappers call a sequential vendor library. The padding variant can allocate, round the row count up to a multiple of eight, copy the inputs needed by the provider, and copy back the logical output during the measured call.

A row’s fastest method has the lowest median time among all qualified methods. A paired speedup is peer time divided by candidate time within the same seed and round. Values above one favor the candidate. oneDNN retained can reuse a matmul plan and LIBXSMM retained can reuse a generated kernel pointer. Both policies appear in the retained-plan field.

| CPU and run | Packed A / release | Fresh medians won | Fastest external one-shot beaten | Fastest external incl. retained beaten |
| --- | ---: | ---: | ---: | ---: |
| [Xeon 6975P-C, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45697897911/job/139500093897) | 0.896× | 0/8 | 0/8 | 0/8 |
| [Xeon 6975P-C, replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45697897911/job/139500093862) | 0.893× | 0/8 | 0/8 | 0/8 |

Repeated jobs on the same model appear in registered replica order. The first complete replica supplies the model’s primary result.

The installed COB release took the lowest full-field median on a fresh layout in one replica. That win did not recur in every replica on the same CPU model.

## Xeon 6975P-C, replica 1

The incumbent criterion failed on this CPU. Across the eight fresh layouts, packed A ran 0.896× as fast as the installed release, with 0/8 higher paired medians and 0/48 individual wins. Its two seed-block speedups were 0.897× and 0.893×. The slowest layout reached 0.850×. Omitting one layout at a time gave 0.890–0.903×.

Packed A had the lowest time against the one-shot independent field on 0/8 fresh layouts and against the independent field including retained plans on 0/8. Its geometric-mean speedups against the respective per-layout leaders were 0.810× and 0.810×. The registered all-layout test required a paired median above 1.02× and at least five of six wins on each fresh layout. It failed against one-shot methods and failed against the field including retained plans.

The installed release beat all independent methods on 0/8 fresh layouts.

Times are warm per-call medians in microseconds. The speedup columns use six matched ratios. Parentheses give the number of individual wins.

| M × N × K, stride padding | Packed A, µs | vs release | Best independent one-shot | Best independent incl. retained | Fastest overall, µs |
| --- | ---: | ---: | --- | --- | --- |
| 37 × 2387 × 119, +5 (repeat) | 127.1 | 0.919× (0/6) | oneMKL direct-call, 106.4 · 0.842× (0/6) | oneMKL direct-call, 106.4 · 0.842× (0/6) | oneMKL direct-call, 106.4 |
| 61 × 3451 × 191, +13 (repeat) | 523.2 | 0.918× (0/6) | oneMKL direct-call, 398.8 · 0.762× (0/6) | oneMKL direct-call, 398.8 · 0.762× (0/6) | oneMKL direct-call, 398.8 |
| 137 × 2179 × 347, +7 (repeat) | 1116.3 | 1.005× (5/6) | OpenBLAS, 1158.3 · 1.038× (6/6) | OpenBLAS, 1158.3 · 1.038× (6/6) | COB packed A, 1116.3 |
| 25 × 1087 × 97, +3 (fresh) | 36.0 | 0.850× (0/6) | oneMKL direct-call, 27.9 · 0.779× (0/6) | oneMKL direct-call, 27.9 · 0.779× (0/6) | oneMKL direct-call, 27.9 |
| 41 × 2377 × 139, +11 (fresh) | 170.1 | 0.908× (0/6) | oneMKL direct-call, 152.9 · 0.898× (0/6) | oneMKL direct-call, 152.9 · 0.898× (0/6) | oneMKL direct-call, 152.9 |
| 55 × 3072 × 184, +1 (fresh) | 420.5 | 0.888× (0/6) | oneMKL direct-call, 313.3 · 0.742× (0/6) | oneMKL direct-call, 313.3 · 0.742× (0/6) | oneMKL direct-call, 313.3 |
| 79 × 1793 × 223, +0 (fresh) | 381.8 | 0.944× (0/6) | oneMKL direct-call, 303.3 · 0.794× (0/6) | oneMKL direct-call, 303.3 · 0.794× (0/6) | Host direct (forwarded to oneMKL), 302.6 |
| 96 × 2304 × 288, +0 (fresh) | 822.9 | 0.868× (0/6) | oneMKL direct-call, 605.7 · 0.735× (0/6) | oneMKL direct-call, 605.7 · 0.735× (0/6) | oneMKL direct-call, 605.7 |
| 73 × 4079 × 337, +9 (fresh) | 1293.3 | 0.916× (0/6) | oneMKL direct-call, 1028.0 · 0.796× (0/6) | oneMKL direct-call, 1028.0 · 0.796× (0/6) | oneMKL direct-call, 1028.0 |
| 91 × 1217 × 377, +5 (fresh) | 535.6 | 0.905× (0/6) | oneMKL direct-call, 468.1 · 0.874× (0/6) | oneMKL direct-call, 468.1 · 0.874× (0/6) | oneMKL direct-call, 468.1 |
| 83 × 2459 × 307, +7 (fresh) | 810.5 | 0.896× (0/6) | oneMKL direct-call, 712.5 · 0.883× (0/6) | oneMKL direct-call, 712.5 · 0.883× (0/6) | oneMKL direct-call, 712.5 |
| 45 × 997 × 173, +9 (control) | 102.2 | 0.989× (3/6) | oneMKL direct-call, 78.8 · 0.772× (0/6) | oneMKL direct-call, 78.8 · 0.772× (0/6) | Host pad 8 (forwarded to oneMKL), 78.1 |

The wrapper selected oneMKL. The padded wrapper’s geometric-mean speedup over calling that library directly was 0.929× on the 8 qualified fresh layouts, with 3/8 higher paired medians. A vendor-assisted method had the lowest time in the full field on 1/8. The unpadded wrapper reached 1.000× against the same library. The raw-median winner Host direct on 79 × 1793 × 223 forwarded the original matrices to oneMKL. Its timing advantage belongs to a repeated call to that same provider.

The padding rule added rows on 5/8 fresh layouts and forwarded 3/8 unchanged. On the padded subset it reached 0.886× against the provider and 0.887× against the forwarding wrapper. The respective minimum paired medians were 0.869× and 0.877×. 0 padded layouts cleared both 1.02× and five-of-six win requirements. The separate padding criterion failed.

On the first call in a fresh worker, packed A had the lowest two-seed median among eligible methods on 2/8 fresh layouts. Its first-call time was 1.15–1.58× its warm time across the qualified fresh layouts. First-call comparisons use two observations per layout.

Qualification covered 156 admitted method/layout combinations, 624 FP64 fixtures, 99,385,104 output values and 936 timed batches. Packed A’s largest observed absolute error was 2.66617e-05.

## Xeon 6975P-C, replica 2

The incumbent criterion failed on this CPU. Across the eight fresh layouts, packed A ran 0.893× as fast as the installed release, with 0/8 higher paired medians and 0/48 individual wins. Its two seed-block speedups were 0.892× and 0.894×. The slowest layout reached 0.850×. Omitting one layout at a time gave 0.887–0.900×.

Packed A had the lowest time against the one-shot independent field on 0/8 fresh layouts and against the independent field including retained plans on 0/8. Its geometric-mean speedups against the respective per-layout leaders were 0.810× and 0.810×. The registered all-layout test required a paired median above 1.02× and at least five of six wins on each fresh layout. It failed against one-shot methods and failed against the field including retained plans.

The installed release beat all independent methods on 1/8 fresh layouts. It also led the full field on 41 × 2377 × 139.

Times are warm per-call medians in microseconds. The speedup columns use six matched ratios. Parentheses give the number of individual wins.

| M × N × K, stride padding | Packed A, µs | vs release | Best independent one-shot | Best independent incl. retained | Fastest overall, µs |
| --- | ---: | ---: | --- | --- | --- |
| 37 × 2387 × 119, +5 (repeat) | 128.0 | 0.917× (0/6) | oneMKL, 107.4 · 0.841× (0/6) | oneMKL, 107.4 · 0.841× (0/6) | oneMKL, 107.4 |
| 61 × 3451 × 191, +13 (repeat) | 525.8 | 0.920× (0/6) | oneMKL direct-call, 397.2 · 0.762× (0/6) | oneMKL direct-call, 397.2 · 0.762× (0/6) | oneMKL direct-call, 397.2 |
| 137 × 2179 × 347, +7 (repeat) | 1119.0 | 1.000× (4/6) | OpenBLAS, 1162.4 · 1.039× (6/6) | OpenBLAS, 1162.4 · 1.039× (6/6) | COB release, 1118.7 |
| 25 × 1087 × 97, +3 (fresh) | 35.6 | 0.850× (0/6) | oneMKL direct-call, 28.0 · 0.782× (0/6) | oneMKL direct-call, 28.0 · 0.782× (0/6) | Host pad 8 (forwarded to oneMKL), 27.8 |
| 41 × 2377 × 139, +11 (fresh) | 171.1 | 0.895× (0/6) | oneMKL direct-call, 154.1 · 0.898× (0/6) | oneMKL direct-call, 154.1 · 0.898× (0/6) | COB release, 153.2 |
| 55 × 3072 × 184, +1 (fresh) | 413.4 | 0.887× (0/6) | oneMKL direct-call, 308.0 · 0.741× (0/6) | oneMKL direct-call, 308.0 · 0.741× (0/6) | oneMKL direct-call, 308.0 |
| 79 × 1793 × 223, +0 (fresh) | 384.3 | 0.938× (0/6) | oneMKL direct-call, 302.9 · 0.786× (0/6) | oneMKL direct-call, 302.9 · 0.786× (0/6) | oneMKL direct-call, 302.9 |
| 96 × 2304 × 288, +0 (fresh) | 822.2 | 0.867× (0/6) | oneMKL direct-call, 609.3 · 0.742× (0/6) | oneMKL direct-call, 609.3 · 0.742× (0/6) | oneMKL direct-call, 609.3 |
| 73 × 4079 × 337, +9 (fresh) | 1294.6 | 0.914× (0/6) | oneMKL direct-call, 1031.8 · 0.798× (0/6) | oneMKL direct-call, 1031.8 · 0.798× (0/6) | oneMKL direct-call, 1031.8 |
| 91 × 1217 × 377, +5 (fresh) | 538.5 | 0.900× (0/6) | oneMKL direct-call, 469.7 · 0.872× (0/6) | oneMKL direct-call, 469.7 · 0.872× (0/6) | oneMKL direct-call, 469.7 |
| 83 × 2459 × 307, +7 (fresh) | 810.2 | 0.899× (0/6) | oneMKL direct-call, 712.9 · 0.879× (0/6) | oneMKL direct-call, 712.9 · 0.879× (0/6) | oneMKL direct-call, 712.9 |
| 45 × 997 × 173, +9 (control) | 101.6 | 1.005× (3/6) | oneMKL direct-call, 78.7 · 0.768× (0/6) | oneMKL direct-call, 78.7 · 0.768× (0/6) | oneMKL direct-call, 78.7 |

The wrapper selected oneMKL. The padded wrapper’s geometric-mean speedup over calling that library directly was 0.928× on the 8 qualified fresh layouts, with 2/8 higher paired medians. A vendor-assisted method had the lowest time in the full field on 1/8. The unpadded wrapper reached 1.001× against the same library. The raw-median winner Host pad 8 on 25 × 1087 × 97 forwarded the original matrices to oneMKL. Its timing advantage belongs to a repeated call to that same provider.

The padding rule added rows on 5/8 fresh layouts and forwarded 3/8 unchanged. On the padded subset it reached 0.887× against the provider and 0.889× against the forwarding wrapper. The respective minimum paired medians were 0.867× and 0.870×. 0 padded layouts cleared both 1.02× and five-of-six win requirements. The separate padding criterion failed.

On the first call in a fresh worker, packed A had the lowest two-seed median among eligible methods on 2/8 fresh layouts. Its first-call time was 1.08–1.42× its warm time across the qualified fresh layouts. First-call comparisons use two observations per layout.

Qualification covered 156 admitted method/layout combinations, 624 FP64 fixtures, 99,385,104 output values and 936 timed batches. Packed A’s largest observed absolute error was 2.66617e-05.

## Reproduce the comparisons

Extract the evidence bundle and run `python3 -B verify_bundle.py` from its root. The verifier rechecks raw hashes, per-method numerical admission, actual kernel routing, and the paired comparisons for each job. The original GitHub job and archive digest are in each host’s provenance.json. With that original ZIP, `python3 members.py HOST_DIRECTORY --archive ORIGINAL_ZIP` also checks the published projection against the downloaded bytes. The per-host records include four FP64 fixtures and first calls, the individual warm batches, signed errors, eligibility, and build and route receipts. `layout-comparisons.csv` includes per-layout whole-call time and all six paired ratios for both the standalone and vendor-assisted entries. `registered/` contains the exact candidate patch and the pre-timing protocols.

To rerun the native experiment, use the pinned benchmark checkout and its `native-gemm.yml` x86 job. The job restores the registered COB release, builds the single-threaded libraries, runs the installed public entry and records full numerical and route witnesses. [Workflow at the measured revision](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/668d794ea6485bb77c4178e8bffc848b7e187a19/.github/workflows/native-gemm.yml)

Each warm observation comprises 32 complete calls on a single pinned Linux process thread. There are three observations for each of two seeds. First calls are recorded in fresh worker processes after initialization. The full-output maximum absolute error threshold was 0.002, with finite output, unchanged inputs and intact padding required. These results cover row-major FP32 SGEMM on the stated shapes and CPUs. [Installed release](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6) · [Benchmark source and candidate patch](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/668d794ea6485bb77c4178e8bffc848b7e187a19/tools/world-rank/frontier)
