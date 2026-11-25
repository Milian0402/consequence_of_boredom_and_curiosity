# Standalone COB on two AMD EPYC processors

On six fresh wide SGEMMs, standalone COB had the lowest warm median in the qualified field on five shapes on the EPYC 7763 and zero on the EPYC 9V45. For the first call after initialization, it had the lower raw median on six and five shapes, respectively. Its paired first-call median ratio favored COB on all six shapes on each CPU.

The two jobs measured different processors and selected different COB kernels. These are exploratory results from an existing panel registered for an Intel vendor-assisted experiment. Each AMD assignment provides one job on its CPU model.

## Six fresh shapes

Each aggregate is the geometric mean of six equally weighted layout ratios. The table counts wins using two calculations. “Raw” compares each method’s median time over six observations. “Paired” divides peer time by COB time within the same seed and round, then takes the median. Ratios above one favor COB. The two seed columns repeat the paired calculation using the three rounds in each seed block.

| CPU | Call | Qualified independent field | Paired geomean | Raw case wins | Paired case wins | Paired round wins | Seed 1 | Seed 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7763 | Warm | one-shot | 1.0195× | 5/6 | 5/6 | 27/36 | 1.0191× | 1.0193× |
| 7763 | Warm | retained-inclusive | 1.0173× | 5/6 | 5/6 | 27/36 | 1.0207× | 1.0144× |
| 7763 | First | one-shot | 1.1228× | 6/6 | 6/6 | 34/36 | 1.0914× | 1.1217× |
| 7763 | First | retained-inclusive | 1.1228× | 6/6 | 6/6 | 34/36 | 1.0914× | 1.1217× |
| 9V45 | Warm | one-shot | 0.9375× | 0/6 | 1/6 | 4/36 | 0.9409× | 0.9379× |
| 9V45 | Warm | retained-inclusive | 0.9251× | 0/6 | 0/6 | 0/36 | 0.9281× | 0.9266× |
| 9V45 | First | one-shot | 1.1579× | 5/6 | 6/6 | 31/36 | 1.1601× | 1.1512× |
| 9V45 | First | retained-inclusive | 1.1579× | 5/6 | 6/6 | 31/36 | 1.1601× | 1.1512× |

The one-shot field comprised OpenBLAS, sequential oneMKL, its JIT-capable direct-call adapter, AOCL, BLASFEO, strict FP32 oneDNN one-shot and LIBXSMM per-call dispatch. The retained-inclusive field also admitted cached-plan oneDNN and LIBXSMM with a retained kernel pointer. For each CPU, layout, regime and tier, the peer was the method with the lowest raw median in that regime. First-call comparisons therefore use first-call leaders.

Workers were pinned to one logical CPU. Each warm observation averages 32 completed public multiplications with resident inputs and output. A first observation times the first complete multiplication in a fresh process after library initialization, including any on-demand plan or kernel creation inside that call. Both regimes use the same six random workers, three apiece for seeds 1047891527 and 1936510481.

## AMD EPYC 7763 64-Core Processor · AVX2

Each of the eleven native traces recorded one public call and one AVX2 kernel call. The timed ordinary COB worker had SHA-256 `2e037198b5cdf2d7ba03d40e359e95a0e61e423b408df061752b977eb719450b`. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139521811697)

### Warm calls

Each ratio cell gives raw / paired. The second peer column also admits retained plans.

| Layout | M × N × K | COB µs | Fastest one-shot | Raw / paired | Fastest incl. retained | Raw / paired |
| --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | 1265.16 | BLASFEO 1325.94 µs | 1.0480 / 1.0451 | BLASFEO 1325.94 µs | 1.0480 / 1.0451 |
| repeat-ragged | 137 × 2179 × 347 | 2223.72 | oneMKL 2401.93 µs | 1.0801 / 1.0778 | oneDNN cached 2363.30 µs | 1.0628 / 1.0630 |
| repeat-short | 43 × 2291 × 137 | 310.44 | AOCL 335.95 µs | 1.0822 / 1.0851 | oneDNN cached 330.07 µs | 1.0632 / 1.0632 |
| fresh-thin | 29 × 1229 × 103 | 87.38 | AOCL 82.90 µs | 0.9487 / 0.9475 | AOCL 82.90 µs | 0.9487 / 0.9475 |
| fresh-odd | 44 × 1897 × 151 | 292.85 | AOCL 312.42 µs | 1.0668 / 1.0693 | oneDNN cached 308.74 µs | 1.0543 / 1.0570 |
| fresh-exact | 68 × 2176 × 244 | 793.70 | AOCL 794.70 µs | 1.0013 / 1.0027 | AOCL 794.70 µs | 1.0013 / 1.0027 |
| fresh-long | 83 × 3067 × 271 | 1509.67 | BLASFEO 1597.47 µs | 1.0582 / 1.0586 | oneDNN cached 1592.61 µs | 1.0549 / 1.0569 |
| fresh-high-depth | 112 × 3581 × 337 | 2922.22 | BLASFEO 2991.08 µs | 1.0236 / 1.0225 | BLASFEO 2991.08 µs | 1.0236 / 1.0225 |
| fresh-upper | 151 × 2529 × 311 | 2552.30 | BLASFEO 2606.89 µs | 1.0214 / 1.0214 | BLASFEO 2606.89 µs | 1.0214 / 1.0214 |
| control-cropped-deeper | 108 × 1728 × 320 | 1276.15 | BLASFEO 1324.93 µs | 1.0382 / 1.0379 | BLASFEO 1324.93 µs | 1.0382 / 1.0379 |
| control-cropped-short | 43 × 2291 × 137 | 312.28 | BLASFEO 363.85 µs | 1.1651 / 1.1615 | oneDNN cached 332.53 µs | 1.0648 / 1.0597 |

### First calls

Each ratio cell gives raw / paired. Both independent fields selected the same leader on every layout.

| Layout | M × N × K | COB µs | Fastest independent | Raw / paired |
| --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | 1302.44 | AOCL 1400.02 µs | 1.0749 / 1.0801 |
| repeat-ragged | 137 × 2179 × 347 | 2280.41 | AOCL 2602.80 µs | 1.1414 / 1.1390 |
| repeat-short | 43 × 2291 × 137 | 318.19 | AOCL 380.69 µs | 1.1964 / 1.2029 |
| fresh-thin | 29 × 1229 × 103 | 96.60 | BLASFEO 108.20 µs | 1.1200 / 1.1377 |
| fresh-odd | 44 × 1897 × 151 | 298.15 | BLASFEO 335.55 µs | 1.1254 / 1.1268 |
| fresh-exact | 68 × 2176 × 244 | 813.25 | AOCL 846.92 µs | 1.0414 / 1.0500 |
| fresh-long | 83 × 3067 × 271 | 1542.75 | AOCL 1775.11 µs | 1.1506 / 1.1428 |
| fresh-high-depth | 112 × 3581 × 337 | 2986.60 | OpenBLAS 3271.37 µs | 1.0953 / 1.0929 |
| fresh-upper | 151 × 2529 × 311 | 2607.23 | AOCL 3112.52 µs | 1.1938 / 1.1919 |
| control-cropped-deeper | 108 × 1728 × 320 | 1317.49 | AOCL 1559.99 µs | 1.1841 / 1.1890 |
| control-cropped-short | 43 × 2291 × 137 | 321.45 | BLASFEO 390.69 µs | 1.2154 / 1.2040 |

## AMD EPYC 9V45 96-Core Processor · AVX-512

Each of the eleven native traces recorded one public call and one AVX-512 kernel call. The timed ordinary COB worker had SHA-256 `357eb8f3348982bcc8efde5ba3a0dd9bd7bc3ce8ed77728b9ec45b50b6be1ace`. [Native job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139521811716)

### Warm calls

Each ratio cell gives raw / paired. The second peer column also admits retained plans.

| Layout | M × N × K | COB µs | Fastest one-shot | Raw / paired | Fastest incl. retained | Raw / paired |
| --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | 492.07 | AOCL 462.27 µs | 0.9394 / 0.9443 | AOCL 462.27 µs | 0.9394 / 0.9443 |
| repeat-ragged | 137 × 2179 × 347 | 858.89 | AOCL 839.48 µs | 0.9774 / 0.9773 | oneDNN cached 812.67 µs | 0.9462 / 0.9465 |
| repeat-short | 43 × 2291 × 137 | 124.56 | AOCL 119.83 µs | 0.9621 / 0.9630 | AOCL 119.83 µs | 0.9621 / 0.9630 |
| fresh-thin | 29 × 1229 × 103 | 34.54 | LIBXSMM dispatch 28.40 µs | 0.8222 / 0.8234 | LIBXSMM retained 28.31 µs | 0.8194 / 0.8172 |
| fresh-odd | 44 × 1897 × 151 | 116.50 | AOCL 111.78 µs | 0.9594 / 0.9622 | oneDNN cached 111.63 µs | 0.9582 / 0.9459 |
| fresh-exact | 68 × 2176 × 244 | 316.04 | AOCL 289.94 µs | 0.9174 / 0.9168 | AOCL 289.94 µs | 0.9174 / 0.9168 |
| fresh-long | 83 × 3067 × 271 | 584.42 | AOCL 583.58 µs | 0.9986 / 1.0021 | oneDNN cached 561.39 µs | 0.9606 / 0.9478 |
| fresh-high-depth | 112 × 3581 × 337 | 1111.22 | AOCL 1070.94 µs | 0.9638 / 0.9564 | AOCL 1070.94 µs | 0.9638 / 0.9564 |
| fresh-upper | 151 × 2529 × 311 | 974.71 | AOCL 951.80 µs | 0.9765 / 0.9758 | AOCL 951.80 µs | 0.9765 / 0.9758 |
| control-cropped-deeper | 108 × 1728 × 320 | 488.82 | AOCL 463.45 µs | 0.9481 / 0.9432 | AOCL 463.45 µs | 0.9481 / 0.9432 |
| control-cropped-short | 43 × 2291 × 137 | 125.06 | AOCL 120.96 µs | 0.9672 / 0.9504 | AOCL 120.96 µs | 0.9672 / 0.9504 |

### First calls

Each ratio cell gives raw / paired. Both independent fields selected the same leader on every layout.

| Layout | M × N × K | COB µs | Fastest independent | Raw / paired |
| --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | 521.64 | AOCL 507.84 µs | 0.9736 / 0.9791 |
| repeat-ragged | 137 × 2179 × 347 | 901.95 | AOCL 902.20 µs | 1.0003 / 0.9935 |
| repeat-short | 43 × 2291 × 137 | 137.60 | AOCL 174.32 µs | 1.2669 / 1.2613 |
| fresh-thin | 29 × 1229 × 103 | 45.57 | BLASFEO 80.87 µs | 1.7745 / 1.7498 |
| fresh-odd | 44 × 1897 × 151 | 130.55 | AOCL 164.37 µs | 1.2590 / 1.2589 |
| fresh-exact | 68 × 2176 × 244 | 337.91 | AOCL 346.14 µs | 1.0243 / 1.0236 |
| fresh-long | 83 × 3067 × 271 | 606.85 | AOCL 647.79 µs | 1.0675 / 1.0482 |
| fresh-high-depth | 112 × 3581 × 337 | 1154.57 | AOCL 1148.18 µs | 0.9945 / 1.0031 |
| fresh-upper | 151 × 2529 × 311 | 1013.74 | AOCL 1033.66 µs | 1.0197 / 1.0167 |
| control-cropped-deeper | 108 × 1728 × 320 | 521.05 | AOCL 512.06 µs | 0.9827 / 0.9792 |
| control-cropped-short | 43 × 2291 × 137 | 140.63 | AOCL 174.32 µs | 1.2396 / 1.2302 |

At 112 × 3581 × 337, AOCL’s first-call median of 1148.18 µs was lower than COB’s 1154.57 µs. COB finished first in 4/6 matched rounds. The ratio of raw medians was 0.99446× and the median paired ratio was 1.00306×. This layout accounts for the difference between five fresh first-call wins by raw median and six by paired median.

## Coverage and provenance

Both jobs had all nine independent methods qualified on all six fresh shapes. On the historical 137 × 2179 × 347 layout, BLASFEO exited with signal 11 on both jobs and its time is missing. The other independent comparisons on that layout use eight qualified methods. The “control-cropped” labels identify the original experiment’s two cropped views. COB executed its ordinary standalone path there.

The subject was the installed single-threaded `cob_sgemm_rowmajor` overwrite operation from the same tuned source release on both machines. The original score replays check ordinary binary identity, native dispatch, FP64-reference accuracy, input protection, output padding and guards. The three oneMKL-assisted wrappers ran as forwarding controls on AMD. Their measurements are available in the CSV, outside the independent field. [Standalone source at a30d71f0](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6)

The CSV gives all six round/seed IDs, COB and peer times, ratios, binary identities, qualification status and independent leaders for each registered method, layout and regime. It also preserves the four dedicated first-call fixture observations, which use distinct qualification workers and are excluded from the first-call ranking. [All comparisons](comparisons.csv)
