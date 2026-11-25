# Standalone COB against explicit oneMKL handles

Two explicit oneMKL API policies extend the independent field. The one-shot variant constructs and destroys a handle for each multiplication. The prepared variant constructs once per worker and reuses its callable. The subject is the installed standalone COB release performing a single-threaded row-major FP32 SGEMM with α = 1 and β = 0. Each host was scored separately.

The overview covers six principal layouts reused from the preceding study. The table compares the original and expanded retained-inclusive fields. The speed ratio is the geometric mean of the six layout-level paired medians. A ratio above one favors COB.

| Native job | Regime | Original 9 | Expanded 11 | COB raw wins after expansion | Leader changes |
| --- | --- | --- | --- | --- | --- |
| [Intel(R) Xeon(R) 6975P-C, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45710128444/job/139542168202) | Warm | 0.8849× | 0.8829× | 0/6 | 1/6 |
| [Intel(R) Xeon(R) 6975P-C, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45710128444/job/139542168202) | First execution | 1.2185× | 1.0160× | 4/6 | 5/6 |
| [Intel(R) Xeon(R) 6975P-C, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45710128444/job/139542168202) | Setup + first | 1.2185× | 1.2185× | 5/6 | 0/6 |
| [Intel(R) Xeon(R) 6975P-C, replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45710128444/job/139542168188) | Warm | 0.8791× | 0.8791× | 0/6 | 0/6 |
| [Intel(R) Xeon(R) 6975P-C, replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45710128444/job/139542168188) | First execution | 1.2251× | 1.0071× | 4/6 | 6/6 |
| [Intel(R) Xeon(R) 6975P-C, replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45710128444/job/139542168188) | Setup + first | 1.2251× | 1.2251× | 5/6 | 0/6 |

All eleven physical layouts, including three historical anchors and two cropped views, preserve their dimensions and strides. The rerun used new input draws from seeds 1587984119 and 384761239. Each qualified method supplied three workers per seed. Warm samples time 32 complete products per worker on resident data. A first execution is the worker’s first complete product after initialization. Only the explicitly prepared method sets up its handle before that timer. “Setup + first” adds its measured construction and pointer-lookup time to the same worker’s first execution. Process startup, library initialization and eventual prepared-handle destruction are outside this interval. All other policies retain their observed first-call times. The six matched rounds repeat two input draws on one VM per job, so small timing differences carry no estimate of variation across machines.

The original seven one-shot methods are OpenBLAS, ordinary sequential oneMKL, its JIT-capable direct-call macro, AOCL, BLASFEO, oneDNN one-shot and LIBXSMM per-call dispatch. The original retained-inclusive field also admits oneDNN cached-plan and retained LIBXSMM. The expanded one-shot field adds the explicit one-shot policy. The expanded retained-inclusive field admits both explicit policies. MKL_JIT_SUCCESS records a generated implementation and MKL_NO_JIT records an admitted callable fallback.

### Original assignments

The original AMD EPYC 7763 and Intel Xeon Platinum 8573C jobs both ended before timing because the adapter build lacked the oneMKL SDK path manifest. Their receipts and native checks are preserved separately. The results below come from the two newly assigned replacement jobs after correction of the manifest export. [AMD job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45708206524/job/139535589638) · [Intel job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45708206524/job/139535589668) · [Original assignment record](initial-attempt/README.md)

## Intel(R) Xeon(R) 6975P-C · replica 1

Raw wins compare COB’s median time with the independent leader’s median. Paired wins use the median of the six within-round leader/COB ratios. The seed columns give the three-round seed results aggregated across the principal layouts. The final column counts layouts whose leader, raw COB win, or paired COB win changed when the two policies entered their respective fields.

| Regime | Independent field | Paired geomean | COB raw wins | COB paired wins | COB round wins | Seed 1 / 2 | Complete fields | Changes (leader / raw / paired) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Warm | Original one-shot (7) | 0.8849× | 0/6 | 0/6 | 0/36 | 0.8786 / 0.8866 | 6/6 | reference |
| Warm | Expanded one-shot (8) | 0.8849× | 0/6 | 0/6 | 0/36 | 0.8786 / 0.8866 | 6/6 | 0 / 0 / 0 of 6 |
| Warm | Original incl. retained (9) | 0.8849× | 0/6 | 0/6 | 0/36 | 0.8786 / 0.8866 | 6/6 | reference |
| Warm | Expanded incl. retained (11) | 0.8829× | 0/6 | 0/6 | 0/36 | 0.8740 / 0.8853 | 6/6 | 1 / 0 / 0 of 6 |
| First execution | Original one-shot (7) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | reference |
| First execution | Expanded one-shot (8) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | 0 / 0 / 0 of 6 |
| First execution | Original incl. retained (9) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | reference |
| First execution | Expanded incl. retained (11) | 1.0160× | 4/6 | 4/6 | 20/36 | 1.0099 / 1.0208 | 6/6 | 5 / 1 / 1 of 6 |
| Setup + first | Original one-shot (7) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | reference |
| Setup + first | Expanded one-shot (8) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | 0 / 0 / 0 of 6 |
| Setup + first | Original incl. retained (9) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | reference |
| Setup + first | Expanded incl. retained (11) | 1.2185× | 5/6 | 5/6 | 33/36 | 1.2217 / 1.2138 | 6/6 | 0 / 0 / 0 of 6 |

### Fastest independent method on each layout

Each entry gives the original leader → the expanded leader, with its median and its paired time ratio against COB. A single entry means the same leader won both fields. Each regime selects its leaders from the measurements in that regime.

#### Warm

| Physical layout · M × N × K | COB µs | One-shot leaders (7 → 8) | Retained-inclusive leaders (9 → 11) |
| --- | --- | --- | --- |
| repeat-deeper · 108 × 1728 × 320 | 669.78 | oneMKL direct-call 697.23 µs · 1.0416× | oneDNN cached 686.16 µs · 1.0211× |
| repeat-ragged · 137 × 2179 × 347 | 1151.95 | OpenBLAS 1199.35 µs · 1.0429× | OpenBLAS 1199.35 µs · 1.0429× |
| repeat-short · 43 × 2291 × 137 | 159.48 | oneMKL direct-call 157.70 µs · 0.9905× | oneMKL direct-call 157.70 µs · 0.9905× → Explicit prepared 156.53 µs · 0.9779× |
| fresh-thin · 29 × 1229 × 103 | 42.91 | LIBXSMM dispatch 36.21 µs · 0.8441× | LIBXSMM dispatch 36.21 µs · 0.8441× → Explicit prepared 35.80 µs · 0.8325× |
| fresh-odd · 44 × 1897 × 151 | 147.40 | oneMKL direct-call 133.68 µs · 0.9165× | oneMKL direct-call 133.68 µs · 0.9165× |
| fresh-exact · 68 × 2176 × 244 | 429.74 | oneMKL direct-call 375.03 µs · 0.8694× | oneMKL direct-call 375.03 µs · 0.8694× |
| fresh-long · 83 × 3067 × 271 | 817.10 | oneMKL 681.40 µs · 0.8486× | oneMKL 681.40 µs · 0.8486× |
| fresh-high-depth · 112 × 3581 × 337 | 1528.32 | oneMKL 1389.31 µs · 0.9129× | oneMKL 1389.31 µs · 0.9129× |
| fresh-upper · 151 × 2529 × 311 | 1319.70 | oneMKL direct-call 1213.47 µs · 0.9215× | oneMKL direct-call 1213.47 µs · 0.9215× |
| control-cropped-deeper · 108 × 1728 × 320 | 668.80 | oneMKL direct-call 624.34 µs · 0.9326× | oneMKL direct-call 624.34 µs · 0.9326× |
| control-cropped-short · 43 × 2291 × 137 | 158.41 | oneMKL direct-call 143.21 µs · 0.8970× | oneMKL direct-call 143.21 µs · 0.8970× |

#### First execution

| Physical layout · M × N × K | COB µs | One-shot leaders (7 → 8) | Retained-inclusive leaders (9 → 11) |
| --- | --- | --- | --- |
| repeat-deeper · 108 × 1728 × 320 | 768.29 | AOCL 752.26 µs · 1.0101× | AOCL 752.26 µs · 1.0101× → Explicit prepared 734.45 µs · 0.9604× |
| repeat-ragged · 137 × 2179 × 347 | 1278.51 | OpenBLAS 1408.81 µs · 1.1069× | OpenBLAS 1408.81 µs · 1.1069× → Explicit prepared 1394.42 µs · 1.0916× |
| repeat-short · 43 × 2291 × 137 | 168.65 | AOCL 274.32 µs · 1.6163× | AOCL 274.32 µs · 1.6163× → Explicit prepared 216.37 µs · 1.2735× |
| fresh-thin · 29 × 1229 × 103 | 53.43 | AOCL 84.57 µs · 1.5571× | AOCL 84.57 µs · 1.5571× → Explicit prepared 54.20 µs · 1.0135× |
| fresh-odd · 44 × 1897 × 151 | 156.18 | AOCL 220.99 µs · 1.3890× | AOCL 220.99 µs · 1.3890× → Explicit prepared 210.88 µs · 1.3338× |
| fresh-exact · 68 × 2176 × 244 | 562.50 | AOCL 540.36 µs · 0.9887× | AOCL 540.36 µs · 0.9887× → Explicit prepared 435.58 µs · 0.7879× |
| fresh-long · 83 × 3067 × 271 | 995.06 | OpenBLAS 1174.67 µs · 1.1681× | OpenBLAS 1174.67 µs · 1.1681× → Explicit prepared 1003.19 µs · 1.0071× |
| fresh-high-depth · 112 × 3581 × 337 | 1617.49 | OpenBLAS 1752.21 µs · 1.0900× | OpenBLAS 1752.21 µs · 1.0900× |
| fresh-upper · 151 × 2529 × 311 | 1418.31 | OpenBLAS 1712.24 µs · 1.2019× | OpenBLAS 1712.24 µs · 1.2019× → Explicit prepared 1359.01 µs · 0.9410× |
| control-cropped-deeper · 108 × 1728 × 320 | 741.47 | AOCL 822.42 µs · 1.1092× | AOCL 822.42 µs · 1.1092× → Explicit prepared 777.47 µs · 1.0473× |
| control-cropped-short · 43 × 2291 × 137 | 169.05 | AOCL 285.17 µs · 1.6866× | AOCL 285.17 µs · 1.6866× → Explicit prepared 217.42 µs · 1.3067× |

#### Setup + first

| Physical layout · M × N × K | COB µs | One-shot leaders (7 → 8) | Retained-inclusive leaders (9 → 11) |
| --- | --- | --- | --- |
| repeat-deeper · 108 × 1728 × 320 | 768.29 | AOCL 752.26 µs · 1.0101× | AOCL 752.26 µs · 1.0101× |
| repeat-ragged · 137 × 2179 × 347 | 1278.51 | OpenBLAS 1408.81 µs · 1.1069× | OpenBLAS 1408.81 µs · 1.1069× |
| repeat-short · 43 × 2291 × 137 | 168.65 | AOCL 274.32 µs · 1.6163× | AOCL 274.32 µs · 1.6163× |
| fresh-thin · 29 × 1229 × 103 | 53.43 | AOCL 84.57 µs · 1.5571× | AOCL 84.57 µs · 1.5571× |
| fresh-odd · 44 × 1897 × 151 | 156.18 | AOCL 220.99 µs · 1.3890× | AOCL 220.99 µs · 1.3890× |
| fresh-exact · 68 × 2176 × 244 | 562.50 | AOCL 540.36 µs · 0.9887× | AOCL 540.36 µs · 0.9887× |
| fresh-long · 83 × 3067 × 271 | 995.06 | OpenBLAS 1174.67 µs · 1.1681× | OpenBLAS 1174.67 µs · 1.1681× |
| fresh-high-depth · 112 × 3581 × 337 | 1617.49 | OpenBLAS 1752.21 µs · 1.0900× | OpenBLAS 1752.21 µs · 1.0900× |
| fresh-upper · 151 × 2529 × 311 | 1418.31 | OpenBLAS 1712.24 µs · 1.2019× | OpenBLAS 1712.24 µs · 1.2019× |
| control-cropped-deeper · 108 × 1728 × 320 | 741.47 | AOCL 822.42 µs · 1.1092× | AOCL 822.42 µs · 1.1092× |
| control-cropped-short · 43 × 2291 × 137 | 169.05 | AOCL 285.17 µs · 1.6866× | AOCL 285.17 µs · 1.6866× |

### Explicit handle execution and setup

All times are medians in microseconds. “Setup + first” takes the median of six within-worker sums. The single-call one-shot time includes create, obtain pointer, execute and destroy. Each prepared first and warm execution checks that the requested geometry matches the cached geometry and reuses the worker’s handle. Prepared destruction occurs after the measured products and is outside the setup-inclusive interval.

Explicit one-shot had 9 generated and 2 callable fallback layouts. Explicit prepared had 9 generated and 2 callable fallback layouts.

| Layout | One-shot / prepared API status | One-shot warm | One-shot first | Prepared warm | Setup | Ready first | Setup + first |
| --- | --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | MKL_JIT_SUCCESS | 747.35 | 1858.84 | 699.65 | 1069.35 | 734.45 | 1804.97 |
| repeat-ragged | MKL_NO_JIT | 1271.59 | 2503.49 | 1255.21 | 1040.37 | 1394.42 | 2412.07 |
| repeat-short | MKL_JIT_SUCCESS | 223.32 | 1267.51 | 156.53 | 1019.78 | 216.37 | 1232.41 |
| fresh-thin | MKL_JIT_SUCCESS | 66.18 | 1017.88 | 35.80 | 933.22 | 54.20 | 986.95 |
| fresh-odd | MKL_JIT_SUCCESS | 186.73 | 1256.92 | 144.17 | 1009.00 | 210.88 | 1200.50 |
| fresh-exact | MKL_JIT_SUCCESS | 477.37 | 1584.40 | 411.48 | 1109.02 | 435.58 | 1551.23 |
| fresh-long | MKL_JIT_SUCCESS | 1024.53 | 2155.42 | 911.87 | 991.00 | 1003.19 | 2000.95 |
| fresh-high-depth | MKL_JIT_SUCCESS | 1690.88 | 2879.65 | 1650.35 | 1023.26 | 1792.21 | 2830.45 |
| fresh-upper | MKL_NO_JIT | 1241.88 | 2399.50 | 1230.63 | 1044.08 | 1359.01 | 2400.55 |
| control-cropped-deeper | MKL_JIT_SUCCESS | 775.08 | 1898.51 | 733.86 | 1080.21 | 777.47 | 1860.12 |
| control-cropped-short | MKL_JIT_SUCCESS | 226.62 | 1320.86 | 154.67 | 1069.25 | 217.42 | 1283.70 |

### Explicit methods against oneMKL controls

These ratios divide the control time by the explicit method’s time. Values above one favor the explicit method. Each entry gives the geometric mean of six principal-layout paired medians, followed by the count of layouts whose paired median favors the explicit method.

| Explicit method | Control | Warm | First execution | Setup + first |
| --- | --- | --- | --- | --- |
| Explicit one-shot | oneMKL | 0.7455× · 0/6 | 0.9757× · 4/6 | 0.9757× · 4/6 |
| Explicit one-shot | oneMKL direct-call | 0.7486× · 0/6 | 0.9618× · 3/6 | 0.9618× · 3/6 |
| Explicit prepared | oneMKL | 0.9094× · 1/6 | 3.6134× · 6/6 | 1.0051× · 4/6 |
| Explicit prepared | oneMKL direct-call | 0.9094× · 1/6 | 3.5464× · 6/6 | 0.9899× · 4/6 |

### Generated prepared handle on one principal layout

fresh-thin · 29 × 1229 × 103 was the first principal layout in panel order whose prepared call returned MKL_JIT_SUCCESS. Creating the handle and obtaining its callable before the first product took a median 933.220 µs in the six measured workers.
The warm medians were 36.883 µs for ordinary oneMKL and 35.802 µs for the prepared public call. Subtracting prepared time from ordinary time in each matched round gave a median +1.028 µs per product, with 6/6 positive savings. The seed-block medians were +0.897 and +1.159 µs.
At least one round’s saving was within 1% of ordinary time or within the observed time range of one method in that seed block. The modeled reuse count would be sensitive to that small difference.

131/132 registered method/layout combinations qualified.

| Excluded method | Layouts | Recorded reason |
| --- | --- | --- |
| BLASFEO | repeat-ragged | worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 · missing, duplicate or invalid FP64 fixtures · missing, duplicate or invalid matched timings |

The installed COB release `a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6` used its AVX-512 kernel on this host. Its ordinary worker SHA-256 was `a1648abd3d90a1832001eff0656a472ba1d2ebf35bb32f19e25e2065efcbae61`.

The numerical checks examined 257,821,090 FP64-reference output values and 18,075,230 guard or padding locations. The largest recorded absolute error was 2.45527356e-05. Native workers completed 25,152 warm multiplies in 786 batches.

## Intel(R) Xeon(R) 6975P-C · replica 2

Raw wins compare COB’s median time with the independent leader’s median. Paired wins use the median of the six within-round leader/COB ratios. The seed columns give the three-round seed results aggregated across the principal layouts. The final column counts layouts whose leader, raw COB win, or paired COB win changed when the two policies entered their respective fields.

| Regime | Independent field | Paired geomean | COB raw wins | COB paired wins | COB round wins | Seed 1 / 2 | Complete fields | Changes (leader / raw / paired) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Warm | Original one-shot (7) | 0.8796× | 0/6 | 0/6 | 0/36 | 0.8789 / 0.8810 | 6/6 | reference |
| Warm | Expanded one-shot (8) | 0.8796× | 0/6 | 0/6 | 0/36 | 0.8789 / 0.8810 | 6/6 | 0 / 0 / 0 of 6 |
| Warm | Original incl. retained (9) | 0.8791× | 0/6 | 0/6 | 0/36 | 0.8775 / 0.8800 | 6/6 | reference |
| Warm | Expanded incl. retained (11) | 0.8791× | 0/6 | 0/6 | 0/36 | 0.8775 / 0.8800 | 6/6 | 0 / 0 / 0 of 6 |
| First execution | Original one-shot (7) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | reference |
| First execution | Expanded one-shot (8) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | 0 / 0 / 0 of 6 |
| First execution | Original incl. retained (9) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | reference |
| First execution | Expanded incl. retained (11) | 1.0071× | 4/6 | 4/6 | 20/36 | 1.0083 / 1.0031 | 6/6 | 6 / 1 / 1 of 6 |
| Setup + first | Original one-shot (7) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | reference |
| Setup + first | Expanded one-shot (8) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | 0 / 0 / 0 of 6 |
| Setup + first | Original incl. retained (9) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | reference |
| Setup + first | Expanded incl. retained (11) | 1.2251× | 5/6 | 5/6 | 32/36 | 1.2429 / 1.2218 | 6/6 | 0 / 0 / 0 of 6 |

### Fastest independent method on each layout

Each entry gives the original leader → the expanded leader, with its median and its paired time ratio against COB. A single entry means the same leader won both fields. Each regime selects its leaders from the measurements in that regime.

#### Warm

| Physical layout · M × N × K | COB µs | One-shot leaders (7 → 8) | Retained-inclusive leaders (9 → 11) |
| --- | --- | --- | --- |
| repeat-deeper · 108 × 1728 × 320 | 640.66 | oneMKL direct-call 663.34 µs · 1.0354× | oneDNN cached 649.29 µs · 1.0125× |
| repeat-ragged · 137 × 2179 × 347 | 1107.65 | OpenBLAS 1146.71 µs · 1.0356× | OpenBLAS 1146.71 µs · 1.0356× |
| repeat-short · 43 × 2291 × 137 | 151.87 | oneMKL direct-call 152.68 µs · 1.0001× | oneMKL direct-call 152.68 µs · 1.0001× → Explicit prepared 148.80 µs · 0.9788× |
| fresh-thin · 29 × 1229 × 103 | 41.23 | LIBXSMM dispatch 34.35 µs · 0.8344× | LIBXSMM retained 34.08 µs · 0.8316× |
| fresh-odd · 44 × 1897 × 151 | 140.78 | oneMKL direct-call 128.31 µs · 0.9095× | oneMKL direct-call 128.31 µs · 0.9095× |
| fresh-exact · 68 × 2176 × 244 | 410.76 | oneMKL direct-call 358.18 µs · 0.8726× | oneMKL direct-call 358.18 µs · 0.8726× |
| fresh-long · 83 × 3067 × 271 | 778.74 | oneMKL direct-call 651.37 µs · 0.8364× | oneMKL direct-call 651.37 µs · 0.8364× |
| fresh-high-depth · 112 × 3581 × 337 | 1455.37 | oneMKL direct-call 1321.18 µs · 0.9058× | oneMKL direct-call 1321.18 µs · 0.9058× |
| fresh-upper · 151 × 2529 × 311 | 1261.46 | oneMKL direct-call 1164.02 µs · 0.9233× | oneMKL direct-call 1164.02 µs · 0.9233× |
| control-cropped-deeper · 108 × 1728 × 320 | 645.70 | oneMKL direct-call 597.43 µs · 0.9245× | oneMKL direct-call 597.43 µs · 0.9245× |
| control-cropped-short · 43 × 2291 × 137 | 153.56 | oneMKL direct-call 138.19 µs · 0.8980× | oneMKL direct-call 138.19 µs · 0.8980× |

#### First execution

| Physical layout · M × N × K | COB µs | One-shot leaders (7 → 8) | Retained-inclusive leaders (9 → 11) |
| --- | --- | --- | --- |
| repeat-deeper · 108 × 1728 × 320 | 737.50 | AOCL 739.46 µs · 0.9901× | AOCL 739.46 µs · 0.9901× → Explicit prepared 719.09 µs · 0.9751× |
| repeat-ragged · 137 × 2179 × 347 | 1194.80 | OpenBLAS 1335.48 µs · 1.1016× | OpenBLAS 1335.48 µs · 1.1016× → Explicit prepared 1330.80 µs · 1.0987× |
| repeat-short · 43 × 2291 × 137 | 160.64 | AOCL 257.35 µs · 1.5801× | AOCL 257.35 µs · 1.5801× → Explicit prepared 192.71 µs · 1.1983× |
| fresh-thin · 29 × 1229 × 103 | 51.38 | AOCL 84.51 µs · 1.6497× | AOCL 84.51 µs · 1.6497× → Explicit prepared 52.34 µs · 1.0095× |
| fresh-odd · 44 × 1897 × 151 | 156.13 | AOCL 206.14 µs · 1.3165× | AOCL 206.14 µs · 1.3165× → Explicit prepared 182.87 µs · 1.2013× |
| fresh-exact · 68 × 2176 × 244 | 501.74 | AOCL 486.68 µs · 0.9596× | AOCL 486.68 µs · 0.9596× → Explicit prepared 426.57 µs · 0.8291× |
| fresh-long · 83 × 3067 × 271 | 869.92 | OpenBLAS 1072.77 µs · 1.2447× | OpenBLAS 1072.77 µs · 1.2447× → Explicit prepared 912.80 µs · 1.0457× |
| fresh-high-depth · 112 × 3581 × 337 | 1568.71 | OpenBLAS 1687.45 µs · 1.0756× | OpenBLAS 1687.45 µs · 1.0756× → Explicit prepared 1663.80 µs · 1.0635× |
| fresh-upper · 151 × 2529 × 311 | 1378.39 | OpenBLAS 1641.31 µs · 1.2115× | OpenBLAS 1641.31 µs · 1.2115× → Explicit prepared 1274.69 µs · 0.9333× |
| control-cropped-deeper · 108 × 1728 × 320 | 735.87 | AOCL 796.26 µs · 1.0734× | AOCL 796.26 µs · 1.0734× → Explicit prepared 756.17 µs · 1.0122× |
| control-cropped-short · 43 × 2291 × 137 | 163.98 | AOCL 279.68 µs · 1.6245× | AOCL 279.68 µs · 1.6245× → Explicit prepared 198.07 µs · 1.1950× |

#### Setup + first

| Physical layout · M × N × K | COB µs | One-shot leaders (7 → 8) | Retained-inclusive leaders (9 → 11) |
| --- | --- | --- | --- |
| repeat-deeper · 108 × 1728 × 320 | 737.50 | AOCL 739.46 µs · 0.9901× | AOCL 739.46 µs · 0.9901× |
| repeat-ragged · 137 × 2179 × 347 | 1194.80 | OpenBLAS 1335.48 µs · 1.1016× | OpenBLAS 1335.48 µs · 1.1016× |
| repeat-short · 43 × 2291 × 137 | 160.64 | AOCL 257.35 µs · 1.5801× | AOCL 257.35 µs · 1.5801× |
| fresh-thin · 29 × 1229 × 103 | 51.38 | AOCL 84.51 µs · 1.6497× | AOCL 84.51 µs · 1.6497× |
| fresh-odd · 44 × 1897 × 151 | 156.13 | AOCL 206.14 µs · 1.3165× | AOCL 206.14 µs · 1.3165× |
| fresh-exact · 68 × 2176 × 244 | 501.74 | AOCL 486.68 µs · 0.9596× | AOCL 486.68 µs · 0.9596× |
| fresh-long · 83 × 3067 × 271 | 869.92 | OpenBLAS 1072.77 µs · 1.2447× | OpenBLAS 1072.77 µs · 1.2447× |
| fresh-high-depth · 112 × 3581 × 337 | 1568.71 | OpenBLAS 1687.45 µs · 1.0756× | OpenBLAS 1687.45 µs · 1.0756× |
| fresh-upper · 151 × 2529 × 311 | 1378.39 | OpenBLAS 1641.31 µs · 1.2115× | OpenBLAS 1641.31 µs · 1.2115× |
| control-cropped-deeper · 108 × 1728 × 320 | 735.87 | AOCL 796.26 µs · 1.0734× | AOCL 796.26 µs · 1.0734× |
| control-cropped-short · 43 × 2291 × 137 | 163.98 | AOCL 279.68 µs · 1.6245× | AOCL 279.68 µs · 1.6245× |

### Explicit handle execution and setup

All times are medians in microseconds. “Setup + first” takes the median of six within-worker sums. The single-call one-shot time includes create, obtain pointer, execute and destroy. Each prepared first and warm execution checks that the requested geometry matches the cached geometry and reuses the worker’s handle. Prepared destruction occurs after the measured products and is outside the setup-inclusive interval.

Explicit one-shot had 9 generated and 2 callable fallback layouts. Explicit prepared had 9 generated and 2 callable fallback layouts.

| Layout | One-shot / prepared API status | One-shot warm | One-shot first | Prepared warm | Setup | Ready first | Setup + first |
| --- | --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | MKL_JIT_SUCCESS | 706.71 | 1850.95 | 666.93 | 1089.42 | 719.09 | 1812.76 |
| repeat-ragged | MKL_NO_JIT | 1223.71 | 2375.36 | 1203.89 | 1043.75 | 1330.80 | 2365.40 |
| repeat-short | MKL_JIT_SUCCESS | 217.43 | 1244.19 | 148.80 | 1024.11 | 192.71 | 1215.41 |
| fresh-thin | MKL_JIT_SUCCESS | 64.50 | 1032.61 | 34.30 | 964.92 | 52.34 | 1036.66 |
| fresh-odd | MKL_JIT_SUCCESS | 179.24 | 1188.58 | 136.04 | 997.51 | 182.87 | 1178.80 |
| fresh-exact | MKL_JIT_SUCCESS | 447.67 | 1529.15 | 369.62 | 1093.48 | 426.57 | 1506.62 |
| fresh-long | MKL_JIT_SUCCESS | 969.53 | 2090.99 | 845.49 | 1081.96 | 912.80 | 1997.29 |
| fresh-high-depth | MKL_JIT_SUCCESS | 1613.83 | 2799.58 | 1558.55 | 1099.85 | 1663.80 | 2763.65 |
| fresh-upper | MKL_NO_JIT | 1188.57 | 2356.54 | 1169.71 | 1043.96 | 1274.69 | 2309.87 |
| control-cropped-deeper | MKL_JIT_SUCCESS | 741.06 | 1854.14 | 700.93 | 1061.51 | 756.17 | 1822.43 |
| control-cropped-short | MKL_JIT_SUCCESS | 216.52 | 1279.45 | 149.36 | 1079.33 | 198.07 | 1268.86 |

### Explicit methods against oneMKL controls

These ratios divide the control time by the explicit method’s time. Values above one favor the explicit method. Each entry gives the geometric mean of six principal-layout paired medians, followed by the count of layouts whose paired median favors the explicit method.

| Explicit method | Control | Warm | First execution | Setup + first |
| --- | --- | --- | --- | --- |
| Explicit one-shot | oneMKL | 0.7470× · 0/6 | 0.9839× · 4/6 | 0.9839× · 4/6 |
| Explicit one-shot | oneMKL direct-call | 0.7455× · 0/6 | 0.9692× · 3/6 | 0.9692× · 3/6 |
| Explicit prepared | oneMKL | 0.9234× · 1/6 | 3.7510× · 6/6 | 0.9916× · 4/6 |
| Explicit prepared | oneMKL direct-call | 0.9243× · 1/6 | 3.6707× · 6/6 | 0.9911× · 3/6 |

### Generated prepared handle on one principal layout

fresh-thin · 29 × 1229 × 103 was the first principal layout in panel order whose prepared call returned MKL_JIT_SUCCESS. Creating the handle and obtaining its callable before the first product took a median 964.921 µs in the six measured workers.
The warm medians were 35.320 µs for ordinary oneMKL and 34.303 µs for the prepared public call. Subtracting prepared time from ordinary time in each matched round gave a median +0.948 µs per product, with 6/6 positive savings. The seed-block medians were +1.590 and +0.839 µs.
At least one round’s saving was within 1% of ordinary time or within the observed time range of one method in that seed block. The modeled reuse count would be sensitive to that small difference.

131/132 registered method/layout combinations qualified.

| Excluded method | Layouts | Recorded reason |
| --- | --- | --- |
| BLASFEO | repeat-ragged | worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 · missing, duplicate or invalid FP64 fixtures · missing, duplicate or invalid matched timings |

The installed COB release `a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6` used its AVX-512 kernel on this host. Its ordinary worker SHA-256 was `a1648abd3d90a1832001eff0656a472ba1d2ebf35bb32f19e25e2065efcbae61`.

The numerical checks examined 257,821,090 FP64-reference output values and 18,075,230 guard or padding locations. The largest recorded absolute error was 2.45527356e-05. Native workers completed 25,152 warm multiplies in 786 batches.

The [comparison CSV](comparison.csv) records each qualified method’s absolute measurements and all six ratios for every COB comparison, along with the four explicit-versus-control comparisons per layout. The ready-handle rows include setup and first execution from each matched worker. The recorded timings describe these processors, geometries and call policies.

From the extracted evidence directory, run `python3 -B packet.py .` to replay the verification, rankings, report and CSV.

[Benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/0fcba28139f3d0f64492d5e9938f50d32280625d/tools/world-rank/handles) · [Native workflow](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/0fcba28139f3d0f64492d5e9938f50d32280625d/.github/workflows/native-gemm.yml#L543)
