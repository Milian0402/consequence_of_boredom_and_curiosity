# Checked signed INT8 for exact integer SGEMM

The new dispatcher accepts ordinary row-major FP32 matrices. For a supported geometry and eligible input values it uses a signed INT8 product with 32-bit accumulation and writes FP32 results. Other calls go through the selected FP32 backing policy. The previous checked BF16 dispatcher was measured on the same inputs and in the same scheduling rounds.

- **Intel(R) Xeon(R) 6975P-C.** All 7 candidate fixture outcomes verified. On 4 verified native fixtures, the candidate cleared the paired improvement rule against the fastest qualifying one-shot FP32 opponent on 3/4 first and 3/4 warm comparisons. [Hosted job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553/job/139815068937)
- **Neoverse-N2.** All 7 candidate fixture outcomes verified. On 4 verified native fixtures, the candidate cleared the paired improvement rule against the fastest qualifying one-shot FP32 opponent on 4/4 first and 4/4 warm comparisons. [Hosted job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553/job/139815068579)

### New 3584³ square

The large-effect test compares the exact native signed result with the fastest numerically qualified ordinary FP32 method, including retained plans. Both the median of six warm ratios and the ratio of separate warm medians must reach 2.00×, with at least five strict wins.

| CPU | 2× gate | Verified exact native | Checked S8 ms | Strongest FP32, ms | Ratio of medians | Paired median | Wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intel(R) Xeon(R) 6975P-C | passed | yes | 137.646 | `release-mkl-aligned-two` 410.580 | 2.9829× | 2.9778× | 6/6 |
| Neoverse-N2 | passed | yes | 300.245 | `fastmmw-768` 1,447.279 | 4.8203× | 4.8069× | 6/6 |

## What was measured

The seven fixtures include signed 4096³ and 3584³ cubes, a cropped 1079 × 1553 × 1201 signed rectangle, and two copies of that rectangle with ½ added to the first A value or last B value. The UCI handwritten-digit fixtures multiply 1,797 test examples against 3,823 training examples at K = 64, and 1,797 against 1,934 binary images at K = 1,024. The K = 64 fixture exercises the new shape-only FP32 bypass for K ≤ 128. Both digit products come from one collection. [UCI dataset](https://doi.org/10.24432/C50P49) · [Frozen transformations and hashes](../fixtures/ATTRIBUTION.md)

The signed path checks input integrality, signed-byte range, and K × max|A| × max|B| ≤ 2²⁴. That last bound keeps the result and all partial integer sums exactly representable in FP32. The application decides from runtime CPU capability, input values and the actual oneDNN primitive. It requires AMX INT8 with usable tiles on x86 or SVE I8MM on Arm. Each accelerated call validates and packs current inputs, creates the signed primitive, and writes FP32 output. An independently recorded geometry-specific execution and opcode witness is used to verify native attribution. The raw BF16 comparator skips the application’s integer certificate. The signed oneDNN specialist checks its input and can abort on an unsupported call. It charges conversion, primitive construction, multiplication and output conversion. The popcount specialist packs the binary input in the timed operation. The earlier checked BF16 implementation is the one frozen in the [original experiment](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642).

Within each fixture, the six ratios correspond to three matched rounds at seed 2341279043 and three at 916583729. The real corpus operands remain fixed between blocks. Each worker times its first complete call after initialization. It checks and hashes the full output, poisons C again, and times the second complete call in the same process. “Warm” denotes that second call. Cache residency was unmeasured. Every time ratio divides opponent time by S8-policy time. A median paired ratio above one favors the new policy. The scoped improvement rule requires a median above 1.02× and at least five strict wins in six paired rounds. Each architecture has one native job.

Numerical admission checks every output for finiteness, absolute error ≤ 0.002 and error ≤ K/(2²⁴−K) times the sum of absolute products. Exact qualification additionally requires zero deviation. Exact native product, numerical FP32 fallback and numerical admission of an independent peer are recorded separately. [Every method, raw timing and paired ratio](peer-comparisons.csv)

## Intel(R) Xeon(R) 6975P-C

Six fresh workers per active method and fixture produced 666 process observations on logical CPU 0. [Job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553/job/139815068937) · Raw timing records (`x86_64/raw/observations.jsonl` in [evidence.zip](evidence.zip)) · Independent audit (`x86_64/audit.json` in [evidence.zip](evidence.zip))

### Checked S8 versus the previous checked BF16 policy

Times are raw medians in milliseconds. Each ratio divides the previous policy’s time by the S8 policy’s time on the same seed and round. Both policies keep their own fallback behavior.

| Fixture | Call | Checked S8 ms | Checked BF16 ms | Median paired ratio (wins) | Six paired BF16/S8 ratios |
| --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | 222.135 | 336.712 | 1.507× (6/6) | 1.4669, 1.4568, 1.5473, 1.4604, 1.5463, 1.5493 |
| Signed 4096³ | warm | 220.830 | 341.508 | 1.544× (6/6) | 1.5020, 1.4718, 1.5686, 1.5369, 1.5507, 1.5652 |
| Signed 3584³ | first | 163.365 | 235.501 | 1.431× (6/6) | 1.4607, 1.4607, 1.4000, 1.4062, 1.4478, 1.4138 |
| Signed 3584³ | warm | 137.646 | 233.381 | 1.688× (6/6) | 1.6840, 1.7475, 1.6517, 1.7310, 1.6868, 1.6897 |
| Signed rectangle | first | 24.540 | 23.527 | 0.955× (1/6) | 0.9534, 0.9483, 0.9340, 0.9566, 1.0324, 0.9770 |
| Signed rectangle | warm | 22.569 | 23.734 | 1.037× (6/6) | 1.0373, 1.0228, 1.0432, 1.0717, 1.0255, 1.0368 |
| First A + ½ | first | 21.915 | 21.429 | 0.980× (0/6) | 0.9707, 0.9788, 0.9928, 0.9805, 0.9624, 0.9917 |
| First A + ½ | warm | 19.363 | 19.335 | 0.997× (2/6) | 0.9967, 1.0011, 0.9979, 0.9962, 0.9953, 1.0032 |
| Last B + ½ | first | 30.233 | 22.459 | 0.743× (0/6) | 0.7396, 0.7390, 0.7342, 0.7460, 0.7469, 0.7501 |
| Last B + ½ | warm | 27.516 | 20.453 | 0.741× (0/6) | 0.7384, 0.7426, 0.7339, 0.7460, 0.7481, 0.7389 |
| UCI feature counts | first | 6.426 | 19.007 | 2.920× (6/6) | 3.0209, 2.8000, 2.8251, 2.9015, 2.9692, 2.9394 |
| UCI feature counts | warm | 4.934 | 17.093 | 3.492× (6/6) | 3.6160, 3.4826, 3.2970, 3.3711, 3.5015, 3.5516 |
| UCI binary pixels | first | 28.486 | 30.897 | 1.073× (6/6) | 1.0508, 1.0903, 1.0747, 1.1079, 1.0710, 1.0477 |
| UCI binary pixels | warm | 26.226 | 23.873 | 0.924× (0/6) | 0.9033, 0.9501, 0.9114, 0.9359, 0.8887, 0.9416 |

### Qualified comparisons by field

Each entry gives the fastest other eligible method by raw median time, its milliseconds, and its paired time ratio against checked S8. One-shot entries charge per-call setup. The retained-inclusive fields also admit oneDNN’s cached plan.

| Fixture | Call | One-shot FP32 | FP32 incl. retained | Exact one-shot FP32 | Exact FP32 incl. retained |
| --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | `fp32-policy` 605.134 ms · 2.727× (6/6) | `fp32-policy` 605.134 ms · 2.727× (6/6) | `fp32-policy` 605.134 ms · 2.727× (6/6) | `fp32-policy` 605.134 ms · 2.727× (6/6) |
| Signed 4096³ | warm | `fp32-policy` 602.859 ms · 2.734× (6/6) | `fp32-policy` 602.859 ms · 2.734× (6/6) | `fp32-policy` 602.859 ms · 2.734× (6/6) | `fp32-policy` 602.859 ms · 2.734× (6/6) |
| Signed 3584³ | first | `release-mkl-aligned-two` 414.433 ms · 2.522× (6/6) | `release-mkl-aligned-two` 414.433 ms · 2.522× (6/6) | `release-mkl-aligned-two` 414.433 ms · 2.522× (6/6) | `release-mkl-aligned-two` 414.433 ms · 2.522× (6/6) |
| Signed 3584³ | warm | `release-mkl-aligned-two` 410.580 ms · 2.978× (6/6) | `release-mkl-aligned-two` 410.580 ms · 2.978× (6/6) | `release-mkl-aligned-two` 410.580 ms · 2.978× (6/6) | `release-mkl-aligned-two` 410.580 ms · 2.978× (6/6) |
| Signed rectangle | first | `openblas` 19.850 ms · 0.802× (0/6) | `openblas` 19.850 ms · 0.802× (0/6) | `openblas` 19.850 ms · 0.802× (0/6) | `openblas` 19.850 ms · 0.802× (0/6) |
| Signed rectangle | warm | `openblas` 19.219 ms · 0.850× (0/6) | `openblas` 19.219 ms · 0.850× (0/6) | `openblas` 19.219 ms · 0.850× (0/6) | `openblas` 19.219 ms · 0.850× (0/6) |
| First A + ½ | first | `openblas` 19.774 ms · 0.904× (0/6) | `openblas` 19.774 ms · 0.904× (0/6) | `openblas` 19.774 ms · 0.904× (0/6) | `openblas` 19.774 ms · 0.904× (0/6) |
| First A + ½ | warm | `fastmmw-1536` 19.120 ms · 0.987× (2/6) | `fastmmw-1536` 19.120 ms · 0.987× (2/6) | `fastmmw-1536` 19.120 ms · 0.987× (2/6) | `fastmmw-1536` 19.120 ms · 0.987× (2/6) |
| Last B + ½ | first | `fastmmw-1536` 19.756 ms · 0.656× (0/6) | `fastmmw-1536` 19.756 ms · 0.656× (0/6) | `fastmmw-1536` 19.756 ms · 0.656× (0/6) | `fastmmw-1536` 19.756 ms · 0.656× (0/6) |
| Last B + ½ | warm | `fastmmw-1536` 19.170 ms · 0.695× (0/6) | `fastmmw-1536` 19.170 ms · 0.695× (0/6) | `fastmmw-1536` 19.170 ms · 0.695× (0/6) | `fastmmw-1536` 19.170 ms · 0.695× (0/6) |
| UCI feature counts | first | `cob-standalone` 5.597 ms · 0.860× (0/6) | `cob-standalone` 5.597 ms · 0.860× (0/6) | `cob-standalone` 5.597 ms · 0.860× (0/6) | `cob-standalone` 5.597 ms · 0.860× (0/6) |
| UCI feature counts | warm | `fp32-policy` 4.841 ms · 0.994× (3/6) | `fp32-policy` 4.841 ms · 0.994× (3/6) | `fp32-policy` 4.841 ms · 0.994× (3/6) | `fp32-policy` 4.841 ms · 0.994× (3/6) |
| UCI binary pixels | first | `openblas` 35.284 ms · 1.238× (6/6) | `openblas` 35.284 ms · 1.238× (6/6) | `openblas` 35.284 ms · 1.238× (6/6) | `openblas` 35.284 ms · 1.238× (6/6) |
| UCI binary pixels | warm | `mkl-jit` 33.610 ms · 1.282× (6/6) | `mkl-jit` 33.610 ms · 1.282× (6/6) | `mkl-jit` 33.610 ms · 1.282× (6/6) | `mkl-jit` 33.610 ms · 1.282× (6/6) |

| Fixture | Call | Independent one-shot FP32 | Independent FP32 incl. retained | Exact integer/binary specialist | Other checked/backing policy |
| --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | `openblas` 634.297 ms · 2.856× (6/6) | `openblas` 634.297 ms · 2.856× (6/6) | `onednn-int8` 203.033 ms · 0.916× (0/6) | `exact-dispatch` 336.712 ms · 1.507× (6/6) |
| Signed 4096³ | warm | `fastmmw-1536` 620.326 ms · 2.815× (6/6) | `fastmmw-1536` 620.326 ms · 2.815× (6/6) | `onednn-int8` 162.552 ms · 0.739× (0/6) | `exact-dispatch` 341.508 ms · 1.544× (6/6) |
| Signed 3584³ | first | `openblas` 426.853 ms · 2.611× (6/6) | `openblas` 426.853 ms · 2.611× (6/6) | `onednn-int8` 148.821 ms · 0.910× (0/6) | `exact-dispatch` 235.501 ms · 1.431× (6/6) |
| Signed 3584³ | warm | `fastmmw-1536` 423.777 ms · 3.068× (6/6) | `fastmmw-1536` 423.777 ms · 3.068× (6/6) | `onednn-int8` 146.080 ms · 1.055× (6/6) | `exact-dispatch` 233.381 ms · 1.688× (6/6) |
| Signed rectangle | first | `openblas` 19.850 ms · 0.802× (0/6) | `openblas` 19.850 ms · 0.802× (0/6) | `onednn-int8` 21.095 ms · 0.864× (0/6) | `fp32-policy` 21.533 ms · 0.870× (0/6) |
| Signed rectangle | warm | `openblas` 19.219 ms · 0.850× (0/6) | `openblas` 19.219 ms · 0.850× (0/6) | `onednn-int8` 19.787 ms · 0.875× (0/6) | `fp32-policy` 19.390 ms · 0.859× (0/6) |
| First A + ½ | first | `openblas` 19.774 ms · 0.904× (0/6) | `openblas` 19.774 ms · 0.904× (0/6) | — | `fp32-policy` 21.349 ms · 0.973× (1/6) |
| First A + ½ | warm | `fastmmw-1536` 19.120 ms · 0.987× (2/6) | `fastmmw-1536` 19.120 ms · 0.987× (2/6) | — | `exact-dispatch` 19.335 ms · 0.997× (2/6) |
| Last B + ½ | first | `fastmmw-1536` 19.756 ms · 0.656× (0/6) | `fastmmw-1536` 19.756 ms · 0.656× (0/6) | — | `fp32-policy` 21.358 ms · 0.706× (0/6) |
| Last B + ½ | warm | `fastmmw-1536` 19.170 ms · 0.695× (0/6) | `fastmmw-1536` 19.170 ms · 0.695× (0/6) | — | `fp32-policy` 19.329 ms · 0.699× (0/6) |
| UCI feature counts | first | `fastmmw-1536` 6.124 ms · 0.929× (1/6) | `fastmmw-1536` 6.124 ms · 0.929× (1/6) | `onednn-int8` 18.347 ms · 2.759× (6/6) | `fp32-policy` 6.454 ms · 1.006× (3/6) |
| UCI feature counts | warm | `mkl-jit` 4.859 ms · 0.990× (1/6) | `mkl-jit` 4.859 ms · 0.990× (1/6) | `onednn-int8` 15.453 ms · 3.135× (6/6) | `fp32-policy` 4.841 ms · 0.994× (3/6) |
| UCI binary pixels | first | `openblas` 35.284 ms · 1.238× (6/6) | `openblas` 35.284 ms · 1.238× (6/6) | `bitset-popcount` 20.265 ms · 0.719× (0/6) | `exact-dispatch` 30.897 ms · 1.073× (6/6) |
| UCI binary pixels | warm | `mkl-jit` 33.610 ms · 1.282× (6/6) | `mkl-jit` 33.610 ms · 1.282× (6/6) | `bitset-popcount` 18.959 ms · 0.728× (0/6) | `exact-dispatch` 23.873 ms · 0.924× (0/6) |

### Complete field and fixed controls

The full-field winner column includes checked S8 itself. The next column compares it with the fastest eligible rival. Each fixed-control cell gives the peer’s milliseconds, paired ratio and strict wins for checked S8. The binary specialist is eligible on the pixel fixture.

| Fixture | Call | Whole-field winner, ms | Fastest other eligible peer | Backing FP32 | Standalone COB | Unchecked BF16 | oneDNN S8 specialist | Binary AND/popcount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | `onednn-int8` 203.033 | `onednn-int8` 203.033 ms · 0.916× (0/6) | 605.134 · 2.727× (6/6) | 676.461 · 3.035× (6/6) | 309.289 · 1.397× (6/6) | 203.033 · 0.916× (0/6) | — |
| Signed 4096³ | warm | `onednn-int8` 162.552 | `onednn-int8` 162.552 ms · 0.739× (0/6) | 602.859 · 2.734× (6/6) | 682.098 · 3.092× (6/6) | 312.529 · 1.419× (6/6) | 162.552 · 0.739× (0/6) | — |
| Signed 3584³ | first | `onednn-int8` 148.821 | `onednn-int8` 148.821 ms · 0.910× (0/6) | 438.945 · 2.675× (6/6) | 472.284 · 2.889× (6/6) | 222.865 · 1.349× (6/6) | 148.821 · 0.910× (0/6) | — |
| Signed 3584³ | warm | `s8-dispatch` 137.646 | `onednn-int8` 146.080 ms · 1.055× (6/6) | 436.058 · 3.143× (6/6) | 471.373 · 3.396× (6/6) | 219.822 · 1.599× (6/6) | 146.080 · 1.055× (6/6) | — |
| Signed rectangle | first | `openblas` 19.850 | `openblas` 19.850 ms · 0.802× (0/6) | 21.533 · 0.870× (0/6) | 24.169 · 0.975× (0/6) | 22.569 · 0.911× (0/6) | 21.095 · 0.864× (0/6) | — |
| Signed rectangle | warm | `openblas` 19.219 | `openblas` 19.219 ms · 0.850× (0/6) | 19.390 · 0.859× (0/6) | 23.659 · 1.048× (5/6) | 22.033 · 0.969× (0/6) | 19.787 · 0.875× (0/6) | — |
| First A + ½ | first | `openblas` 19.774 | `openblas` 19.774 ms · 0.904× (0/6) | 21.349 · 0.973× (1/6) | 24.011 · 1.092× (6/6) | 22.556 · 1.031× (6/6) | — | — |
| First A + ½ | warm | `fastmmw-1536` 19.120 | `fastmmw-1536` 19.120 ms · 0.987× (2/6) | 19.410 · 1.002× (3/6) | 23.590 · 1.218× (6/6) | 21.182 · 1.095× (6/6) | — | — |
| Last B + ½ | first | `fastmmw-1536` 19.756 | `fastmmw-1536` 19.756 ms · 0.656× (0/6) | 21.358 · 0.706× (0/6) | 24.269 · 0.805× (0/6) | 22.482 · 0.747× (0/6) | — | — |
| Last B + ½ | warm | `fastmmw-1536` 19.170 | `fastmmw-1536` 19.170 ms · 0.695× (0/6) | 19.329 · 0.699× (0/6) | 23.732 · 0.859× (0/6) | 21.416 · 0.773× (0/6) | — | — |
| UCI feature counts | first | `cob-standalone` 5.597 | `cob-standalone` 5.597 ms · 0.860× (0/6) | 6.454 · 1.006× (3/6) | 5.597 · 0.860× (0/6) | 19.030 · 2.967× (6/6) | 18.347 · 2.759× (6/6) | — |
| UCI feature counts | warm | `fp32-policy` 4.841 | `fp32-policy` 4.841 ms · 0.994× (3/6) | 4.841 · 0.994× (3/6) | 5.576 · 1.130× (6/6) | 17.163 · 3.464× (6/6) | 15.453 · 3.135× (6/6) | — |
| UCI binary pixels | first | `bitset-popcount` 20.265 | `bitset-popcount` 20.265 ms · 0.719× (0/6) | 36.509 · 1.270× (6/6) | 39.724 · 1.396× (6/6) | 28.318 · 0.999× (2/6) | 25.297 · 0.893× (0/6) | 20.265 · 0.719× (0/6) |
| UCI binary pixels | warm | `bitset-popcount` 18.959 | `bitset-popcount` 18.959 ms · 0.728× (0/6) | 33.735 · 1.286× (6/6) | 39.013 · 1.512× (6/6) | 28.223 · 1.080× (6/6) | 23.063 · 0.918× (0/6) | 18.959 · 0.728× (0/6) |

### Runtime routes and numerical checks

The independent signed preflight tests the backend on the geometry. The public candidate makes its own decision on each call. The table lists observed branch reasons, the number of logical input elements inspected per call, and the signed native result expected by the frozen scorer. A checked S8 fallback is compared bit-for-bit with the FP32 backing call.

| Fixture | M × N × K | Backing FP32 | Signed capability | Checked probe | Expected native product | Primary outcome | Observed public call | Inexact / checked outputs | Max absolute error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Signed 4096³ | 4096 × 4096 × 4096 | `release-mkl-aligned-two` | yes | passed | yes | verified-native | native-exact (33,554,432 inspected, 12 receipts) · `brg_matmul:avx10_1_512_amx_fp16` | 0 / 201,326,592 | 0 |
| Signed 3584³ | 3584 × 3584 × 3584 | `mkl-jit` | yes | passed | yes | verified-native | native-exact (25,690,112 inspected, 12 receipts) · `brg_matmul:avx10_1_512_amx_fp16` | 0 / 154,140,672 | 0 |
| Signed rectangle | 1079 × 1553 × 1201 | `mkl-jit` | yes | passed | yes | verified-native | native-exact (3,161,032 inspected, 12 receipts) · `brg_matmul:avx10_1_512_amx_fp16` | 0 / 20,108,244 | 0 |
| First A + ½ | 1079 × 1553 × 1201 | `mkl-jit` | yes | passed | no | verified-fallback | integer-domain (64 inspected, 12 receipts) | 0 / 20,108,244 | 0 |
| Last B + ½ | 1079 × 1553 × 1201 | `mkl-jit` | yes | passed | no | verified-fallback | integer-domain (3,161,032 inspected, 12 receipts) | 0 / 20,108,244 | 0 |
| UCI feature counts | 1797 × 3823 × 64 | `mkl-jit` | yes | passed | no | verified-fallback | depth-shallow (0 inspected, 12 receipts) | 0 / 82,439,172 | 0 |
| UCI binary pixels | 1797 × 1934 × 1024 | `mkl-jit` | yes | passed | yes | verified-native | native-exact (3,820,544 inspected, 12 receipts) · `brg_matmul:avx10_1_512_amx_fp16` | 0 / 41,704,776 | 0 |

The independent primitive probes report the following implementations and arithmetic instruction evidence. Instruction names identify code in the retained probe evidence. They do not count retired instructions in the timed workers.

| Independently probed geometry | Checked S8 operator | Signed specialist | oneDNN signed primitive | Raw BF16 |
| --- | --- | --- | --- | --- |
| Signed 4096³ | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (5632 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (5632 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` |
| Signed 3584³ | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (5376 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (5376 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` |
| Signed rectangle | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (33668 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (33668 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` |
| UCI feature counts | FP32 fallback | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (1437 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` |
| UCI binary pixels | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (6192 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` (6192 sites) | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` |

| Timed adapter | Binary SHA-256 |
| --- | --- |
| `s8-dispatch` | `baeccafe5e367aa4fa6491c49939c1d246c41b37c423fcdfab9245cebec43ce4` |
| `exact-dispatch` | `701384002045458115efb0ec8180c0f840259b5e1778dd6328e69dafd6bf2427` |
| `raw-bf16` | `c2552d365fe19885001059d770757c0bba8387665d1db2eafd342febaa18fbae` |
| `onednn-int8` | `53d00f0b33444f1463544eadf6f04cbe5eef819139b24c36931acf19909c145b` |
| `bitset-popcount` | `9b2a94011ab6bd449a9cfc5b101501dd2bef6b0ac4a19f4901c02115e37bfcef` |
| `fp32-policy` | `f8322fb640d86387dda6a2992cd27e0f760e3953a8a574053b77cb3daa67d727` |

The audit checked 8,598,758,616 logical output values and 22,625,064 padding or guard locations in 1,320 timed result checks. It recorded 36 bit-identical S8-policy fallback comparisons.

### Qualification by method

| Method | Outcome | Fixtures |
| --- | --- | --- |
| `bitset-popcount` | outside registered binary image workload | Signed 4096³, Signed 3584³, Signed rectangle, First A + ½, Last B + ½, UCI feature counts |
| `onednn-int8` | fractional input outside INT8 domain | First A + ½, Last B + ½ |
| `blasfeo` | worker exit -11: BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 | UCI binary pixels |

Scored observations (`x86_64/raw/score.json` in [evidence.zip](evidence.zip)) · Field comparisons (`x86_64/raw/comparisons.csv` in [evidence.zip](evidence.zip)) · Admission receipts (`x86_64/raw/admissions.json` in [evidence.zip](evidence.zip))

## Neoverse-N2

Six fresh workers per active method and fixture produced 540 process observations on logical CPU 0. [Job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553/job/139815068579) · Raw timing records (`aarch64/raw/observations.jsonl` in [evidence.zip](evidence.zip)) · Independent audit (`aarch64/audit.json` in [evidence.zip](evidence.zip))

### Checked S8 versus the previous checked BF16 policy

Times are raw medians in milliseconds. Each ratio divides the previous policy’s time by the S8 policy’s time on the same seed and round. Both policies keep their own fallback behavior.

| Fixture | Call | Checked S8 ms | Checked BF16 ms | Median paired ratio (wins) | Six paired BF16/S8 ratios |
| --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | 485.556 | 892.736 | 1.842× (6/6) | 1.8383, 1.9840, 1.8446, 1.8125, 1.8403, 1.9079 |
| Signed 4096³ | warm | 476.943 | 877.531 | 1.876× (6/6) | 1.8678, 1.9771, 1.8836, 1.8274, 1.7717, 1.9182 |
| Signed 3584³ | first | 299.822 | 591.308 | 1.982× (6/6) | 1.9077, 1.9858, 1.8520, 1.9890, 1.9785, 1.9862 |
| Signed 3584³ | warm | 300.245 | 598.694 | 1.995× (6/6) | 1.9501, 1.9988, 1.8754, 2.0023, 1.9919, 2.0042 |
| Signed rectangle | first | 21.130 | 35.534 | 1.680× (6/6) | 1.6797, 1.6810, 1.6772, 1.6832, 1.6752, 1.6929 |
| Signed rectangle | warm | 19.043 | 33.878 | 1.770× (6/6) | 1.7758, 1.7816, 1.7418, 1.7581, 1.7648, 1.7766 |
| First A + ½ | first | 81.978 | 81.404 | 0.992× (1/6) | 0.9906, 1.0046, 0.9902, 0.9924, 0.9912, 0.9982 |
| First A + ½ | warm | 79.484 | 79.224 | 0.998× (3/6) | 1.0001, 1.0125, 0.9953, 0.9935, 0.9933, 1.0004 |
| Last B + ½ | first | 86.822 | 83.413 | 0.960× (1/6) | 0.9608, 0.9572, 1.0133, 0.9586, 0.9496, 0.9641 |
| Last B + ½ | warm | 84.270 | 81.350 | 0.966× (0/6) | 0.9627, 0.9640, 0.9801, 0.9685, 0.9689, 0.9612 |
| UCI feature counts | first | 18.205 | 17.992 | 0.987× (2/6) | 0.9543, 0.9755, 0.9911, 1.0214, 1.0069, 0.9819 |
| UCI feature counts | warm | 17.939 | 16.042 | 0.895× (0/6) | 0.8714, 0.8917, 0.9041, 0.9087, 0.8990, 0.8910 |
| UCI binary pixels | first | 34.496 | 58.014 | 1.691× (6/6) | 1.6748, 1.7088, 1.6787, 1.7031, 1.4645, 1.7293 |
| UCI binary pixels | warm | 31.413 | 55.141 | 1.756× (6/6) | 1.7576, 1.7544, 1.7475, 1.7779, 1.5264, 1.7569 |

### Qualified comparisons by field

Each entry gives the fastest other eligible method by raw median time, its milliseconds, and its paired time ratio against checked S8. One-shot entries charge per-call setup. The retained-inclusive fields also admit oneDNN’s cached plan.

| Fixture | Call | One-shot FP32 | FP32 incl. retained | Exact one-shot FP32 | Exact FP32 incl. retained |
| --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | `fastmmw-1536` 2,172.941 ms · 4.464× (6/6) | `fastmmw-1536` 2,172.941 ms · 4.464× (6/6) | `fastmmw-1536` 2,172.941 ms · 4.464× (6/6) | `fastmmw-1536` 2,172.941 ms · 4.464× (6/6) |
| Signed 4096³ | warm | `fp32-policy` 2,149.978 ms · 4.524× (6/6) | `fp32-policy` 2,149.978 ms · 4.524× (6/6) | `fp32-policy` 2,149.978 ms · 4.524× (6/6) | `fp32-policy` 2,149.978 ms · 4.524× (6/6) |
| Signed 3584³ | first | `fastmmw-768` 1,470.781 ms · 4.916× (6/6) | `fastmmw-768` 1,470.781 ms · 4.916× (6/6) | `fastmmw-768` 1,470.781 ms · 4.916× (6/6) | `fastmmw-768` 1,470.781 ms · 4.916× (6/6) |
| Signed 3584³ | warm | `fastmmw-768` 1,447.279 ms · 4.807× (6/6) | `fastmmw-768` 1,447.279 ms · 4.807× (6/6) | `fastmmw-768` 1,447.279 ms · 4.807× (6/6) | `fastmmw-768` 1,447.279 ms · 4.807× (6/6) |
| Signed rectangle | first | `cob-standalone` 77.014 ms · 3.641× (6/6) | `cob-standalone` 77.014 ms · 3.641× (6/6) | `cob-standalone` 77.014 ms · 3.641× (6/6) | `cob-standalone` 77.014 ms · 3.641× (6/6) |
| Signed rectangle | warm | `cob-standalone` 76.600 ms · 4.013× (6/6) | `cob-standalone` 76.600 ms · 4.013× (6/6) | `cob-standalone` 76.600 ms · 4.013× (6/6) | `cob-standalone` 76.600 ms · 4.013× (6/6) |
| First A + ½ | first | `cob-standalone` 77.448 ms · 0.944× (0/6) | `cob-standalone` 77.448 ms · 0.944× (0/6) | `cob-standalone` 77.448 ms · 0.944× (0/6) | `cob-standalone` 77.448 ms · 0.944× (0/6) |
| First A + ½ | warm | `cob-standalone` 77.343 ms · 0.968× (0/6) | `cob-standalone` 77.343 ms · 0.968× (0/6) | `cob-standalone` 77.343 ms · 0.968× (0/6) | `cob-standalone` 77.343 ms · 0.968× (0/6) |
| Last B + ½ | first | `cob-standalone` 77.595 ms · 0.894× (0/6) | `cob-standalone` 77.595 ms · 0.894× (0/6) | `cob-standalone` 77.595 ms · 0.894× (0/6) | `cob-standalone` 77.595 ms · 0.894× (0/6) |
| Last B + ½ | warm | `cob-standalone` 77.061 ms · 0.915× (0/6) | `cob-standalone` 77.061 ms · 0.915× (0/6) | `cob-standalone` 77.061 ms · 0.915× (0/6) | `cob-standalone` 77.061 ms · 0.915× (0/6) |
| UCI feature counts | first | `cob-standalone` 17.939 ms · 0.985× (0/6) | `cob-standalone` 17.939 ms · 0.985× (0/6) | `cob-standalone` 17.939 ms · 0.985× (0/6) | `cob-standalone` 17.939 ms · 0.985× (0/6) |
| UCI feature counts | warm | `cob-standalone` 17.906 ms · 1.000× (2/6) | `cob-standalone` 17.906 ms · 1.000× (2/6) | `cob-standalone` 17.906 ms · 1.000× (2/6) | `cob-standalone` 17.906 ms · 1.000× (2/6) |
| UCI binary pixels | first | `cob-standalone` 133.673 ms · 3.856× (6/6) | `cob-standalone` 133.673 ms · 3.856× (6/6) | `cob-standalone` 133.673 ms · 3.856× (6/6) | `cob-standalone` 133.673 ms · 3.856× (6/6) |
| UCI binary pixels | warm | `cob-standalone` 133.734 ms · 4.228× (6/6) | `cob-standalone` 133.734 ms · 4.228× (6/6) | `cob-standalone` 133.734 ms · 4.228× (6/6) | `cob-standalone` 133.734 ms · 4.228× (6/6) |

| Fixture | Call | Independent one-shot FP32 | Independent FP32 incl. retained | Exact integer/binary specialist | Other checked/backing policy |
| --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | `fastmmw-1536` 2,172.941 ms · 4.464× (6/6) | `fastmmw-1536` 2,172.941 ms · 4.464× (6/6) | `onednn-int8` 473.225 ms · 0.998× (3/6) | `exact-dispatch` 892.736 ms · 1.842× (6/6) |
| Signed 4096³ | warm | `fastmmw-1536` 2,152.888 ms · 4.525× (6/6) | `fastmmw-1536` 2,152.888 ms · 4.525× (6/6) | `onednn-int8` 454.511 ms · 1.004× (3/6) | `exact-dispatch` 877.531 ms · 1.876× (6/6) |
| Signed 3584³ | first | `fastmmw-768` 1,470.781 ms · 4.916× (6/6) | `fastmmw-768` 1,470.781 ms · 4.916× (6/6) | `onednn-int8` 307.789 ms · 1.018× (5/6) | `exact-dispatch` 591.308 ms · 1.982× (6/6) |
| Signed 3584³ | warm | `fastmmw-768` 1,447.279 ms · 4.807× (6/6) | `fastmmw-768` 1,447.279 ms · 4.807× (6/6) | `onednn-int8` 305.780 ms · 1.020× (5/6) | `exact-dispatch` 598.694 ms · 1.995× (6/6) |
| Signed rectangle | first | `fastmmw-1536` 79.189 ms · 3.761× (6/6) | `fastmmw-1536` 79.189 ms · 3.761× (6/6) | `onednn-int8` 21.172 ms · 1.000× (3/6) | `exact-dispatch` 35.534 ms · 1.680× (6/6) |
| Signed rectangle | warm | `openblas` 78.500 ms · 4.122× (6/6) | `openblas` 78.500 ms · 4.122× (6/6) | `onednn-int8` 19.697 ms · 1.032× (6/6) | `exact-dispatch` 33.878 ms · 1.770× (6/6) |
| First A + ½ | first | `fastmmw-1536` 79.353 ms · 0.965× (0/6) | `fastmmw-1536` 79.353 ms · 0.965× (0/6) | — | `exact-dispatch` 81.404 ms · 0.992× (1/6) |
| First A + ½ | warm | `armpl` 78.763 ms · 0.993× (3/6) | `armpl` 78.763 ms · 0.993× (3/6) | — | `fp32-policy` 79.027 ms · 0.997× (2/6) |
| Last B + ½ | first | `openblas` 79.796 ms · 0.916× (0/6) | `openblas` 79.796 ms · 0.916× (0/6) | — | `fp32-policy` 81.369 ms · 0.936× (0/6) |
| Last B + ½ | warm | `openblas` 78.993 ms · 0.938× (0/6) | `openblas` 78.993 ms · 0.938× (0/6) | — | `fp32-policy` 79.172 ms · 0.938× (0/6) |
| UCI feature counts | first | `armpl` 18.231 ms · 1.002× (4/6) | `armpl` 18.231 ms · 1.002× (4/6) | `onednn-int8` 15.537 ms · 0.855× (0/6) | `exact-dispatch` 17.992 ms · 0.987× (2/6) |
| UCI feature counts | warm | `armpl` 17.926 ms · 1.001× (4/6) | `armpl` 17.926 ms · 1.001× (4/6) | `onednn-int8` 13.156 ms · 0.734× (0/6) | `exact-dispatch` 16.042 ms · 0.895× (0/6) |
| UCI binary pixels | first | `openblas` 140.542 ms · 4.069× (6/6) | `openblas` 140.542 ms · 4.069× (6/6) | `bitset-popcount` 26.097 ms · 0.755× (0/6) | `exact-dispatch` 58.014 ms · 1.691× (6/6) |
| UCI binary pixels | warm | `openblas` 139.270 ms · 4.427× (6/6) | `openblas` 139.270 ms · 4.427× (6/6) | `bitset-popcount` 25.574 ms · 0.813× (0/6) | `exact-dispatch` 55.141 ms · 1.756× (6/6) |

### Complete field and fixed controls

The full-field winner column includes checked S8 itself. The next column compares it with the fastest eligible rival. Each fixed-control cell gives the peer’s milliseconds, paired ratio and strict wins for checked S8. The binary specialist is eligible on the pixel fixture.

| Fixture | Call | Whole-field winner, ms | Fastest other eligible peer | Backing FP32 | Standalone COB | Unchecked BF16 | oneDNN S8 specialist | Binary AND/popcount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Signed 4096³ | first | `onednn-int8` 473.225 | `onednn-int8` 473.225 ms · 0.998× (3/6) | 2,178.305 · 4.487× (6/6) | 2,211.560 · 4.566× (6/6) | 861.564 · 1.802× (6/6) | 473.225 · 0.998× (3/6) | — |
| Signed 4096³ | warm | `onednn-int8` 454.511 | `onednn-int8` 454.511 ms · 1.004× (3/6) | 2,149.978 · 4.524× (6/6) | 2,213.384 · 4.650× (6/6) | 855.923 · 1.816× (6/6) | 454.511 · 1.004× (3/6) | — |
| Signed 3584³ | first | `s8-dispatch` 299.822 | `onednn-int8` 307.789 ms · 1.018× (5/6) | 1,825.190 · 6.059× (6/6) | 1,494.465 · 4.971× (6/6) | 577.221 · 1.918× (6/6) | 307.789 · 1.018× (5/6) | — |
| Signed 3584³ | warm | `s8-dispatch` 300.245 | `onednn-int8` 305.780 ms · 1.020× (5/6) | 1,818.760 · 6.074× (6/6) | 1,488.018 · 4.948× (6/6) | 583.973 · 1.943× (6/6) | 305.780 · 1.020× (5/6) | — |
| Signed rectangle | first | `s8-dispatch` 21.130 | `onednn-int8` 21.172 ms · 1.000× (3/6) | 80.805 · 3.821× (6/6) | 77.014 · 3.641× (6/6) | 33.711 · 1.590× (6/6) | 21.172 · 1.000× (3/6) | — |
| Signed rectangle | warm | `s8-dispatch` 19.043 | `onednn-int8` 19.697 ms · 1.032× (6/6) | 78.595 · 4.127× (6/6) | 76.600 · 4.013× (6/6) | 31.722 · 1.660× (6/6) | 19.697 · 1.032× (6/6) | — |
| First A + ½ | first | `raw-bf16` 34.446 | `raw-bf16` 34.446 ms · 0.421× (0/6) | 81.503 · 0.993× (2/6) | 77.448 · 0.944× (0/6) | 34.446 · 0.421× (0/6) | — | — |
| First A + ½ | warm | `raw-bf16` 32.331 | `raw-bf16` 32.331 ms · 0.408× (0/6) | 79.027 · 0.997× (2/6) | 77.343 · 0.968× (0/6) | 32.331 · 0.408× (0/6) | — | — |
| Last B + ½ | first | `raw-bf16` 34.481 | `raw-bf16` 34.481 ms · 0.396× (0/6) | 81.369 · 0.936× (0/6) | 77.595 · 0.894× (0/6) | 34.481 · 0.396× (0/6) | — | — |
| Last B + ½ | warm | `raw-bf16` 32.405 | `raw-bf16` 32.405 ms · 0.384× (0/6) | 79.172 · 0.938× (0/6) | 77.061 · 0.915× (0/6) | 32.405 · 0.384× (0/6) | — | — |
| UCI feature counts | first | `onednn-int8` 15.537 | `onednn-int8` 15.537 ms · 0.855× (0/6) | 18.204 · 0.999× (3/6) | 17.939 · 0.985× (0/6) | 18.005 · 0.988× (0/6) | 15.537 · 0.855× (0/6) | — |
| UCI feature counts | warm | `onednn-int8` 13.156 | `onednn-int8` 13.156 ms · 0.734× (0/6) | 17.926 · 0.999× (3/6) | 17.906 · 1.000× (2/6) | 15.999 · 0.891× (0/6) | 13.156 · 0.734× (0/6) | — |
| UCI binary pixels | first | `bitset-popcount` 26.097 | `bitset-popcount` 26.097 ms · 0.755× (0/6) | 144.199 · 4.154× (6/6) | 133.673 · 3.856× (6/6) | 55.568 · 1.630× (6/6) | 33.751 · 0.981× (1/6) | 26.097 · 0.755× (0/6) |
| UCI binary pixels | warm | `bitset-popcount` 25.574 | `bitset-popcount` 25.574 ms · 0.813× (0/6) | 142.365 · 4.480× (6/6) | 133.734 · 4.228× (6/6) | 53.212 · 1.671× (6/6) | 31.762 · 1.011× (4/6) | 25.574 · 0.813× (0/6) |

### Runtime routes and numerical checks

The independent signed preflight tests the backend on the geometry. The public candidate makes its own decision on each call. The table lists observed branch reasons, the number of logical input elements inspected per call, and the signed native result expected by the frozen scorer. A checked S8 fallback is compared bit-for-bit with the FP32 backing call.

| Fixture | M × N × K | Backing FP32 | Signed capability | Checked probe | Expected native product | Primary outcome | Observed public call | Inexact / checked outputs | Max absolute error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Signed 4096³ | 4096 × 4096 × 4096 | `fastmmw-1536` | yes | passed | yes | verified-native | native-exact (33,554,432 inspected, 12 receipts) · `jit:int8` | 0 / 201,326,592 | 0 |
| Signed 3584³ | 3584 × 3584 × 3584 | `armpl` | yes | passed | yes | verified-native | native-exact (25,690,112 inspected, 12 receipts) · `jit:int8` | 0 / 154,140,672 | 0 |
| Signed rectangle | 1079 × 1553 × 1201 | `armpl` | yes | passed | yes | verified-native | native-exact (3,161,032 inspected, 12 receipts) · `jit:int8` | 0 / 20,108,244 | 0 |
| First A + ½ | 1079 × 1553 × 1201 | `armpl` | yes | passed | no | verified-fallback | integer-domain (64 inspected, 12 receipts) | 0 / 20,108,244 | 0 |
| Last B + ½ | 1079 × 1553 × 1201 | `armpl` | yes | passed | no | verified-fallback | integer-domain (3,161,032 inspected, 12 receipts) | 0 / 20,108,244 | 0 |
| UCI feature counts | 1797 × 3823 × 64 | `armpl` | yes | passed | no | verified-fallback | depth-shallow (0 inspected, 12 receipts) | 0 / 82,439,172 | 0 |
| UCI binary pixels | 1797 × 1934 × 1024 | `armpl` | yes | passed | yes | verified-native | native-exact (3,820,544 inspected, 12 receipts) · `jit:int8` | 0 / 41,704,776 | 0 |

The independent primitive probes report the following implementations and arithmetic instruction evidence. Instruction names identify code in the retained probe evidence. They do not count retired instructions in the timed workers.

| Independently probed geometry | Checked S8 operator | Signed specialist | oneDNN signed primitive | Raw BF16 |
| --- | --- | --- | --- | --- |
| Signed 4096³ | `jit:int8` · `smmla` (128 sites) | `jit:int8` · `smmla` (128 sites) | `jit:int8` · `smmla` | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` |
| Signed 3584³ | `jit:int8` · `smmla` (160 sites) | `jit:int8` · `smmla` (160 sites) | `jit:int8` · `smmla` | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` |
| Signed rectangle | `jit:int8` · `smmla` (1008 sites) | `jit:int8` · `smmla` (1008 sites) | `jit:int8` · `smmla` | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` |
| UCI feature counts | FP32 fallback | `jit:int8` · `smmla` (280 sites) | `jit:int8` · `smmla` | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` |
| UCI binary pixels | `jit:int8` · `smmla` (196 sites) | `jit:int8` · `smmla` (196 sites) | `jit:int8` · `smmla` | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` |

| Timed adapter | Binary SHA-256 |
| --- | --- |
| `s8-dispatch` | `b87aa6e3e6a787d1ad198ec7dc23f79b64ffa78eb4c1ee4de034f1725f32a6f8` |
| `exact-dispatch` | `be6440b79b8af82beb5d0f2c88a8712695a5a777782aa714c4cd815b5e3d0224` |
| `raw-bf16` | `8c9fb22488649465f80c26f7a88032658d2e6e9d9e9a6444106183af0baa2506` |
| `onednn-int8` | `d6cbae834cbbf294f174f84b6433d88e15893fdbad0293a511fc4440d06288fb` |
| `bitset-popcount` | `b126e107dd88e54df0e62cd355b1768e4246c723a500f3ffc0e9945d4397d029` |
| `fp32-policy` | `8dccccd2afa7062dcc719cf3a2330702994888c8a6dbb1c0eed9cae70b4dbc90` |

The audit checked 6,978,950,784 logical output values and 18,241,920 padding or guard locations in 1,068 timed result checks. It recorded 36 bit-identical S8-policy fallback comparisons.

### Qualification by method

| Method | Outcome | Fixtures |
| --- | --- | --- |
| `bitset-popcount` | outside registered binary image workload | Signed 4096³, Signed 3584³, Signed rectangle, First A + ½, Last B + ½, UCI feature counts |
| `onednn-int8` | fractional input outside INT8 domain | First A + ½, Last B + ½ |
| `blasfeo` | worker exit -11: BLASFEO target=ARMV8A_ARM_CORTEX_A76 API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=1 | UCI binary pixels |

Scored observations (`aarch64/raw/score.json` in [evidence.zip](evidence.zip)) · Field comparisons (`aarch64/raw/comparisons.csv` in [evidence.zip](evidence.zip)) · Admission receipts (`aarch64/raw/admissions.json` in [evidence.zip](evidence.zip))

## Local correctness supplement

Separate Ubuntu 24.04 Docker checks on an Apple host used the frozen public entry and header. The Neoverse-N2 tests under QEMU produced exactly 2²⁴ across 513 outputs. For a true product of 2²⁴ + 1 the certificate declined and ordinary FP32 rounded each of 513 outputs to 2²⁴. Injected failures during execution and after the wait left the caller’s output safe for fallback, and the subsequent retry succeeded. Arm and x86 transfer tests also exercised an unavailable backend. These are untimed correctness checks. Test receipts (`supplementary/verified.json` in [evidence.zip](evidence.zip)) · [C API](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/6a9c6a696903926b83caf19364b4d19b23f8f2f6/tools/world-rank/exact/s8/checked-int8.h) · [Build and use](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/6a9c6a696903926b83caf19364b4d19b23f8f2f6/tools/world-rank/exact/s8/protocol.md)

## Reproduction

Run `python3 -B verify.py` from the [Linux directory](../README.md). The [study archive](evidence.zip) contains frozen source, raw observations, qualification and comparison records. [All six paired ratios by peer](peer-comparisons.csv)

x86_64 · [Job 139815068937](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553/job/139815068937) · artifact 15104162186, SHA-256 `f4044fb5a5294ff6b58bd2df6882ae65f1c5eb3996b530819e28e1e71fb3a9b7`.

aarch64 · [Job 139815068579](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553/job/139815068579) · artifact 15103644670, SHA-256 `d03a308d69770c7dd30b01b4c6d7c5a5e7c29b0049d5f79b3be4a630176709ab`.

[Frozen benchmark and protocol](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/6a9c6a696903926b83caf19364b4d19b23f8f2f6/tools/world-rank/exact/s8) · [Timed COB release](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/e4b7ce19cdf49f9175c86d3e91ea8d3a756eae9b)
