# M5 Max one-thread Strassen leaf comparison

On November 25, 2026, an Apple M5 Max MacBook Pro running macOS 27.2.1 measured the COB Strassen path with a strided C adaptation of MaxMulSK v5.1. Six large balanced shapes were reserved before tuning. The 4096-square control takes the existing classical path. All inputs were contiguous row-major FP32. Times cover the public one-shot multiply, including internal allocation, packing and recombination. No GPU or prepacked inputs were used.

| M × N × K | Previous COB / new | MaxMulSK / new | Accelerate / new | MpGEMM / new |
| --- | ---: | ---: | ---: | ---: |
| 5888 square | 1.092 | 1.028 | 1.171 | 1.208 |
| 6271 × 6337 × 6209 | 1.059 | 1.115 | 1.275 | 1.113 |
| 6399 square | 1.032 | 1.109 | 1.179 | 1.181 |
| 6720 × 6656 × 6592 | 1.047 | 1.035 | 1.192 | 1.211 |
| 7168 square | 1.080 | 1.061 | 1.179 | 1.223 |
| 8192 square | 1.095 | 1.060 | 1.262 | 1.248 |
| Six-shape geometric mean | **1.067** | **1.067** | **1.209** | **1.197** |
| 4096 square control | 1.005 | 0.947 | 1.093 | 1.089 |

Each cell is the median paired ratio of elapsed times. Higher than one favors new COB. Two fresh processes per comparison used different input seeds (233 and 7591). Each warmed both public entry points, then alternated which library ran first over five pairs. Library positions were reversed for the second process. The three external comparisons and the previous COB comparisons produced 280 pairs. Over the six target shapes COB won 50 of 60 pairs against previous COB, 56 against MaxMulSK, 55 against Accelerate, and 56 against MpGEMM. The geometric mean versus previous COB was 1.074 in the first block and 1.060 in the second. This is a single-machine desktop measurement.

The previous COB was `3858a00`. The selected build used Apple Clang 21.0.0. Standalone MaxMulSK used upstream `7473af2`, its default v5.1 blocking and Homebrew Clang 23.1.4. MpGEMM used `419d283` and its single-thread `row_sgemm` entry with an ABI wrapper preserving `d8`–`d15`. Accelerate used `BLAS_THREADING_SINGLE_THREADED`. The comparison driver granted all libraries 64 readable zero floats after each array for standalone implementations that can read complete vector panels at a ragged edge. COB's adapted leaf selects complete 64-column half-product panels and falls back on other widths.

During discovery, explicit reuse of all seven operands in a panel, K-blocked AMX accumulation and partial second-level recursion each lost on at least one tuning shape. The selected code keeps one Strassen level and uses an SME leaf that packs B while transposing A, handles independent input and output strides, and allocates its scratch inside the call. Tuning used 5632 square and 6144 × 6400 × 6144. The default upstream 64-row, 1024-column, 1024-inner block sizes outperformed the tested alternatives in direct repeats.

All 657 shapes passed in each of the three repository test configurations. Independent DGEMM checks covered 552,229,314 output elements over 13 calls, including six target shapes, off-route widths, stride padding and cancellation inputs. The greatest absolute error was 0.00074630, against a limit of 0.002 for those inputs. The greatest error on the six target random inputs was 0.00034329. All checked outputs were finite, and padding checks passed.
