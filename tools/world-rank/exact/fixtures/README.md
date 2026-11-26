# Exact-input SGEMM fixtures and reference

`exact_fixture_fill` accepts FP32 logical origins and independent row strides. The positive-dimensional fixtures use a signed synthetic integer matrix pair (`exact`), one fractional first-A or last-B element (`early`, `late`), or one of two fixed UCI data products (`optdigits`, `optdigits-images`). Padding and neighboring parent elements are untouched. The synthetic A and B streams use separate SplitMix64 domains, residues modulo 127 shifted by −63, and opposite fixed endpoints in the first and last logical cells. At equal dimensions and seed each fractional case has exactly the same underlying integers as its exact partner. The UCI inputs use the published observation order and ignore the seed.

| Fixture | M × N × K | Meaning |
|---|---:|---|
| `optdigits` | 1797 × 3823 × 64 | Feature-vector similarities for writer-independent test samples versus the published training partition |
| `optdigits-images` | 1797 × 1934 × 1024 | Shared set pixels for each pair of 32×32 binary images from the writer-independent and original training partitions |

`exact_fixture_golden` resolves the real, serial OpenBLAS adapter's initialization and double GEMM symbols. It widens the logical FP32 views to contiguous FP64, computes the complete signed product and the complete product of absolute input values, and independently checks up to 32 distinct result coordinates with integer arithmetic. These include all four corners for the registered shapes. The full references are mathematically exact on the fixture domain. Multiplying a finite half-integer of magnitude at most 512 by two yields an exact integer of magnitude at most 1,024. At any positive int-sized reduction depth, the sum of absolute pair products in these integer units is at most (2³¹−1)×2²⁰, below 2⁵¹. Every intermediate signed or absolute partial sum, in any order, is exactly representable in binary64. Scaling the result by one quarter recovers the original dot.

`plugin_check.c` is an untimed integration check. It dynamically binds the real dispatch and matched ordinary plugins, protects cropped A/B parents read-only, and checks all FP32 output cells and guard cells. Tests cover the last in-domain magnitude, excessive magnitude, equality and failure of the 2²⁴ accumulator bound, zero signs, early/late fractions, NaN, infinity, and the K=4096/4097 depth boundary. Rejected outputs must match the ordinary fallback bit-for-bit. The `full` option also verifies 4096³, the 1079×1553×1201 holdout and both UCI products against whole-matrix OpenBLAS FP64 references. UCI outputs and their references cast to FP32 must reproduce independent complete known-answer hashes.

From this directory, check the stored raw bytes using `(cd data && shasum -a 256 -c SHA256SUMS)`. To regenerate from the original UCI archive, run `python3 -B rebuild_digits.py optdigits.zip regenerated`. [Data provenance, transformations and attribution](data/ATTRIBUTION.md)

The final physical Arm and translated x86-64 checks, provider digests and replay commands are in the [local preflight record](preflight/README.md).
