# Single-level 2 × 3 × 3 Hopcroft–Kerr SGEMM

This experimental plugin splits M into two row groups, K into three inner groups and N into three column groups. For divisible shapes it calls the supplied standalone COB library 15 times, compared with 18 classical block products. Each leaf accepts its actual input and output row strides. Other shapes and unsuccessful workspace allocations call the unmodified standalone public entry.

The exact coefficient table comes from Benson and Ballard’s BSD-2-licensed repository, revision `132358be7a6ab57c83dd5ce70ece460bd1ec5dfa`. The included upstream table describes the original 3 × 2 × 3 orientation. A cyclic permutation of the three tensor factors yields 2 × 3 × 3. The integer verifier checks all 324 possible input-monomial/output coefficients, and checks that the compiled expressions match its generated output. [Coefficient table](https://github.com/arbenson/fast-matmul/blob/132358be7a6ab57c83dd5ce70ece460bd1ec5dfa/codegen/algorithms/hk323-15-94) · [Hopcroft and Kerr](https://doi.org/10.1137/0120004) · [Finite tensor duality](https://doi.org/10.1137/0202013)

The direct schedule forms 11 A and 9 B temporaries across the 15 products. The other inputs directly reference caller submatrices. It uses 17 A-block additions, 15 B-block additions and 26 C-block additions. Blocks can have different areas on rectangular inputs. The three reused scratch regions occupy exactly `4 × ((M/2)(K/3) + (K/3)(N/3) + (M/2)(N/3))` bytes. An available output tile sometimes receives its initial product directly. The timing includes all workspace allocation, forming operands and accumulating output. The 1.20× reduction in leaf count is an arithmetic-only ceiling assuming identical leaf throughput and zero overhead.

```sh
bash build.sh OUTPUT_DIRECTORY /path/to/standalone-COB.so
```

Call the plugin's `rank_vendor_sgemm` or `mtmm_sgemm_rowmajor` entry. Its dependency exports the original `cob_sgemm_rowmajor` leaf. The current thread can call `rank_hk233_last_products()` after GEMM to distinguish a rank-15 execution (15) from fallback (0). The builder includes the leaf library beside the plugin, records the exact integer certificate, and checks full logical outputs against independently accumulated FP64. Smoke tests include cancellation along K, NaN initial C, distinct row paddings, unchanged inputs and fallback. The BSD license text is supplied as `LICENSE.fast-matmul` and must accompany redistribution of the coefficient table and derived expressions.
