# Hopcroft–Kerr coefficient table

The 3 × 2 × 3 coefficient table is from Benson and Ballard’s fast-matmul repository at revision `132358be7a6ab57c83dd5ce70ece460bd1ec5dfa`. It is distributed under the accompanying BSD 2-Clause license. [Upstream coefficient table](https://github.com/arbenson/fast-matmul/blob/132358be7a6ab57c83dd5ce70ece460bd1ec5dfa/codegen/algorithms/hk323-15-94)

Run `python3 verify.py` in this directory to certify all 324 integer coefficients of the cyclically permuted 2 × 3 × 3 multiplication tensor and verify that its generated C expressions match `hk233.inc` exactly. The fifteen leaf calls retain the original COB one-shot dispatch. Raw coefficients and generated expressions are included so building the library requires only a C compiler.
