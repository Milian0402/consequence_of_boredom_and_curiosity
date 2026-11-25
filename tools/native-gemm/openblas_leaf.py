"""Build the experimental COB translation unit that uses single-thread OpenBLAS leaves.

Usage: python3 openblas_leaf.py src/gemm.c > /tmp/hybrid.c
Compile with -Isrc -Iinclude and OpenBLAS CFLAGS, then link OpenBLAS.
The ordinary project sources and normal build flags remain independent.
"""
from pathlib import Path
import sys


def with_openblas_leaves(text):
    start = text.index("static int cob_strassen_leaf(")
    end = text.index("\n}", start)
    leaf = text[start:end]
    fallback = "    cob_sgemm_rowmajor_fallback(m, n, k, a, lda, b, ldb, c, ldc);"
    if leaf.count(fallback) != 1:
        raise ValueError("expected one portable COB Strassen leaf fallback")
    replacement = """    if (openblas_get_num_threads() == 1) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
            1.0f, a, lda, b, ldb, 0.0f, c, ldc);
        return 1;
    }
""" + fallback
    return text[:start] + "#include <cblas.h>\n" + leaf.replace(fallback, replacement) + text[end:]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 openblas_leaf.py src/gemm.c")
    sys.stdout.write(with_openblas_leaves(Path(sys.argv[1]).read_text()))
