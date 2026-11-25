"""Generate an isolated vendor-leaf COB using the same public routing and seven/49-product schedule."""
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
anchor = "static int cob_strassen_leaf("
leaf_start = text.index(anchor)
leaf_end = text.index("\n}", leaf_start)
public_start = text.index("void cob_sgemm_rowmajor(\n")
public_end = text.index("\n}", public_start)
fallback = "    cob_sgemm_rowmajor_fallback(m, n, k, a, lda, b, ldb, c, ldc);"
replacement = "    rank_bridge(m, n, k, a, lda, b, ldb, c, ldc);"
if text[leaf_start:leaf_end].count(fallback) != 1 or text[public_start:public_end].count(fallback) != 1:
    raise ValueError("The pinned source has different Strassen or one-shot fallback code")
body = text[:leaf_start] + text[leaf_start:leaf_end].replace(fallback, replacement) + text[leaf_end:public_start]
# Use the selected vendor for classical public fallbacks, retaining COB's short-row
# override. Limit the replacement to the supported one-shot disjoint-input contract.
public = text[public_start:public_end].replace(fallback, """    if (m > 0 && n > 0 && k > 0 && a && b && c && lda >= k && ldb >= n && ldc >= n &&
        !cob_f32_ranges_overlap(c, (size_t)(m-1)*ldc+n, a, (size_t)(m-1)*lda+k) &&
        !cob_f32_ranges_overlap(c, (size_t)(m-1)*ldc+n, b, (size_t)(k-1)*ldb+n)) {
        rank_bridge(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }
""" + fallback)
sys.stdout.write('#include "vendor.h"\nextern rank_sgemm_fn rank_bridge;\n' + body + public + text[public_end:])
