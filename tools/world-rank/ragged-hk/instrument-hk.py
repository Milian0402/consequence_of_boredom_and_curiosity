"""Add forwarding-only witnesses to a disposable diagnostic copy of COB."""
from pathlib import Path
import sys

src=Path(sys.argv[1])/'src'
def edit(name,old,new):
    path=src/name;txt=path.read_text()
    if txt.count(old)!=1: raise ValueError('unknown source anchor: '+name)
    path.write_text(txt.replace(old,new))
args='int,int,int,const float*,int,const float*,int,float*,int'
insert=f"""
typedef void (*hk_trace_fn)({args});
extern void hk_trace_bind(int,int,int,float*,float*,float*,hk_trace_fn,int);
extern void hk_trace_leaf({args});
extern void hk_trace_fused(void);
"""
edit('gemm.c','#include <stdint.h>','#include <stdint.h>\n'+insert)
mark='#include "../third_party/fast-matmul/hk233.inc"'
edit('cob_hk15.inc',mark,'''    int hk_lowered = 0;
#if defined(COB_USE_FUSED_STRASSEN)
    hk_lowered = cob_sgemm_rowmajor == cob_hk15_fused_child;
#endif
    hk_trace_bind(mh,nh,kh,x,y,product,cob_sgemm_rowmajor,hk_lowered);
    cob_sgemm_rowmajor = hk_trace_leaf;
'''+mark)
mark='    cob_fused_finish_edges(m, n, k, a, lda, b, ldb, c, ldc);\n    return 1;'
edit('cob_fused_strassen.inc',mark,mark.replace('    return 1;','    hk_trace_fused();\n    return 1;'))
