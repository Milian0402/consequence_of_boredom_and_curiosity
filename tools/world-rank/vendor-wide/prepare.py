"""Apply only the frozen C source change to the verified production source."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

here=Path(__file__).resolve().parent
baseline,build=map(lambda s:Path(s).resolve(),sys.argv[1:])
build.mkdir(exist_ok=True)
release=baseline/'src/gemm.c'
if hashlib.sha256(release.read_bytes()).hexdigest() != '834ff10f10e1e9c4d287099d58bab87118167d74567122894a468dc7f202a48a':
    raise SystemExit('unexpected release source')
candidate=build/'candidate'
candidate.mkdir()
for name in ('include','src'):
    shutil.copytree(baseline/name,candidate/name)
subprocess.run(['git','apply','--check','--include=src/gemm.c',here/'vendor-wide.patch'],cwd=candidate,check=True)
subprocess.run(['git','apply','--include=src/gemm.c',here/'vendor-wide.patch'],cwd=candidate,check=True)
leaf=candidate/'src/gemm.c'
if hashlib.sha256(leaf.read_bytes()).hexdigest()!='f75c696ee151c4ee85e52b25054c9395dd8292b749ccba02ba29069e2e11033c':
    raise SystemExit('unexpected patched source')
text=leaf.read_text()
trace=build/'trace.c'
# Replace the one actual CBLAS call in this public route with a counting forwarding C call.
# Captures sent m/n/k as opposed to merely inferring them from the caller's guard.
entry='static int cob_try_vendor_wide(int m'
insert='''static uint64_t wide_counts[8];
void wide_backend_counts(uint64_t *out) { memcpy(out,wide_counts,sizeof wide_counts); }
static void wide_actual_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    ++wide_counts[3]; wide_counts[4]=m; wide_counts[5]=n; wide_counts[6]=k;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,a,lda,b,ldb,0.0f,c,ldc);
}
'''
start=text.index(entry)
stop=text.index('\n    return 1;\n}',start)
body=text[start:stop]
call='''    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, main_rows, n, k,
                1.0f, a, lda, b, ldb, 0.0f, c, ldc);'''
assert body.count(call)==1
body=body.replace(call,'''    ++wide_counts[0]; wide_counts[1]=main_rows; wide_counts[2]=tail;
    wide_actual_sgemm(main_rows,n,k,a,lda,b,ldb,c,ldc);''')
trace.write_text(text[:start]+insert+body+'\n    wide_counts[7]+=(tail!=0);'+text[stop:])
