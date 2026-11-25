"""Exercise the public HK15 entry and its fallback without a large timing run.

Pass a patched COB checkout. A temporary fixture reduces only the two size
bounds. It counts entries and selected leaves, and injects allocation failure.
The exact production shape predicate is separately tested at its true bounds.
"""
import ctypes as ct
import math
import os
from pathlib import Path
import platform
from random import Random
import subprocess
import sys
import tempfile

repo = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else next(
    path for path in Path(__file__).resolve().parents if (path/'src/gemm.c').is_file())
src = repo/'src'
P = ct.POINTER(ct.c_float)
args = [ct.c_int]*3+[P,ct.c_int,P,ct.c_int,P,ct.c_int]
with tempfile.TemporaryDirectory() as folder:
    tmp=Path(folder)
    code=(src/'gemm.c').read_text()
    hk=(src/'cob_hk15.inc').read_text()
    original_predicate=hk[hk.index('static int cob_hk15_shape'):hk.index('static int cob_hk15_try')]
    gate=original_predicate.replace('static int cob_hk15_shape','int hk_test_shape')
    schedule=next((repo/p for p in ('third_party/fast-matmul/hk233.inc','tools/world-rank/finite/hk233.inc')
                   if (repo/p).is_file()))
    hk=hk.replace(original_predicate,original_predicate.replace('768','8').replace('1280','32'))
    old_bind='= cob_sgemm_rowmajor_base;'
    assert hk.count(old_bind)==1
    hk=hk.replace(old_bind,'= hk_test_leaf;')
    alloc='cob_aligned_alloc(128, (ac+bc+pc)*sizeof(float))'
    assert hk.count(alloc)==1
    hk=hk.replace(alloc,'hk_test_noalloc ? NULL : '+alloc)
    hk=hk[:hk.index('#include "')]+f'#include "{schedule}"'+hk[hk.index('\n',hk.index('#include "')):]
    inspector='''
int hk_test_leaves, hk_test_entries, hk_test_noalloc;
static void hk_test_leaf(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    ++hk_test_leaves;
    cob_sgemm_rowmajor_base(m,n,k,a,lda,b,ldb,c,ldc);
}
'''
    code=code.replace('#include "cob_hk15.inc"',inspector+gate+hk)
    hook='    if (cob_hk15_try(m, n, k, a, lda, b, ldb, c, ldc)) return;'
    assert code.count(hook)==1
    code=code.replace(hook,'    ++hk_test_entries;\n'+hook)
    code+='''
int hk_test_try(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{ return cob_hk15_try(m,n,k,a,lda,b,ldb,c,ldc); }
void hk_test_base(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{ cob_sgemm_rowmajor_base(m,n,k,a,lda,b,ldb,c,ldc); }
'''
    cfile=tmp/'fixture.c'; cfile.write_text(code)
    out=tmp/('hk-test.dylib' if platform.system()=='Darwin' else 'hk-test.so')
    flags=['-O2','-std=c11','-fPIC','-DCOB_ENABLE_HK15=1','-DCOB_ENABLE_PORTABLE_STRASSEN=1',
           '-DCOB_ENABLE_RECURSIVE_STRASSEN=1','-DCOB_ENABLE_BLOCKED_LEAF=1',
           '-DCOB_DISABLE_APPLE_AMX=1','-DCOB_DISABLE_APPLE_SME=1']
    if platform.machine().lower() in ('arm64','aarch64'): flags+=['-DCOB_USE_NEON=1']
    linker='-dynamiclib' if platform.system()=='Darwin' else '-shared'
    subprocess.run([os.environ.get('CC','cc'),*flags,'-I'+str(repo/'include'),'-I'+str(src),
                    linker,str(cfile),'-lm','-o',str(out)],check=True)
    lib=ct.CDLL(str(out))
    lib.cob_sgemm_rowmajor.argtypes=args;lib.cob_sgemm_rowmajor.restype=None
    lib.hk_test_base.argtypes=args;lib.hk_test_base.restype=None
    lib.hk_test_try.argtypes=args
    lib.hk_test_shape.argtypes=[ct.c_int]*3
    leaves=ct.c_int.in_dll(lib,'hk_test_leaves')
    entries=ct.c_int.in_dll(lib,'hk_test_entries')
    noalloc=ct.c_int.in_dll(lib,'hk_test_noalloc')
    for shape,expected in [
        ((1536,2304,2304),1),((2560,3840,3840),1),((1920,2880,2880),1),
        ((2432,3648,3648),1),((2048,2880,3264),1),((1920,2304,2304),1),
        ((1922,2304,2304),0),((1534,2304,2304),0),((2562,3840,3840),0),
        ((1536,2303,2304),0),((1536,2304,2303),0),((1537,2304,2304),0)]:
        assert lib.hk_test_shape(*shape)==expected,(shape,expected)
    rng=Random(38421);worst=0.;checked=0
    def matrix(count): return (ct.c_float*count)(*([713.25]*count))
    for m,n,k,route in [(32,48,48,15),(44,60,72,15),(16,24,24,15),(64,96,96,15),
                        (33,48,48,0),(32,49,48,0),(32,48,49,0),(32,63,48,0),
                        (14,24,24,0),(66,96,96,0),(32,48,48,0)]:
        for cancel in (False,True):
            lda,ldb,ldc=k+1,n+3,n+5
            a,b,c=matrix(m*lda+11),matrix(k*ldb+11),matrix(m*ldc+11)
            for i in range(m):
                for p in range(k): a[i*lda+p]=a[i*lda+p-1] if cancel and p%2 else rng.uniform(-1,1)
            for p in range(k):
                for j in range(n): b[p*ldb+j]=-b[(p-1)*ldb+j] if cancel and p%2 else rng.uniform(-1,1)
            for i in range(m):
                for j in range(n): c[i*ldc+j]=float('nan')
            originals=bytes(a),bytes(b)
            leaves.value=0;entries.value=0
            noalloc.value=int((m,n,k,route)==(32,48,48,0))
            lib.cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc)
            assert (leaves.value,entries.value)==(route,1),(m,n,k,leaves.value,entries.value)
            for i in range(m):
                for j in range(n):
                    ref=sum(a[i*lda+p]*b[p*ldb+j] for p in range(k))
                    err=abs(c[i*ldc+j]-ref)
                    assert math.isfinite(err) and err<=.002,(m,n,k,i,j,err)
                    worst=max(worst,err);checked+=1
                assert list(c[i*ldc+n:(i+1)*ldc])==[713.25]*(ldc-n)
            assert list(c[-11:])==[713.25]*11 and originals==(bytes(a),bytes(b))
    noalloc.value=0
    m,n,k=32,48,48
    buffers=[matrix(48*53+11) for _ in range(3)]
    a,b,c=buffers
    for operands in [(None,49,b,51,c,53),(a,49,None,51,c,53),(a,49,b,51,None,53),
                     (a,47,b,51,c,53),(a,49,b,47,c,53),(a,49,b,51,c,47),
                     (a,49,b,51,a,53),(a,49,b,51,b,53)]:
        leaves.value=0
        assert not lib.hk_test_try(m,n,k,*operands) and leaves.value==0
    for alias in (0,1):
        a,b=matrix(48*53+11),matrix(48*53+11)
        for v in (a,b):
            for i in range(len(v)): v[i]=rng.uniform(-1,1)
        a2,b2=type(a).from_buffer_copy(a),type(b).from_buffer_copy(b)
        c,c2=(a,a2) if alias==0 else (b,b2)
        leaves.value=0;entries.value=0
        lib.cob_sgemm_rowmajor(m,n,k,a,49,b,51,c,53)
        lib.hk_test_base(m,n,k,a2,49,b2,51,c2,53)
        assert (leaves.value,entries.value)==(0,1) and bytes(c)==bytes(c2)
    print(f'12 production geometry checks; 22 public calls; {checked} FP64 cells; max error {worst:.9g}')
    print('Verified 15 counted base leaves per accepted call, one public entry, forced-allocation fallback, null/stride/alias guards and untouched padding.')
