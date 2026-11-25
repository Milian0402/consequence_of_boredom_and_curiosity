"""Verify the production predicate and all 18 remainders using the actual public path."""
import ctypes as ct
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as td:
    tmp=Path(td)
    src=repo/'src'
    schedule=repo/'third_party/fast-matmul/hk233.inc'
    hk=(src/'cob_hk15.inc').read_text()
    geometry=hk[hk.index('static int cob_hk15_shape'):hk.index('static int cob_hk15_try')]
    exported=geometry.replace('static int cob_hk15_shape','int full_geometry')
    hk=hk.replace(geometry,geometry.replace('768','8').replace('1280','32'))
    hk=hk.replace('#include "../third_party/fast-matmul/hk233.inc"',f'#include "{schedule}"')
    needle='= cob_sgemm_rowmajor_base;'
    assert hk.count(needle)==1
    hk=hk.replace(needle,'= counted_leaf;')
    alloc='cob_aligned_alloc(128, (ac+bc+pc)*sizeof(float))'
    assert hk.count(alloc)==1
    hk=hk.replace(alloc,'(++attempts, refuse ? NULL : '+alloc+')')
    code=(src/'gemm.c').read_text()
    spec='int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc'
    values='m,n,k,a,lda,b,ldb,c,ldc'
    before=f'''int leaf_count, parent_count, attempts, refuse;
static void counted_leaf({spec}) {{ ++leaf_count; cob_sgemm_rowmajor_base({values}); }}
'''
    code=code.replace('#include "cob_hk15.inc"',exported+before+hk)
    target='    if (cob_hk15_try(m, n, k, a, lda, b, ldb, c, ldc)) return;'
    assert code.count(target)==1
    code=code.replace(target,'    ++parent_count;\n'+target)
    code+=f'\nint attempt_hk({spec}) {{ return cob_hk15_try({values}); }}\n'
    (tmp/'check.c').write_text(code)
    cc=os.environ.get('CC','cc')
    flags=['-O3','-std=c11','-fPIC','-DCOB_ENABLE_HK15=1','-DCOB_ENABLE_HK15_RAGGED=1','-DCOB_DISABLE_APPLE_AMX=1','-DCOB_DISABLE_APPLE_SME=1']
    if platform.machine().lower() in ('arm64','aarch64'):flags+=['-DCOB_USE_NEON=1']
    lib=tmp/'check.so';subprocess.run([cc,*flags,'-I'+str(src),'-I'+str(repo/'include'),
        '-dynamiclib' if sys.platform=='darwin' else '-shared',str(tmp/'check.c'),str(src/'cob_hk15_ragged.c'),'-lm','-o',lib],check=True)
    dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float);I=ct.c_int
    abi=[I,I,I,P,I,P,I,P,I]
    dll.cob_sgemm_rowmajor.argtypes=dll.attempt_hk.argtypes=abi
    dll.attempt_hk.restype=I;dll.cob_sgemm_rowmajor.restype=None
    dll.full_geometry.argtypes=[I,I,I];dll.full_geometry.restype=I
    leaves,entries,attempts,refuse=[I.in_dll(dll,key) for key in ('leaf_count','parent_count','attempts','refuse')]
    extreme=[((1536,2304,2304),1),((1537,2306,2306),1),((2561,3842,3842),1),
             ((1535,2304,2304),0),((2562,3842,3842),0),((2561,3843,3842),0),
             ((2561,3842,3843),0),((1921,2306,2306),1),((1922,2306,2306),0),
             ((2**31-1,2**31-1,2**31-1),0),((-2**31,2304,2304),0)]
    for dims,expected in extreme: assert dll.full_geometry(*dims)==expected,(dims,expected)
    def pointer(array,offset):return ct.cast(ct.byref(array,offset*4),P)
    def buf(count):return (ct.c_float*count)(*([551.75]*count))
    def snapshot(array):return bytes(array)
    rng=random.Random(29384);worst=0;cells=0;calls=0
    for dr in range(2):
        for dc in range(3):
            for dd in range(3):
                m,n,k=18+dr,27+dc,27+dd
                la,lb,lc=k+13,n+17,n+21
                a,b,c=buf((m+3)*la),buf((k+4)*lb),buf((m+5)*lc)
                a0,b0,c0=la+3,2*lb+5,3*lc+7
                pa,pb,pc=pointer(a,a0),pointer(b,b0),pointer(c,c0)
                for mode in ('random','scaled','cancel','integer','allocation-fail'):
                    for i in range(m):
                        for p in range(k):
                            val=rng.uniform(-1,1)
                            if mode=='integer': val=(3*i+7*p)%9-4
                            if mode=='cancel': val=a[a0+i*la+p-1] if p%2 else val
                            if mode=='scaled' and p<k//2:val*=2**-8
                            a[a0+i*la+p]=val
                    for p in range(k):
                        for j in range(n):
                            val=rng.uniform(-1,1)
                            if mode=='integer':val=(5*j+11*p)%7-3
                            if mode=='cancel':val=-b[b0+(p-1)*lb+j]+1e-6 if p%2 else val
                            if mode=='scaled' and p>=k//2:val*=2**-8
                            b[b0+p*lb+j]=val
                    for i in range(m):
                        for j in range(n): c[c0+i*lc+j]=math.nan
                    original=snapshot(a),snapshot(b)
                    leaves.value=entries.value=attempts.value=0
                    refuse.value=int(mode=='allocation-fail')
                    if refuse.value:
                        untouched=bytes(c)
                        assert dll.attempt_hk(m,n,k,pa,la,pb,lb,pc,lc)==0 and bytes(c)==untouched and attempts.value==1
                        attempts.value=0
                    dll.cob_sgemm_rowmajor(m,n,k,pa,la,pb,lb,pc,lc)
                    assert (entries.value,leaves.value,attempts.value)==(1,0 if refuse.value else 15,1)
                    assert original==(snapshot(a),snapshot(b))
                    for index,val in enumerate(c):
                        off=index-c0
                        if off<0 or off//lc>=m or off%lc>=n:assert val==551.75
                        else:
                            i,j=divmod(off,lc)
                            ref=sum(a[a0+i*la+p]*b[b0+p*lb+j] for p in range(k))
                            error=abs(val-ref)
                            assert math.isfinite(error) and error<=.002 and (mode!='integer' or error==0),(dr,dc,dd,mode,error)
                            worst=max(worst,error);cells+=1
                    calls+=1
    saved=(bytes(a),bytes(b),bytes(c))
    high=ct.cast(ct.c_void_p(2**(8*ct.sizeof(ct.c_void_p))-4),P)
    cases=[(None,la,pb,lb,pc,lc),(pa,la,None,lb,pc,lc),(pa,la,pb,lb,None,lc),
        (pa,k-1,pb,lb,pc,lc),(pa,la,pb,n-1,pc,lc),(pa,la,pb,lb,pc,n-1),
        (pa,la,pb,lb,pointer(a,a0+(m-1)*la+k-1),lc),
        (pa,la,pb,lb,pointer(b,b0+(k-1)*lb+n-1),lc),
        (high,la,pb,lb,pc,lc),(pa,la,high,lb,pc,lc),(pa,la,pb,lb,high,lc)]
    refuse.value=0
    for bad in cases:
        attempts.value=leaves.value=0
        assert dll.attempt_hk(m,n,k,*bad)==0 and not attempts.value and not leaves.value
    assert saved==(bytes(a),bytes(b),bytes(c))
    print(f'PASS 11 actual size boundaries, 18 residue classes x 5 fixtures, {calls} one-shot calls, {cells} FP64 outputs, max_abs_error={worst:.9g}; exact integer outputs, 11 null/stride/tail-overlap/address-overflow guards; scratch failure before write with full public retry.')
