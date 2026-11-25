"""Public-path tests for HK15 arithmetic selection at the value-range boundaries."""
import ctypes as ct
import math
import struct


def regular_inputs(a, b):
    """Independent definition for the bounded finite initial fixtures."""
    for matrix in (a,b):
        line_maxima = [max(map(abs,row)) for row in matrix]
        line_maxima.extend(max(map(abs,column)) for column in zip(*matrix))
        if min(line_maxima)==0 or max(line_maxima)>4*min(line_maxima):
            return False
    required=(3*len(b)+3)//4
    for lines in (a,list(zip(*b))):
        threshold=max(max(map(abs,line)) for line in lines)/8
        if any(sum(abs(x)>=threshold for x in line)<required for line in lines):
            return False
    return True


def check_guard_cases(lib):
    f32=ct.c_float
    pointer=ct.POINTER(f32)
    def view(buffer,offset):
        return ct.cast(ct.byref(buffer,offset*4),pointer)
    names=('parent_count','leaf_count','attempts','classical','wide','scans','normalized','normal_allocs','refuse')
    counts={name:ct.c_int.in_dll(lib,name) for name in names}
    m,n=19,29
    sentinel=711.75
    def make(rows,ld):
        return (f32*((rows+6)*ld))(*([sentinel]*((rows+6)*ld)))
    bits=lambda x: struct.unpack('<I',struct.pack('<f',x))[0]
    value=lambda v: struct.unpack('<f',struct.pack('<I',v))[0]
    cases={
        'balanced':'regular','exceptional-padding':'regular','interior-zeros':'classical',
        'isolated-A-row':'classical','isolated-B-column':'classical',
        'isolated-A-column':'classical','isolated-B-row':'classical','reciprocal-depth':'classical',
        'zero-A-row':'classical','zero-B-column':'classical','zero-A-column':'classical',
        'zero-B-row':'classical','four-exact':'regular','four-next':'classical',
        'magnitude-inclusives':'regular','tiny-next':'wide','huge-next':'wide',
        'subnormal':'wide','wide-cancellation':'wide','positive-infinity':'wide',
        'negative-infinity':'wide','nan':'wide','allocation-failure':'classical',
        'sparse-A-only':'classical','sparse-B-only':'classical','dense-tiny-complement':'classical',
        'A-exact-eighth':'regular','A-below-eighth':'classical',
        'B-exact-eighth':'regular','B-below-eighth':'classical',
        'normalized-powers':'normalized','normalized-reverse':'normalized',
        'subnormal-normalized':'normalized','normalization-failure':'wide',
        'span24-A':'normalized','span25-A':'wide','span24-B':'normalized','span25-B':'wide',
        'safe-product-bound':'normalized','unsafe-product-bound':'wide',
        'exact-product-bound':'normalized','next-product-bound':'wide',
    }
    scales={
        'normalized-powers':(70,-67),'normalized-reverse':(-65,61),
        'normalization-failure':(70,-67),'subnormal-normalized':(-149,126),
        'span24-A':(56,-56),'span25-A':(56,-56),'span24-B':(-56,56),'span25-B':(-56,56),
        'safe-product-bound':(65,57),'unsafe-product-bound':(65,58),
    }
    cells=0
    for test,expected_route in cases.items():
        bound_test=test in ('exact-product-bound','next-product-bound')
        k=32 if bound_test else 28
        lda,ldb,ldc=k+7,n+13,n+11
        ao,bo,co=lda+3,2*ldb+5,3*ldc+2
        a,b,c=make(m,lda),make(k,ldb),make(m,ldc)
        if test=='exceptional-padding':
            for array in (a,b):
                for index in range(len(array)):
                    array[index]=(math.inf,-math.inf,math.nan,2.0**-149)[index%4]
        for i in range(m):
            for p in range(k):
                x=1.0 if (i+2*p)%3 else -1.0
                if test=='interior-zeros' and (i+p)%3==1:x=0.0
                if test=='isolated-A-row' and i==m-1:x*=0.125
                if test=='isolated-A-column' and p==k-1:x*=0.125
                if test=='zero-A-row' and i==m-1:x=0.0
                if test=='zero-A-column' and p==k-1:x=0.0
                if test=='reciprocal-depth' and p<k//2:x*=0.125
                if test in ('four-exact','four-next') and i==m-1:
                    x*=4.0 if test=='four-exact' else value(bits(4.0)+1)
                if test=='magnitude-inclusives':x*=2.0**-48
                if (i,p)==(m-1,k-1):
                    if test=='tiny-next':x=value(bits(2.0**-48)-1)
                    if test=='huge-next':x=value(bits(2.0**48)+1)
                    if test=='subnormal':x=2.0**-149
                    if test=='nan':x=math.nan
                    if test=='positive-infinity':x=math.inf
                    if test=='negative-infinity':x=-math.inf
                if test=='wide-cancellation' and p<2:x=2.0**127
                if test=='sparse-A-only' and (p-i)%5:x=0.0
                if test=='dense-tiny-complement' and (p+i)%4!=0:x*=2.0**-15
                if test in ('A-exact-eighth','A-below-eighth') and i==m-1:
                    if p==20:x*=0.125 if test=='A-exact-eighth' else value(bits(.125)-1)
                    if p>20:x*=2.0**-15
                if test in scales:
                    x*=2.0**scales[test][0]
                    if test in ('span24-A','span25-A') and (i,p)==(m-1,k-1):
                        x*=2.0**(-24 if test=='span24-A' else -25)
                    if test=='normalized-powers' and (i+p)%5==0:x=0.0
                if bound_test:x=2.0**60
                a[ao+i*lda+p]=x
        for p in range(k):
            for j in range(n):
                x=1.0 if (3*p+j)%5 else -1.0
                if test=='interior-zeros' and (p+j)%4==1:x=-0.0
                if test=='isolated-B-column' and j==n-1:x*=0.125
                if test=='isolated-B-row' and p==k-1:x*=0.125
                if test=='zero-B-column' and j==n-1:x=0.0
                if test=='zero-B-row' and p==k-1:x=0.0
                if test=='reciprocal-depth' and p<k//2:x*=8.0
                if test=='magnitude-inclusives':x*=2.0**48
                if test=='wide-cancellation' and p<2:x=2.0 if p==0 else -2.0
                if test in ('positive-infinity','negative-infinity') and (p,j)==(k-1,0):x=0.0
                if test=='sparse-B-only' and (p-j)%5:x=0.0
                if test=='dense-tiny-complement' and (p+j)%4!=2:x*=2.0**-15
                if test in ('B-exact-eighth','B-below-eighth') and j==n-1:
                    if p==20:x*=0.125 if test=='B-exact-eighth' else value(bits(.125)-1)
                    if p>20:x*=2.0**-15
                if test in scales:
                    x*=2.0**scales[test][1]
                    if test in ('span24-B','span25-B') and (p,j)==(k-1,n-1):
                        x*=2.0**(-24 if test=='span24-B' else -25)
                if bound_test:
                    # K*maxA*maxB equals FLT_MAX/2, then exceeds it by one RHS ULP.
                    magnitude=float.fromhex('0x1.fffffep61') if test=='exact-product-bound' else 2.0**62
                    x=0.0 if p==k-1 else math.copysign(magnitude,(-1)**p)
                b[bo+p*ldb+j]=x
        for i in range(m):
            for j in range(n):c[co+i*ldc+j]=math.nan
        preserved=bytes(a),bytes(b)
        for item in counts.values():item.value=0
        counts['refuse'].value=int(test in ('allocation-failure','normalization-failure'))
        lib.cob_sgemm_rowmajor(m,n,k,view(a,ao),lda,view(b,bo),ldb,view(c,co),ldc)
        expected=(1,15 if expected_route=='regular' else 0,int(expected_route=='regular' or test=='allocation-failure'),
                  int(expected_route=='classical'),int(expected_route=='wide'),1,
                  int(expected_route in ('normalized','wide')),int(expected_route=='normalized' or test=='normalization-failure'))
        actual=tuple(counts[name].value for name in names[:-1])
        assert actual==expected,(test,actual,expected)
        assert (bytes(a),bytes(b))==preserved,test
        for index,result in enumerate(c):
            i,j=divmod(index-co,ldc)
            if i<0 or i>=m or j>=n:
                assert result==sentinel,(test,index)
                continue
            oracle=f32(sum(a[ao+i*lda+p]*b[bo+p*ldb+j] for p in range(k))).value
            if math.isnan(oracle):
                assert math.isnan(result),(test,i,j,result)
            elif math.isinf(oracle):
                assert result==oracle,(test,i,j,result,oracle)
            else:
                assert math.isfinite(result) and abs(result-oracle)<=0.002,(test,i,j,result,oracle)
                if bound_test or (test in scales and not test.startswith('span')):
                    assert result==oracle,(test,i,j,result,oracle)
            cells+=1
    print(f'PASS {len(cases)} guard cases, {cells} FP64-referenced outputs, complete parent guards and route counters.')
