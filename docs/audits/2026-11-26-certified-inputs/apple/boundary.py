"""Portable exhaustive bit-boundary test, with optional M5 protected-ABI comparisons."""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import random
import struct

def decode(u):
    mag=u&0x7fffffff
    if not mag:return 0
    e=(mag>>23)-127
    if e<0 or e>5:return None
    fraction=mag&0x7fffff
    if fraction&((1<<(23-e))-1):return None
    q=((1<<23)|fraction)>>(23-e)
    return (-q if u>>31 else q) if q<=63 else None

def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def real(u):return struct.unpack('<f',struct.pack('<I',u))[0]
def oracle(u):
    x=real(u)
    return int(x) if math.isfinite(x) and -63<=x<=63 and x.is_integer() else None

def portable():
    words={0,1,0x7fffff,0x800000,0x7f800000,0x7fc10000,0x7fffffff}
    for q in range(129):
        w=bits(float(q));words.update(max(0,w+d) for d in (-2,-1,0,1,2))
    for e in range(256):
        for lo in (0,1,0x300011,0x7ffffe,0x7fffff):words.add((e<<23)|lo)
    words |= {w|0x80000000 for w in words}
    for w in words:assert decode(w)==oracle(w),hex(w)
    assert 4096*63*63==16257024<2**24
    return len(words)

def native(path):
    L=C.CDLL(str(path.resolve()));I=C.c_int;P=C.POINTER(C.c_float)
    fn=L.apple_exact;direct=L.exact_baseline
    for f in (fn,direct):f.argtypes=[I,I,I,P,I,P,I,P,I]
    direct.restype=None;fn.restype=I
    assert L.apple_int8_lanes()==16 and L.exact_blas_serial()==1
    rng=random.Random(92460837);count=0;entries=0;fallback=0
    def ptr(a,offset):return C.cast(C.byref(a,4*offset),P)
    for m,n,k in [(1,1,1),(3,5,7),(15,17,4),(31,32,13),(32,33,63),(33,31,65),(37,61,129),(2,3,4096),(2,3,4097)]:
        la,lb,lc=k+3,n+5,n+7;oa,ob,oc=la+1,lb+2,lc+3
        a=(C.c_float*((m+3)*la))();b=(C.c_float*((k+3)*lb))()
        c=(C.c_float*((m+3)*lc))();ref=type(c)()
        for x in (a,b): C.memset(C.addressof(x),0xff,C.sizeof(x))
        for i in range(m):
            for p in range(k):a[oa+i*la+p]=63 if k>=4096 else rng.randint(-63,63)
        for p in range(k):
            for j in range(n):b[ob+p*lb+j]=(-63 if j%2 else 63) if k>=4096 else rng.randint(-63,63)
        if k<4096:a[oa]=-0.
        for side,word in [('valid',None),('right',bits(63.)+1),('left',bits(-64.)),('right',1),
                          ('left',0x7fc01003),('right',0x7f800000)]:
            v=a if side=='left' else b
            at=oa+(m-1)*la+k-1 if side=='left' else ob+(k-1)*lb+n-1
            old=bytes(v)[at*4:at*4+4]
            if word is not None:C.memmove(C.addressof(v)+at*4,struct.pack('=I',word),4)
            for x in (c,ref):
                for ix in range(len(x)):x[ix]=713.25
            oldin=bytes(a),bytes(b)
            aa,bb,cc,rr=ptr(a,oa),ptr(b,ob),ptr(c,oc),ptr(ref,oc)
            rc=fn(m,n,k,aa,la,bb,lb,cc,lc)
            assert rc==int(side=='valid' and k<=4096),(m,n,k,side,rc)
            direct(m,n,k,aa,la,bb,lb,rr,lc)
            for i in range(m):
                for j in range(n):
                    z=cc[i*lc+j];r=rr[i*lc+j]
                    if rc:
                        expect=sum(int(aa[i*la+p])*int(bb[p*lb+j]) for p in range(k))
                        assert z==expect and r==expect,(m,n,k,i,j,z,expect);entries+=1
                    else:assert z==r or (math.isnan(z) and math.isnan(r))
            assert bytes(a)==oldin[0] and bytes(b)==oldin[1]
            logical={oc+i*lc+j for i in range(m) for j in range(n)}
            assert all(c[ix]==713.25 for ix in range(len(c)) if ix not in logical)
            count+=1;fallback+=rc==0
            C.memmove(C.addressof(v)+at*4,old,4)
    n=16;buf=(C.c_float*n)(*range(n));before=bytes(buf)
    assert fn(2,2,2,buf,2,buf,2,buf,2)==-1 and bytes(buf)==before
    assert fn(2,2,2,buf,1,buf,2,buf,2)==-1 and bytes(buf)==before
    assert fn(2,2,0,None,0,None,0,buf,3)==0 and [buf[0],buf[1],buf[3],buf[4]]==[0]*4
    return {'cases':count,'exact_integer_cells':entries,'fallback_cases':fallback,'alias_stride_zero_checks':3}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path)
    args=ap.parse_args();out={'portable_checked_words':portable()}
    if args.library:out.update(native(args.library))
    print(json.dumps(out,indent=2))
