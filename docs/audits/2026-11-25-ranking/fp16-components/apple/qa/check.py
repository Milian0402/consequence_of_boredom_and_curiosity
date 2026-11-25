"""Tiny un-timed whole-output checks of the public half→float matrix multiply (Darwin)."""
import argparse
import ctypes as C
import hashlib
import json
import math
import mmap
import random
import struct
from pathlib import Path

PAGESZ=mmap.PAGESIZE
F32=C.c_float
FP=C.POINTER(F32)
POISON=0x7FC12345
CANARY=0xC58A17C3
def f32(x):return struct.unpack('=f',struct.pack('=f',x))[0]
def bits(x):return struct.unpack('=I',struct.pack('=f',x))[0]
def require(x,m):
    if not x:raise ValueError(m)
class Parent:
    def __init__(self,rows,cols,offset_row,offset_col,stride,word):
        self.rows,self.cols,self.ld,self.offset=rows,cols,stride,offset_row*stride+offset_col
        self.pages=math.ceil(max((rows+offset_row+2)*stride*4,4)/PAGESZ)
        self.map=mmap.mmap(-1,(self.pages+2)*PAGESZ,prot=mmap.PROT_READ|mmap.PROT_WRITE)
        self.start=C.addressof(C.c_char.from_buffer(self.map))+PAGESZ
        self.words=(C.c_uint32*(self.pages*PAGESZ//4)).from_address(self.start)
        self.floats=(F32*len(self.words)).from_address(self.start)
        self.ptr=C.cast(self.start+4*self.offset,FP)
        self.protect(self.start-PAGESZ,PAGESZ,0);self.protect(self.start+self.pages*PAGESZ,PAGESZ,0)
        for i in range(len(self.words)):self.words[i]=word
    @staticmethod
    def protect(addr,size,mode):
        require(C.CDLL(None).mprotect(C.c_void_p(addr),C.c_size_t(size),C.c_int(mode))==0,'mprotect failed')
    def readonly(self):self.protect(self.start,self.pages*PAGESZ,mmap.PROT_READ)
    def mask(self,i,j):return self.offset+i*self.ld+j
    def digest(self):return hashlib.sha256(C.string_at(self.start,self.pages*PAGESZ)).digest()
    def close(self):self.map.close()

def inputs(m,n,k,kind,seed):
    r=random.Random(seed)
    def rand():return f32(r.uniform(-1,1))
    if kind=='dyadic':
        a=[[f32(((3*i+5*p)%11-5)/8) for p in range(k)] for i in range(m)]
        b=[[f32(((2*p+7*j)%9-4)/4) for j in range(n)] for p in range(k)]
    elif kind=='cancel':
        a=[[f32(r.uniform(.5,1)) for _ in range((k+1)//2)] for i in range(m)]
        a=[[f32(row[p//2]*(1+(2**-16 if p%2 else 0))) for p in range(k)] for row in a]
        base=[[f32(r.uniform(.5,1)) for _ in range(n)] for p in range((k+1)//2)]
        b=[[(-1 if p%2 else 1)*base[p//2][j] for j in range(n)] for p in range(k)]
    else:
        a=[[rand() for p in range(k)] for i in range(m)];b=[[rand() for j in range(n)] for p in range(k)]
        if kind=='scaled':
            a=[[f32(x*2.0**((p%5)-2)) for p,x in enumerate(row)] for row in a]
            b=[[f32(x*2.0**(2-(p%5))) for x in row] for p,row in enumerate(b)]
        if kind=='axis-scaled':
            a=[[f32(x*2.0**((i%5-2)*24)) for x in row] for i,row in enumerate(a)]
            b=[[f32(x*2.0**((j%5-2)*24)) for j,x in enumerate(row)] for row in b]
    return a,b

def case(fn,m,n,k,kind,seed,receipt,extreme=None):
    a,b=inputs(m,n,k,kind,seed)
    if extreme is not None:a[0][0]=extreme
    pa=Parent(m,k,1,3,k+8,POISON);pb=Parent(k,n,2,1,n+7,POISON);pc=Parent(m,n,3,5,n+11,CANARY)
    outside=set(range(len(pc.words)))
    for i in range(m):
        for t in range(k):pa.floats[pa.mask(i,t)]=a[i][t]
    for t in range(k):
        for j in range(n):pb.floats[pb.mask(t,j)]=b[t][j]
    for i in range(m):
        for j in range(n):outside.remove(pc.mask(i,j))
    pa.readonly();pb.readonly();unchanged=pa.digest(),pb.digest()
    ref=[math.fsum(a[i][p]*b[p][j] for p in range(k)) for i in range(m) for j in range(n)] if extreme is None else []
    weight=[math.fsum(abs(a[i][p]*b[p][j]) for p in range(k)) for i in range(m) for j in range(n)] if extreme is None else []
    for mode in (0,1):
        for i in range(m):
            for j in range(n):pc.words[pc.mask(i,j)]=POISON
        pre=pc.digest();rc=fn(mode,m,n,k,pa.ptr,pa.ld,pb.ptr,pb.ld,pc.ptr,pc.ld)
        require(unchanged==(pa.digest(),pb.digest()),'protected input changed')
        require(all(pc.words[t]==CANARY for t in outside),f'C parent clobbered {m,n,k,mode}')
        if extreme is not None:
            require(rc==0 and pc.digest()==pre,f'range reject modified C or returned accepted {extreme,rc,mode}')
            receipt['range_rejections']+=1
            continue
        require(rc==1,f'ordinary test unexpectedly declined {kind,m,n,k,mode}')
        gamma=k/(2**24-k)
        errors=[];rho=[]
        for idx,(truth,magnitude) in enumerate(zip(ref,weight)):
            i,j=divmod(idx,n);got=pc.floats[pc.mask(i,j)]
            require(math.isfinite(got),f'nonfinite output {m,n,k,kind,mode,i,j}')
            e=abs(got-truth);r=e/magnitude if magnitude else (0 if e==0 else math.inf)
            if kind=='dyadic':require(got==truth,f'exact dyadic product {mode,m,n,k,i,j}')
            # RNE conversions of two leading normal halves introduce up to ~2u16 in raw dot.
            limit=0.0011+3*gamma if mode==0 else 1.5e-6+3*gamma
            require(r<=limit,f'componentwise discrepancy {m,n,k,kind,mode,i,j,r,limit}')
            errors.append(e);rho.append(r)
        stats=receipt['modes'][mode];stats['calls']+=1;stats['cells']+=len(errors);stats['guards']+=len(outside)
        stats['max_abs']=max(stats['max_abs'],max(errors,default=0));stats['max_componentwise']=max(stats['max_componentwise'],max(rho,default=0))
        stats['max_abs_by_fixture'][kind]=max(stats['max_abs_by_fixture'].get(kind,0),max(errors,default=0))
    pa.close();pb.close();pc.close()

def invalid_geometry(fn):
    a=Parent(2,4,1,2,9,CANARY);b=Parent(4,3,2,1,8,CANARY);c=Parent(2,3,3,5,11,CANARY)
    a.readonly();b.readonly();before=c.digest();checked=0
    for mode in (0,1):
        for m,n,k,la,lb,lc in [(-1,3,4,9,8,11),(2,-1,4,9,8,11),(2,3,-1,9,8,11),
                                (2,3,4,3,8,11),(2,3,4,9,2,11),(2,3,4,9,8,2)]:
            rc=fn(mode,m,n,k,a.ptr,la,b.ptr,lb,c.ptr,lc)
            require(rc==0 and c.digest()==before,'invalid dimensions or stride modified output')
            checked+=1
        for m,n in ((0,3),(2,0)):
            rc=fn(mode,m,n,4,a.ptr,9,b.ptr,8,c.ptr,11)
            require(rc in (0,1) and c.digest()==before,'empty output modified parent')
            checked+=1
    for mode in (0,1):
        require(fn(mode,2,3,4,a.ptr,9,b.ptr,8,a.ptr,9)==0 and c.digest()==before,'output/input alias accepted')
        require(fn(mode,2,3,4,a.ptr,9,b.ptr,8,b.ptr,8)==0 and c.digest()==before,'output/right-input alias accepted')
        require(fn(mode,2,3,0,None,0,None,0,c.ptr,11)==1,'zero-depth overwrite declined')
        logical={c.mask(i,j) for i in range(2) for j in range(3)}
        require(all(w==(0 if at in logical else CANARY) for at,w in enumerate(c.words)),'incorrect zero overwrite')
        for i in logical:c.words[i]=CANARY
        checked+=3
    a.close();b.close();c.close()
    return checked

def overflow_rejection(fn):
    a=Parent(2,4,1,1,6,POISON);b=Parent(4,3,1,2,7,POISON);c=Parent(2,3,1,1,5,CANARY)
    for i in range(2):
        for p in range(4):a.floats[a.mask(i,p)]=2.0**80
    for p in range(4):
        for j,scale in enumerate((2.0**-80,2.0**80,0.0)):b.floats[b.mask(p,j)]=scale
    a.readonly();b.readonly();before=c.digest()
    for mode in (0,1):
        require(fn(mode,2,3,4,a.ptr,a.ld,b.ptr,b.ld,c.ptr,c.ld)==0 and c.digest()==before,
                'transactional output overflow: preceding representable column written')
    a.close();b.close();c.close()
    return 2

def run(lib,reject):
    d=C.CDLL(str(lib.resolve()));fn=d.cob_half_sgemm
    fn.argtypes=[C.c_int]*4+[FP,C.c_int,FP,C.c_int,FP,C.c_int];fn.restype=C.c_int
    d.cob_half_svl.argtypes=[];d.cob_half_svl.restype=C.c_int;vl=d.cob_half_svl()
    require(1<=vl<=64,'unexpected FP32 streaming vector length')
    receipt=dict(lanes=vl,modes=[dict(calls=0,cells=0,guards=0,max_abs=0,max_componentwise=0,max_abs_by_fixture={}) for _ in (0,1)],range_rejections=0)
    edges=sorted({1,max(1,vl-1),vl,vl+1,2*vl-1,2*vl+1})
    cases=[(m,n,(1,2,3,5,7,17)[(i+2*j)%6],'dyadic') for i,m in enumerate(edges) for j,n in enumerate(edges)]
    cases.extend([(3,5,257,'random'),(vl+1,vl+3,129,'cancel'),(vl+3,7,83,'scaled'),(2*vl+1,vl+1,37,'random'),(5,9,97,'cancel'),(3,7,23,'scaled'),(11,9,37,'axis-scaled')])
    for idx,(m,n,k,kind) in enumerate(cases):case(fn,m,n,k,kind,0x5DC7+idx,receipt)
    for value in reject:case(fn,3,5,7,'dyadic',123,receipt,value)
    receipt['invalid_or_empty_shapes']=invalid_geometry(fn)
    receipt['output_overflow_rejections']=overflow_rejection(fn)
    receipt['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    return receipt
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dylib',type=Path);parser.add_argument('--reject',type=float,nargs='*',default=[float('inf'),float('nan'),2.0**75,2.0**-120]);parser.add_argument('--output',type=Path);arg=parser.parse_args()
    verdict=run(arg.dylib,arg.reject);msg=json.dumps(verdict,indent=2,allow_nan=False)+'\n'
    print(msg,end='')
    if arg.output:arg.output.write_text(msg)
