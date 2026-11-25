"""Check real strided caller storage, independently regenerated B, and actual strict FP32 kernel entry."""
import ctypes as C
import mmap
from pathlib import Path
import sys

P=C.POINTER(C.c_float)
LIBC=C.CDLL(None,use_errno=True);LIBC.mprotect.argtypes=(C.c_void_p,C.c_size_t,C.c_int)

def view(rows,cols,ld,off,fn):
    length=off+rows*ld+17
    mem=mmap.mmap(-1,length*4); values=(C.c_float*length).from_buffer(mem)
    for i in range(length): values[i]=-947.5
    for i in range(rows):
        for j in range(cols): values[off+i*ld+j]=fn(i,j)
    return mem,values,C.cast(C.byref(values,4*off),P)

def check(directory,trace):
    f=directory/('acl-trace.so' if trace else 'acl.so'); lib=C.CDLL(str(f)); assert lib.rank_vendor_init()==0
    lib.rank_vendor_sgemm.argtypes=(C.c_int,C.c_int,C.c_int,P,C.c_int,P,C.c_int,P,C.c_int);lib.rank_vendor_sgemm.restype=None
    if trace:
        lib.rank_acl_name.restype=C.c_char_p;lib.rank_acl_count.restype=C.c_uint64
    else:
        assert not hasattr(lib,'rank_acl_num_routes')
    for m,n,k in ((17,21,13),(97,137,131)):
        lda,ldb,ldc=k+13,n+19,n+24
        for seed in (2,9):
            val_a=lambda i,p: (i*3+p*5)%13-6
            val_b=lambda p,j:(p*7+j*3+seed)%17-8
            a,av,ap=view(m,k,lda,3,val_a); b,bv,bp=view(k,n,ldb,7,val_b); c,cv,cp=view(m,n,ldc,5,lambda i,j:float('nan'))
            for mem,buf in ((a,av),(b,bv)):
                assert LIBC.mprotect(C.c_void_p(C.addressof(buf)),len(mem),mmap.PROT_READ)==0
            before=sum(lib.rank_acl_count(i) for i in range(lib.rank_acl_num_routes())) if trace else 0
            lib.rank_vendor_sgemm(m,n,k,ap,lda,bp,ldb,cp,ldc)
            if trace: assert sum(lib.rank_acl_count(i) for i in range(lib.rank_acl_num_routes()))>before
            # The supplied rows/columns repeat every 13 and 17, respectively.
            values={(i,j):sum(float(val_a(i,p))*val_b(p,j) for p in range(k))
                    for i in range(min(m,13)) for j in range(min(n,17))}
            expected={5+i*ldc+j:values[i%13,j%17] for i in range(m) for j in range(n)}
            assert all(v==expected.get(i,-947.5) for i,v in enumerate(cv)),str((m,n,k,seed))
            print('PASS',f.name,m,n,k,seed, 'full_fp64',m*n,'read_only_inputs=1 outside_C_exact=1')

if __name__=='__main__':check(Path(sys.argv[1]).resolve(),sys.argv[2]=='trace')
