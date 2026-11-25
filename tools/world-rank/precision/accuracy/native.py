"""Call the actual native public entry on cropped, protected input views."""
import ctypes as C
import hashlib
import mmap
from pathlib import Path

U32=C.c_uint32
FP=C.POINTER(C.c_float)
LIBC=C.CDLL(None,use_errno=True)
LIBC.mprotect.argtypes=(C.c_void_p,C.c_size_t,C.c_int)
PAD=0x4af027a1


def storage(data, rows, cols, stride, off, readonly):
    words = off + max(rows,1)*stride + 19
    mm=mmap.mmap(-1,words*4)
    memory=(U32*words).from_buffer(mm)
    for q in range(words):
        memory[q]=PAD
    for i in range(rows):
        for j in range(cols):
            memory[off+i*stride+j]=data[i*cols+j]
    if readonly and LIBC.mprotect(C.addressof(memory),len(mm),mmap.PROT_READ):
        raise OSError(C.get_errno(),'mprotect')
    return mm,memory,C.cast(C.byref(memory,4*off),FP)


class Native:
    def __init__(self,path,control_path=None):
        self.path=Path(path).resolve()
        self.digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.lib=C.CDLL(str(self.path))
        self.lib.precision_sgemm.argtypes=(C.c_int,)*4+(FP,C.c_int,FP,C.c_int,FP,C.c_int)
        self.lib.precision_sgemm.restype=C.c_int
        self.lib.precision_last_reason.restype=C.c_char_p
        self.lib.precision_last_plan_impl.argtypes=(C.c_int,)
        self.lib.precision_last_plan_impl.restype=C.c_char_p
        self.control_path=Path(control_path).resolve() if control_path else self.path
        self.control_digest=hashlib.sha256(self.control_path.read_bytes()).hexdigest()
        controls=C.CDLL(str(self.control_path)) if control_path else self.lib
        self.fp32=getattr(controls,'precision_fp32_control',None) or controls.cob_sgemm_rowmajor
        self.fp32.argtypes=self.lib.precision_sgemm.argtypes[1:]
        self.fp32.restype=None

    def run(self,A,B,m,n,k,mode):
        lda,ldb,ldc=max(1,k)+5,n+7,n+9
        a,ab,ap=storage(A,m,k,lda,3,True)
        b,bb,bp=storage(B,k,n,ldb,5,True)
        c,cb,cp=storage([0x7fc0de34]*(m*n),m,n,ldc,7,False)
        args=(m,n,k,ap,lda,bp,ldb,cp,ldc)
        if mode=='fp32-control':
            self.fp32(*args)
            meta=dict(returncode=0,route='direct-fp32',primitive_calls=0,reason='fp32-control',plans=[])
        else:
            rc=self.lib.precision_sgemm(mode,*args)
            why=self.lib.precision_last_reason()
            plans=[dict(k=self.lib.precision_last_plan_k(i), native=self.lib.precision_last_plan_native(i),
                        implementation=self.lib.precision_last_plan_impl(i).decode())
                   for i in range(self.lib.precision_last_plan_count())]
            meta=dict(returncode=rc,route=self.lib.precision_last_route(),
                      primitive_calls=self.lib.precision_last_products(),reason=why.decode() if why else '',plans=plans)
        meta['sha256']=self.control_digest if mode=='fp32-control' else self.digest
        result=[]
        for idx,value in enumerate(cb):
            i,j=divmod(idx-7,ldc)
            if 0<=i<m and j<n:
                result.append(value)
            elif value!=PAD:
                raise RuntimeError(f'native output padding modified at {idx}')
        return result,meta
