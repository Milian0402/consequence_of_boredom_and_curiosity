"""Frozen compact full-call M5 pilot for exact integer inputs and ordinary FP32 fallback."""
import argparse,csv,ctypes as ct,hashlib,itertools,json,math,platform,statistics,time
from pathlib import Path
import numpy as np

P=ct.c_void_p;I=ct.c_int;D=ct.c_double
ABI=[I,I,I,P,I,P,I,P,I]
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def src(folder,name,shape,expected):
    raw=(folder/name).read_bytes();assert sha_bytes(raw)==expected and len(raw)==math.prod(shape)
    return np.frombuffer(raw,dtype=np.uint8).reshape(shape).astype(np.float32)

def get_data(folder):
    a=src(folder,'digits_A.u8',(1797,64),'8f26b2bd9d135c256808f68f14fdabddde6d9c7f869ae419704b051f0f14b3b3')
    b=src(folder,'digits_B.u8',(64,3823),'89564f6b531142d78068f44701ba3b267c702f54d597a0ad11aa9f3201173e07')
    yield 'optical-counts',a,b,True,False
    rng=np.random.default_rng(33859187)
    yield 'general-float',rng.random(a.shape,dtype=np.float32)-.5,rng.random(b.shape,dtype=np.float32)-.5,False,False
    yield 'integer-cube',rng.integers(-63,64,size=(1024,1024),dtype=np.int8).astype(np.float32),rng.integers(-63,64,size=(1024,1024),dtype=np.int8).astype(np.float32),True,False
    a=src(folder,'optdigits-original-windep.u8',(1797,1024),'48c3d70dcd05ff9cdcb6d4cf2074a11725db988bed683385e332b8be259dae63')
    b=src(folder,'optdigits-original-tra.u8',(1934,1024),'07e2b5395893a72a71d35558d76c766a9ca9e61f5df2a0a498fcb2c2dcb3b9a1')
    yield 'optical-binary',a,np.ascontiguousarray(b.T),True,True

def hw_string(key):
    lib=ct.CDLL(None);size=ct.c_size_t(128);buf=ct.create_string_buffer(128)
    if lib.sysctlbyname(key.encode(),buf,ct.byref(size),None,0):return None
    return buf.value.decode()

def run(args):
    E=ct.CDLL(str(args.kernel.resolve()));C=ct.CDLL(str(args.cob.resolve()));O=ct.CDLL(str(args.oracle.resolve()))
    O.openblas_set_num_threads(1); assert O.openblas_get_num_threads()==1
    assert E.apple_int8_lanes()==16 and E.exact_blas_serial()==1
    choices={'exact-int8':(E.apple_exact,I),'accelerate':(E.exact_baseline,None),'cob':(C.cob_sgemm_rowmajor,None),'binary-bitset':(E.binary_dot,I)}
    for f,t in choices.values():f.argtypes=ABI;f.restype=t
    dgemm=O.cblas_dgemm;dgemm.argtypes=[I,I,I,I,I,I,D,P,I,P,I,D,P,I];dgemm.restype=None
    args.out.mkdir(exist_ok=True,parents=True);obs=[];checks=[];meta={'cpu':hw_string('machdep.cpu.brand_string'),'macos':platform.mac_ver()[0],'machine':platform.machine(),'numpy':np.__version__,'time_unit':'seconds','affinity':'unavailable','serial_accelerate_setting':E.exact_blas_serial(),'reference_threads':O.openblas_get_num_threads(),'svl_words':E.apple_int8_lanes(),'libraries':{k:sha_bytes(p.read_bytes()) for k,p in [('kernel',args.kernel),('cob',args.cob),('openblas',args.oracle)]}}
    (args.out/'identity.json').write_text(json.dumps(meta,indent=2)+'\n')
    summ={}
    for key,a,b,integer,binary in get_data(args.corpus):
        m,k=a.shape;kb,n=b.shape;assert k==kb
        assert a.flags.c_contiguous and b.flags.c_contiguous
        ha,hb=sha_bytes(memoryview(a)),sha_bytes(memoryview(b))
        ac,bc=a.astype(np.float64),b.astype(np.float64)
        g=np.empty((m,n),dtype=np.float64)
        dgemm(101,111,111,m,n,k,1.,ac.ctypes.data,k,bc.ctypes.data,n,0.,g.ctypes.data,n)
        gold=g.astype(np.float32)
        if integer:assert np.array_equal(g,gold)
        for i,j in [(0,0),(m//2,n//2),(m-1,n-1)]:
            exact=sum(int(x)*int(y) for x,y in zip(a[i],b[:,j])) if integer else math.fsum(float(x)*float(y) for x,y in zip(a[i],b[:,j]))
            assert g[i,j]==exact if integer else abs(g[i,j]-exact)<=1e-9
        guard=np.float32(918.5);buf=np.full((m+2,n+8),guard,dtype=np.float32); out=buf[1:-1,3:3+n]
        methods=['exact-int8','accelerate','cob']+(['binary-bitset'] if binary else [])
        plain=np.empty((m,n),dtype=np.float32);arg=(m,n,k,a.ctypes.data,k,b.ctypes.data,n,out.ctypes.data,n+8)
        E.exact_baseline(*arg);plain[:]=out
        def verify(name):
            assert np.all(buf[:1]==guard) and np.all(buf[-1:]==guard) and np.all(buf[1:-1,:3]==guard) and np.all(buf[1:-1,3+n:]==guard)
            if integer and name in ('exact-int8','binary-bitset'):assert np.array_equal(out,gold),(key,name)
            if not integer and name=='exact-int8':assert np.array_equal(out,plain)
        def check_route(name,rc):
            if name=='exact-int8':assert rc==int(integer),(key,name,rc)
            if name=='binary-bitset':assert rc==1
        for name in methods:
            f,_=choices[name];out.fill(np.nan);rc=f(*arg);check_route(name,rc);verify(name)
            diff=out.astype(np.float64)-g
            checks.append(dict(case=key,m=m,n=n,k=k,method=name,route=rc if rc is not None else '',checked=m*n,fp32_mismatches=int(np.count_nonzero(out!=gold)),max_abs=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff*diff))),A_sha256=ha,B_sha256=hb))
        times={name:[] for name in methods}; rounds=3*len(methods)
        for rep in range(rounds):
            seq=list(reversed(methods)) if (rep//len(methods))%2 else methods
            ix=rep%len(methods); order=seq[ix:]+seq[:ix]
            for pos,name in enumerate(order):
                f,_=choices[name];out.fill(np.nan)
                cpu=time.thread_time_ns();start=time.perf_counter_ns();rc=f(*arg);end=time.perf_counter_ns();used=(time.thread_time_ns()-cpu)*1e-9
                check_route(name,rc);verify(name);duration=(end-start)*1e-9
                obs.append(dict(case=key,m=m,n=n,k=k,round=rep,position=pos,method=name,seconds=duration,thread_seconds=used,route=rc if rc is not None else ''))
                times[name].append(duration)
        assert ha==sha_bytes(memoryview(a)) and hb==sha_bytes(memoryview(b))
        paired=lambda base,x:[u/v for u,v in zip(times[base],times[x],strict=True)]
        details={name:dict(median_ms=1000*statistics.median(times[name]),vs_accelerate=statistics.median(paired('accelerate',name)),vs_cob=statistics.median(paired('cob',name)),wins_accel=sum(r>1 for r in paired('accelerate',name)),wins_cob=sum(r>1 for r in paired('cob',name)),rounds=rounds) for name in methods}
        summ[key]=details
        print(key,details,flush=True)
    for file,rows in [('timings.csv',obs),('accuracy.csv',checks)]:
        with (args.out/file).open('w',newline='') as fd: w=csv.DictWriter(fd,fieldnames=tuple(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(summ,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('kernel','cob','oracle','corpus','out'):p.add_argument('--'+a,type=Path,required=True)
    run(p.parse_args())
