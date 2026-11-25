"""Native Apple FP16 feasibility panel. All CLI arguments after --plugins are dylib paths."""
import argparse
import csv
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

P=ct.c_void_p
I=ct.c_int
F=ct.c_float
D=ct.c_double
ABI=[I,I,I,P,I,P,I,P,I]
SENTINEL=np.float32(711.75)
SHAPES=[(512,512,512),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),
        (1471,1973,1559),(187,2711,433)]
SEEDS=[204716321,745821359]
METHODS=["accelerate","cob","maxmulsk","kleidiai","fp16-raw","fp16-three"]

def symbol(lib,name,args,returns):
    f=getattr(lib,name)
    f.argtypes=args
    f.restype=returns
    return f

def storage(rows,cols,padded,extra,origin):
    ld=cols+extra if padded else cols
    offset=ld+origin if padded else 0
    length=offset+rows*ld+128
    space=np.empty(length+32,dtype=np.float32)
    skip=(-space.ctypes.data%128)//4
    raw=space[skip:skip+length]
    raw.fill(SENTINEL)
    return raw,raw[offset:offset+rows*ld].reshape(rows,ld)[:,:cols],ld

def generate(a,b,fixture,seed):
    g=np.random.Generator(np.random.PCG64(seed))
    a[:]=g.random(a.shape,dtype=np.float32)
    b[:]=g.random(b.shape,dtype=np.float32)
    if fixture!="positive":
        a-=np.float32(.5)
        b-=np.float32(.5)
    if fixture=="cancellation":
        pairs=a.shape[1]//2
        a[:,1:2*pairs:2]=a[:,:2*pairs:2]
        b[1:2*pairs:2,:]=-b[:2*pairs:2,:]+np.float32(2**-19)

def digest(data):
    return hashlib.sha256(memoryview(data)).hexdigest()

def gold(dgemm,a,b,seed):
    m,k=a.shape
    n=b.shape[1]
    da=np.ascontiguousarray(a,dtype=np.float64)
    db=np.ascontiguousarray(b,dtype=np.float64)
    target=np.zeros((m,n),dtype=np.float64)
    dgemm(101,111,111,m,n,k,1.,da.ctypes.data,k,db.ctypes.data,n,0.,target.ctypes.data,n)
    g=np.random.default_rng(seed+19)
    for i,j in zip(g.integers(m,size=19),g.integers(n,size=19)):
        expected=math.fsum(float(x)*float(y) for x,y in zip(a[i],b[:,j]))
        if abs(expected-target[i,j])>2e-9: raise ValueError("double oracle disagreement")
    return target

def metrics(c,gold):
    delta=np.abs(c.astype(np.float64)-gold)
    return {"max_abs":float(np.max(delta,initial=0.)),"rms":float(np.sqrt(np.mean(delta*delta))),
            "over_abs":int(np.count_nonzero(delta>.002)),"nonfinite":int(np.count_nonzero(~np.isfinite(c))),
            "ref_max":float(np.max(np.abs(gold))),"ref_rms":float(np.sqrt(np.mean(gold*gold)))}

def study(args):
    paths=[Path(x).resolve() for x in args.plugins]
    accel,cob,maxlib,adapt,kleidi,new,openblas=paths
    os.environ["COMPACT_MAXMULSK"]=str(maxlib)
    libs=[ct.CDLL(str(x),mode=ct.RTLD_LOCAL) for x in paths]
    launch=ct.c_int(1)
    open_threads=symbol(libs[6],"openblas_get_num_threads",[],I)
    assert open_threads()==1
    dgemm=symbol(libs[6],"cblas_dgemm",[I,I,I,I,I,I,D,P,I,P,I,D,P,I],None)
    assert symbol(libs[5],"cob_half_svl",[],I)()>0
    route=symbol(libs[5],"apple_fp16_last_route",[],I)
    simple=[symbol(libs[0],"cob_sgemm_rowmajor",ABI,None),
            symbol(libs[1],"cob_sgemm_rowmajor",ABI,None)]
    compact=symbol(libs[2],"mtmm_sgemm_rowmajor",ABI+[ct.POINTER(I)],None)
    converted=symbol(libs[3],"cob_sgemm_rowmajor",ABI,None)
    kleid=symbol(libs[4],"mtmm_sgemm_rowmajor",ABI+[ct.POINTER(I)],None)
    raw=symbol(libs[5],"apple_fp16_raw",ABI,I)
    three=symbol(libs[5],"apple_fp16_three",ABI,I)
    high=getattr(args,"qualified_dylib",None)
    methods=METHODS
    if high:
        qualified=ct.CDLL(str(Path(high).resolve()),mode=ct.RTLD_LOCAL)
        run_terms=symbol(qualified,"precision_sgemm",[I]+ABI,I)
        route=symbol(qualified,"precision_last_route",[],I)
        terms=symbol(qualified,"precision_last_products",[],I)
        methods=METHODS[:4]+["fp16-six","fp16-nine"]
    expected=(6,9) if high else (1,3)

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"identity.json").write_text(json.dumps({"machine":platform.machine(),"system":platform.platform(),
            "streaming_fp32_lanes":int(libs[5].cob_half_svl()),
            "libraries":{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
            "qualified_dylib_sha256":hashlib.sha256(Path(high).read_bytes()).hexdigest() if high else None,
            "numpy":np.__version__,"one_thread_openblas":open_threads()},indent=2))
    accf=open(out/"accuracy.csv","w",newline=""); timef=open(out/"time.csv","w",newline="")
    def writer(f,cols):
        w=csv.DictWriter(f,fieldnames=cols.split());w.writeheader();return w
    aw=writer(accf,"shape m n k seed fixture method route cells max_abs rms over_abs nonfinite ref_max ref_rms")
    tw=writer(timef,"shape m n k seed method round position seconds thread_seconds")
    for h,(m,n,k) in enumerate(SHAPES):
        padded=h>=4
        ap,a,la=storage(m,k,padded,11,3);bp,b,lb=storage(k,n,padded,19,5);cp,c,lc=storage(m,n,padded,23,7)
        outputs=np.ones(cp.size,dtype=bool)
        offset=(c.ctypes.data-cp.ctypes.data)//4
        outputs[offset:offset+m*lc].reshape(m,lc)[:,:n]=False
        common=(m,n,k,a.ctypes.data,la,b.ctypes.data,lb,c.ctypes.data,lc)
        cfg=ct.pointer(launch)
        fs=[(simple[0],common),(simple[1],common),
            (converted,common) if padded else (compact,common+(cfg,)),(kleid,common+(cfg,)),(raw,common),(three,common)]
        if high:fs[4:]=[(run_terms,(6,)+common),(run_terms,(9,)+common)]
        for sid,seed in enumerate(SEEDS):
            for fixture in (["centered","positive","cancellation"] if sid==0 else ["centered"]):
                generate(a,b,fixture,seed+7919*h)
                original=digest(ap),digest(bp)
                reference=gold(dgemm,a,b,seed)
                active=[]
                for idx,(fn,fnargs) in enumerate(fs):
                    c.fill(np.nan)
                    ret=fn(*fnargs)
                    base=dict(shape=h,m=m,n=n,k=k,seed=seed,fixture=fixture,method=methods[idx],cells=m*n)
                    if idx>=4 and high and ret==0 and route()==2:
                        assert (original==(digest(ap),digest(bp)) and np.all(cp[outputs]==SENTINEL))
                        aw.writerow(dict(**base,route="fallback",**metrics(c,reference)));continue
                    if idx>=4 and ret!=(0 if high else 1):
                        assert np.all(np.isnan(c));aw.writerow(dict(**base,route="refused"))
                    else:
                        if idx>=4: assert (route()==1 and terms()==expected[idx-4]) if high else route()==expected[idx-4]
                        assert (original==(digest(ap),digest(bp)) and np.all(cp[outputs]==SENTINEL))
                        aw.writerow(dict(**base,route="executed",**metrics(c,reference))); active.append(idx)
                accf.flush()
                if fixture!="centered": continue
                for rep in range(6):
                    order=active if sid==0 else list(reversed(active))
                    q=(rep+h)%len(order);order=order[q:]+order[:q]
                    for pos,idx in enumerate(order):
                        fn,fnargs=fs[idx]
                        c.fill(np.nan)
                        tick=time.thread_time_ns(); start=time.perf_counter_ns(); ret=fn(*fnargs); end=time.perf_counter_ns(); cpu=time.thread_time_ns()-tick
                        if idx>=4:assert (ret==0 and route()==1 and terms()==expected[idx-4]) if high else (ret==1 and route()==expected[idx-4])
                        assert np.all(cp[outputs]==SENTINEL) and np.isfinite(c).all()
                        tw.writerow(dict(shape=h,m=m,n=n,k=k,seed=seed,method=methods[idx],round=rep,position=pos,
                                         seconds=(end-start)*1e-9,thread_seconds=cpu*1e-9))
                assert original==(digest(ap),digest(bp))
                timef.flush()
    accf.close();timef.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--plugins",nargs=7,required=True,metavar="DYLIB",
                        help="Accelerate COB MaxMulSK adapted-MaxMulSK KleidiAI FP16 OpenBLAS")
    parser.add_argument("--qualified-dylib",help="Native FP16 six- and nine-product complete frontend")
    study(parser.parse_args())
