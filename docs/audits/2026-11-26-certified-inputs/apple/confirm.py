"""One preregistered fresh-process M5 replication with separately owned buffers."""
import argparse
import csv
import ctypes as ct
import hashlib
import itertools
import json
import platform
from pathlib import Path
import random
import statistics
import time

import numpy as np
from study import ABI, get_data, hw_string

EXPECTED = {
    'optical-counts': 'ec47b13672085cc2096a990dd166c76e2e29843c7f79ea88d1d483db94b05db0',
    'optical-binary': 'eec353b47234bfe0fa3544331dc16aceee6a9b1ee3b7165b51e30fe565a9e825',
}

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f4')).hexdigest()

def aligned(shape):
    size = int(np.prod(shape))
    storage = np.empty(size + 16, dtype=np.float32)
    offset = ((-storage.ctypes.data) % 64) // 4
    return storage[offset:offset + size].reshape(shape)

def dump(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    kernel = ct.CDLL(str(args.kernel.resolve()))
    cob = ct.CDLL(str(args.cob.resolve()))
    assert kernel.apple_int8_lanes() == 16 and kernel.exact_blas_serial() == 1
    functions = {'exact-int8': (kernel.apple_exact, ct.c_int),
                 'accelerate': (kernel.exact_baseline, None),
                 'cob': (cob.cob_sgemm_rowmajor, None),
                 'binary-bitset': (kernel.binary_dot, ct.c_int)}
    for f, t in functions.values():
        f.argtypes, f.restype = ABI, t
    meta = {'cpu':hw_string('machdep.cpu.brand_string'), 'macos':platform.mac_ver()[0],
            'machine':platform.machine(), 'numpy':np.__version__, 'affinity':'unavailable',
            'serial_accelerate_setting':kernel.exact_blas_serial(),'svl_words':kernel.apple_int8_lanes(),
            'libraries':{name:hashlib.sha256(path.read_bytes()).hexdigest()
                         for name,path in [('kernel', args.kernel),('cob',args.cob)]}}
    rows, checks = [], []
    for case, A, B, _, binary in get_data(args.corpus):
        if case not in EXPECTED:
            continue
        m,k = A.shape
        _,n = B.shape
        names = list(functions) if binary else list(functions)[:3]
        copies = []
        a_hash,b_hash = digest(A),digest(B)
        for index in range(5):
            a,b,storage = aligned(A.shape),aligned(B.shape),aligned((m+2,n+8))
            a[:],b[:] = A,B
            storage.fill(918.5)
            out = storage[1:-1,3:3+n]
            assert all(x.ctypes.data % 64 == 0 for x in (a,b,storage))
            copies.append((a,b,storage,out))
        blocks=[v for a,b,c,_ in copies for v in (a,b,c)]
        assert not any(np.shares_memory(x,y) for x,y in itertools.combinations(blocks,2))
        def call(name,slot,run,pos):
            a,b,c,out = copies[slot]
            out.fill(np.nan)
            arg=(m,n,k,a.ctypes.data,k,b.ctypes.data,n,out.ctypes.data,n+8)
            f=functions[name][0]
            used0=time.thread_time_ns()
            t0=time.perf_counter_ns()
            result=f(*arg)
            wall=(time.perf_counter_ns()-t0)*1e-9
            used=(time.thread_time_ns()-used0)*1e-9
            assert result == (1 if name in ('exact-int8','binary-bitset') else None)
            assert digest(out) == EXPECTED[case]
            assert all(np.all(x==918.5) for x in (c[0],c[-1],c[1:-1,:3],c[1:-1,3+n:]))
            if run >= 0:
                rows.append(dict(case=case,m=m,n=n,k=k,round=run,position=pos,method=name,slot=slot,
                                 seconds=wall,thread_seconds=used,route='' if result is None else result))
        for pos,name in enumerate(names):
            call(name,0,-1,pos)
        orders=list(itertools.permutations(names))
        orders*=24//len(orders)
        random.Random(33114791+len(names)).shuffle(orders)
        for r,order in enumerate(orders):
            for pos,name in enumerate(order):
                call(name,r%5,r,pos)
        for index,(a,b,_,out) in enumerate(copies):
            assert digest(a)==a_hash and digest(b)==b_hash
            checks.append(dict(case=case,slot=index,distinct_allocations=3,lda=k,ldb=n,ldc=n+8,
                               a_mod64=a.ctypes.data%64,b_mod64=b.ctypes.data%64,c_mod64=out.ctypes.data%64,
                               cells=m*n,A_sha256=a_hash,B_sha256=b_hash,output_sha256=EXPECTED[case],
                               all_method_call_guards_passed=1,inputs_unchanged=1))
    dump(args.out/'timings.csv',rows)
    dump(args.out/'checks.csv',checks)
    (args.out/'identity.json').write_text(json.dumps(meta,indent=2)+'\n')
    summary={}
    for case in EXPECTED:
        times={}
        for r in rows:
            if r['case']==case:
                times.setdefault(r['method'],{})[r['round']]=r['seconds']
        candidate=times['exact-int8']; summary[case]={}
        for name,obs in times.items():
            assert set(obs)==set(range(24))
            ratios=[obs[r]/candidate[r] for r in range(24)]
            summary[case][name]={'median_ms':1000*statistics.median(obs.values()),
                'over_exact_int8':statistics.median(ratios),'exact_faster':sum(t>1 for t in ratios),
                'pairs':len(ratios)}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('kernel','cob','corpus','out'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
