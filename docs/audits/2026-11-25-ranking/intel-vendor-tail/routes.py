"""Check real dispatch and full public output in a fresh Linux process per shared library."""
import argparse
from array import array
import csv
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys

DIMS = ('m', 'n', 'k', 'pad')
COUNTS = ('selected', 'main_rows', 'tail_rows', 'vendor_calls', 'vendor_m', 'vendor_n', 'vendor_k', 'tail_completed')
NAMES = tuple('vendor-wide-'+p+'-'+mode for p in ('openblas','mkl') for mode in ('full','split'))


def expected(name, m, n, k):
    selected = 48 <= m <= 256 and 512 <= n <= 4096 and n >= 4*m and 96 <= k <= 768 and m*k <= 98304
    if not selected:
        return [0]*8
    tail = m%16 if name.endswith('-split') and 1 <= m%16 <= 4 else 0
    return [1,m-tail,tail,1,m-tail,n,k,int(bool(tail))]


def fixture(path, traced, m, n, k, pad):
    if len(os.sched_getaffinity(0)) != 1 or len(os.listdir('/proc/self/task')) != 1:
        raise ValueError('fixture needs one thread and singleton affinity')
    lib = ct.CDLL(str(path),mode=os.RTLD_LOCAL|os.RTLD_NOW)
    if lib.rank_vendor_init() != 0:
        raise ValueError('vendor init failed')
    snap = getattr(lib,'rank_wide_snapshot',None)
    if bool(snap) != traced:
        raise ValueError('unexpected diagnostic-symbol state')
    fp = ct.POINTER(ct.c_float)
    call = lib.rank_vendor_sgemm
    call.argtypes = [ct.c_int,ct.c_int,ct.c_int,fp,ct.c_int,fp,ct.c_int,fp,ct.c_int]
    call.restype = None
    la,lb,lc = k+pad,n+pad,n+pad
    guard = 32
    sentinel = -913.25
    def matrix(rows,width,ld,fill):
        buf = array('f',[sentinel])*guard
        row = array('f',[fill])*width + array('f',[sentinel])*(ld-width)
        buf.extend(row*rows);buf.extend(array('f',[sentinel])*guard)
        return buf
    a,b,c=matrix(m,k,la,0),matrix(k,n,lb,0),matrix(m,n,lc,float('nan'))
    for i in range(m):
        a[guard+i*la] = i%17-8
        a[guard+i*la+k-1] = i%5+1
    right0 = [j%13-6 for j in range(n)]
    right1 = [j%7-3 for j in range(n)]
    b[guard:guard+n] = array('f',right0)
    last = guard+(k-1)*lb
    b[last:last+n] = array('f',right1)
    before_a,before_b=a.tobytes(),b.tobytes()
    def ptr(buf):
        return ct.cast(buf.buffer_info()[0]+guard*buf.itemsize,fp)
    slots=(ct.c_uint64*8)()
    if snap:
        snap.argtypes=[ct.POINTER(ct.c_uint64)];snap.restype=None
        snap(slots)
        if any(slots):raise ValueError('nonzero trace before first public operation')
    call(m,n,k,ptr(a),la,ptr(b),lb,ptr(c),lc)
    if snap:snap(slots)
    if a.tobytes()!=before_a or b.tobytes()!=before_b:
        raise ValueError('input changed')
    if c[:guard]!=array('f',[sentinel])*guard or c[-guard:]!=array('f',[sentinel])*guard:
        raise ValueError('outer output guard changed')
    for i in range(m):
        off=guard+i*lc;x,y=i%17-8,i%5+1
        exact=array('f',(x*u+y*v for u,v in zip(right0,right1)))
        if c[off:off+n]!=exact or any(v!=sentinel for v in c[off+n:off+lc]):
            raise ValueError(f'public result or padding differs at row {i}')
    if len(os.listdir('/proc/self/task'))!=1:
        raise ValueError('provider left multiple live threads')
    return {'checked':m*n,'cpu':next(iter(os.sched_getaffinity(0))),'slots':list(slots)}


def run(build,root,panel):
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    with panel.open(newline='') as f:cases=list(csv.DictReader(f))
    fields=['implementation','family',*DIMS,*COUNTS,'timed_checked','trace_checked','guards','inputs_unchanged','threads','cpu','passed','error']
    failures=0;root.mkdir(parents=True,exist_ok=True)
    with (root/'routes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name in NAMES:
            for case in cases:
                dims=[int(case[k]) for k in DIMS]
                row=dict(implementation=name,family=case['family'],**dict(zip(DIMS,dims)),passed=0,cpu=cpu)
                try:
                    for kind,ext in (('timed','.so'),('trace','.trace.so')):
                        proc=subprocess.run([sys.executable,__file__,'--fixture',str((build/(name+ext)).resolve()),kind,*map(str,dims)],
                                            text=True,capture_output=True,timeout=120)
                        if proc.returncode:raise ValueError(f'{kind}: {proc.stderr.strip()[-700:] or proc.returncode}')
                        result=json.loads(proc.stdout.splitlines()[-1]);row[kind+'_checked']=result['checked']
                        if result['cpu'] != cpu:raise ValueError(f'{kind}: affinity changed')
                    row.update(zip(COUNTS,result['slots']))
                    wanted=expected(name,*dims[:3])
                    if result['slots']!=wanted:raise ValueError(f'trace {result["slots"]}, expected {wanted}')
                    row.update(guards=1,inputs_unchanged=1,threads=1,passed=1)
                except (subprocess.SubprocessError,IndexError,KeyError,OSError,ValueError) as exc:
                    row['error']=str(exc);failures+=1
                writer.writerow(row);f.flush()
    return failures


if __name__=='__main__':
    if sys.argv[1:2]==['--fixture']:
        print(json.dumps(fixture(Path(sys.argv[2]),sys.argv[3]=='trace',*map(int,sys.argv[4:]))))
    else:
        parser=argparse.ArgumentParser();parser.add_argument('build',type=Path);parser.add_argument('out',type=Path);parser.add_argument('panel',type=Path)
        args=parser.parse_args()
        if run(args.build,args.out,args.panel):raise SystemExit('route or fixture qualification failed')
