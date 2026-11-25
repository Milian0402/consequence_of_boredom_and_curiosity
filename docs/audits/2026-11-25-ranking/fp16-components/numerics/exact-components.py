"""Component-rounding floor, independent of machine conversion and reduction order."""
import csv
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
from locate import reference
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--repo')
reference(ap.parse_args().repo)
from evaluate import score
from fixtures import CASES, KINDS, build

from oracle import exact, units, round_fp32
from quality import compare
from registration import SEEDS, verify

def f32(bits):
    return struct.unpack('<f',struct.pack('<I',bits))[0]

def bits(x):
    return struct.unpack('<I',struct.pack('<f',x))[0]

def half(x):
    return struct.unpack('<e',struct.pack('<e',x))[0]

def fp16_parts(x,shift):
    # The subtractions of two close exact dyadics and the shifts are exact here.
    v=math.ldexp(x,shift)
    h=half(v)
    r=math.ldexp(v-h,11)
    m=half(r)
    l=half(math.ldexp(r-m,11))
    h,m,l=(math.ldexp(h,-shift),math.ldexp(m,-shift-11),math.ldexp(l,-shift-22))
    assert h+m+l==x
    return [units(bits(t)) for t in (h,m,l)]

def get(case,seed):
    m,n,k,A,B=build(case,seed)
    aval,bval=list(map(f32,A)),list(map(f32,B))
    def shift(seq):
        maximum=max(map(abs,seq))
        return 13-math.frexp(maximum)[1] if maximum else 0
    ar=[shift(aval[i*k:(i+1)*k]) for i in range(m)]
    bc=[shift(bval[j::n]) for j in range(n)]
    a=[fp16_parts(x,ar[i//k]) for i,x in enumerate(aval)]
    b=[fp16_parts(x,bc[i%n]) for i,x in enumerate(bval)]
    return m,n,k,A,B,a,b

TERMS={1:((0,0),),3:((0,0),(0,1),(1,0)),4:((0,0),(0,1),(1,0),(1,1)),
       6:tuple((p,q) for p in range(3) for q in range(3) if p+q<=2),9:tuple((p,q) for p in range(3) for q in range(3))}

def main():
    verify()
    def load(name):
        with (HERE/'data'/name).open() as f:return list(csv.DictReader(f))
    blocks={r['sha256']:bytes.fromhex(r['little_endian_binary32_hex']) for r in load('words.csv')}
    controls={(r['case'],int(r['seed'])):blocks[r['control_output_sha256']] for r in load('scores.csv')}
    rows=[]
    for case in CASES:
        if KINDS[case]!='normal':continue
        for seed in SEEDS:
            m,n,k,A,B,a,b=get(case,seed)
            direct=list(struct.unpack(f'<{m*n}I',controls[case,seed]))
            R=exact(A,B,m,n,k)
            components=[]
            for i in range(m):
                for j in range(n):
                    components.append({(p,q):sum(a[i*k+t][p]*b[t*n+j][q] for t in range(k)) for p in range(3) for q in range(3)})
            for count,terms in TERMS.items():
                out=[round_fp32(sum(v[t] for t in terms),-298) for v in components]
                metrics,_=score(R,out,n,k)
                qual=compare(R,out,direct)
                rows.append(dict(case=case,seed=seed,mode=count,**metrics,**qual))
    archived=load('exact-component-sums.csv')
    assert len(archived)==len(rows)
    for row,saved in zip(rows,archived,strict=True):
        for key,value in row.items():
            if isinstance(value,float):
                assert math.isclose(float(saved[key]),value,rel_tol=2e-15,abs_tol=0)
            else:
                assert ('' if value is None else str(value))==saved[key],(key,value,saved[key])
    for mode in TERMS:
        g=[r for r in rows if r['mode']==mode]
        print(mode,[(r['case'],hex(r['seed'])) for r in g if not r['quality_rms_pass'] or not r['quality_max_pass']],
              'wrong rounded',sum(r['wrong_rounded'] for r in g))
if __name__=='__main__':main()
