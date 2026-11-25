#!/usr/bin/env python3
"""Compare the two actual HK15 guard functions in separate local shared libraries."""
import argparse
from array import array
from collections import Counter
import csv
import ctypes as ct
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tarfile
import tempfile

from random_cases import cases as random_cases, expected_policy

BASE = '85a007747ff54241f47c4754216e0a65748f2847'
F = ct.c_float
P = ct.POINTER(F)
I = ct.c_int
ARGS = [I,I,I,P,I,P,I]
WRAPPER = '''
#include "gemm.c"
int qa_policy(int m,int n,int k,const float *a,int la,const float *b,int lb)
{ return cob_hk15_policy_for(m,n,k,a,la,b,lb); }
int qa_shape(int m,int n,int k) { return cob_hk15_shape(m,n,k); }
#if QA_CANDIDATE
int qa_certificate(int m,int n,int k,const float *a,int la,const float *b,int lb)
{
    uint32_t columns[3842];
    return cob_hk15_regular_certificate(m,n,k,a,la,b,lb,columns);
}
int qa_coherence(int m,int n,int k,const float *a,int la,const float *b,int lb)
{ return cob_hk15_has_coherent_dots(m,n,k,a,la,b,lb); }
uint32_t qa_bound(int rows,int cols,const float *x,int ld)
{ return cob_hk15_bound(rows,cols,x,ld); }
int qa_witness(int rows,int cols,const float *x,int ld,uint32_t upper)
{ return !!upper && cob_hk15_axis_witnesses(rows,cols,x,ld,upper); }
#endif
'''
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(root,tmp,which,cc):
    source=tmp/(which+'.c')
    target=tmp/(which+'.so')
    source.write_text(WRAPPER)
    flags=['-O3','-std=c11','-fPIC','-DCOB_ENABLE_HK15=1','-DCOB_ENABLE_HK15_RAGGED=1',
           '-DCOB_DISABLE_APPLE_SME=1','-DCOB_DISABLE_APPLE_AMX=1']
    if platform.machine().lower() in ('arm64','aarch64'):
        flags.append('-DCOB_USE_NEON=1')
    subprocess.run([cc,*flags,'-DQA_CANDIDATE='+str(int(which=='candidate')),
                    '-I'+str(root/'src'),'-I'+str(root/'include'),
                    '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),
                    str(root/'src/cob_hk15_ragged.c'),'-lm','-o',str(target)],check=True)
    lib=ct.CDLL(str(target))
    for fn in ('qa_policy', 'qa_certificate','qa_coherence') if which=='candidate' else ('qa_policy',):
        getattr(lib,fn).argtypes=ARGS
        getattr(lib,fn).restype=I
    lib.qa_shape.argtypes=[I,I,I]; lib.qa_shape.restype=I
    if which=='candidate':
        lib.qa_bound.argtypes=[I,I,P,I];lib.qa_bound.restype=ct.c_uint32
        lib.qa_witness.argtypes=[I,I,P,I,ct.c_uint32];lib.qa_witness.restype=I
    return lib,flags

def grid(rows,cols,x=1.0):
    return [[x]*cols for _ in range(rows)]

def adjacent(value,delta):
    return struct.unpack('=f',struct.pack('=I',struct.unpack('=I',struct.pack('=f',value))[0]+delta))[0]

def hand_cases():
    m,n,k=19,37,40
    for operand in ('A','B'):
        for value,want in ((2.**-149,2),(adjacent(2.**-48,-1),2),(2.**-48,0),
                           (2.**48,0),(adjacent(2.**48,1),2)):
            a=grid(m,k,value if operand=='A' else 1.)
            b=grid(k,n,value if operand=='B' else 1.)
            yield f'{operand}-uniform-{value.hex()}','magnitude-bound',a,b,want
        for value,want in ((.25,0),(adjacent(.25,-1),1),(adjacent(.25,1),0),(-0.,1)):
            for axis in ('row','col'):
                a,b=grid(m,k),grid(k,n)
                x=a if operand=='A' else b
                if axis=='row': x[-1]=[value]*len(x[-1])
                else:
                    for row in x:row[-1]=value
                yield f'{operand}-{axis}-{value.hex()}','axis-maxima',a,b,want
        for value,want in ((1.5,0),(4.,0),(adjacent(4.,1),1),(math.inf,2),(-math.inf,2),
                           (math.nan,2),(2.**-149,2)):
            a,b=grid(m,k),grid(k,n)
            x=a if operand=='A' else b
            x[1][-2]=value # outside all four sampled row strips
            yield f'{operand}-unsampled-{value.hex()}','underestimated-bound',a,b,want
        a,b=grid(m,k),grid(k,n)
        x=a if operand=='A' else b
        x[0][1]=1.25
        # Rounding the nominated bound to 2 raises its sufficient support cutoff
        # to 1/4. Exact max / 8 is 0.15625; all 0.2 terms still qualify.
        for i,row in enumerate(x):
            for j in range(len(row)):
                if (j+i)%4 in (1,2) and (i,j)!=(0,1):row[j]=.2
        yield f'{operand}-loose-upper','certificate-too-conservative',a,b,0
    # Per-column prefix can miss a witness even though it exists farther down.
    a,b=grid(m,k),grid(k,n)
    for i in range(16):a[i][-1]=.2
    yield 'A-late-column-witness','exact-fallback',a,b,0
    a,b=grid(m,k),grid(k,n)
    for j in range(16):b[-1][j]=.2
    yield 'B-late-row-witness','exact-fallback',a,b,0
    for side in ('A','B'):
        a,b=grid(20,64),[[1. if p%2 else -1.]*53 for p in range(64)]
        x=a if side=='A' else b
        for t in range(4):
            row=(len(x)-1)*t//3
            first=(len(x[0])-16)*t//3
            x[row][first:first+16]=[0.]*16
        yield side+'-unrepresentative-zero-sample','sample-misses-bound',a,b,0
    for k in (32,33,34,35):
        for p in (4,5,6,7,k-1):
            a,b=grid(19,k),grid(k,49)
            b[p][17]=-adjacent(2.**-48,-1)
            yield f'B-unsafe-lane-{k}-{p}','four-lane-and-tail',a,b,2
    for k in (16,17,18,19,31,32,33,63):
        q=(3*k+3)//4
        for operand in ('A','B'):
            for live,want in ((q,0),(q-1,1)):
                a,b=grid(19,k),grid(k,37)
                for p in range(live,k):
                    if operand=='A':a[-1][p]=-adjacent(.125,-1)
                    else:b[p][-1]=-adjacent(.125,-1)
                yield f'{operand}-support-k{k}-{live}','three-quarter-ceil',a,b,want
    for v,want in ((-0.125,0),(-adjacent(.125,-1),1),(-adjacent(.125,1),0)):
        a,b=grid(19,32),grid(32,37)
        a[18][23]=v
        a[18][24:]=[0.]*8
        yield 'support-cutoff-'+v.hex(),'inclusive-eighth',a,b,want
    for right in (False,True):
        for lead in ('all-zero','sparse'):
            a,b=grid(m,k),grid(k,n)
            x,y=(a,b) if not right else (b,a)
            x[-1]=[0. if (lead=='all-zero' or j%4) else 1. for j in range(len(x[-1]))]
            y[1][-2]=float('nan')
            yield f'wide-over-classical-{right}-{lead}','exceptional-precedence',a,b,2
    def walsh(idx,p):return -1. if (idx&p).bit_count()%2 else 1.
    for hits in (0,3,4,5):
        for sign in (1.,-1.):
            a=[[walsh(i,p) for p in range(64)] for i in range(16)]
            b=[[sign*walsh(j if j<hits else 16+j,p) for j in range(16)] for p in range(64)]
            yield f'walsh-{hits}-{sign:.0f}','coherence-hit-count',a,b,int(hits>=4)
    for positive in (47,48,49):
        for sign in (1.,-1.):
            a=grid(16,64)
            b=[[sign*(1. if p<positive else -1.) if j==0 else (-1. if p%2 else 1.)
                for j in range(16)] for p in range(64)]
            yield f'cosine-{positive}-{sign:.0f}','strict-cosine',a,b,int(positive>48)
    for m,n,k in ((15,16,64),(16,15,64),(16,16,63),(16,16,64),(17,31,65)):
        yield f'coherence-min-{m}-{n}-{k}','coherence-size',grid(m,k),grid(k,n),int(m>=16 and n>=16 and k>=64)
    for m,n,k in ((3,3842,17),(3,19,3842)):
        yield f'array-limit-{n}-{k}','last-valid-size',grid(m,k),grid(k,n),0

def parent(mat,seed):
    rows,cols=len(mat),len(mat[0])
    ld=cols+1+seed%13
    offset=(1+seed%2)*ld+seed%3
    storage=(F*((rows+4)*ld+9))()
    # Every guard is exceptional. A single unintended read can change the policy.
    ct.memset(storage,255,ct.sizeof(storage))
    for i,row in enumerate(mat):
        packed=array('f',row)
        ct.memmove(ct.byref(storage,(offset+i*ld)*4),packed.buffer_info()[0],cols*4)
    return storage,ct.cast(ct.byref(storage,offset*4),P),ld,offset

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--base-repo',type=Path)
    parser.add_argument('--cc',default=os.environ.get('CC','cc'))
    args=parser.parse_args()
    candidate=args.candidate.resolve()
    base_repo=(args.base_repo or candidate).resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        td=Path(d); base=td/'base'; base.mkdir()
        src=subprocess.check_output(['git','-C',str(base_repo),'archive',BASE,'src','include','third_party/fast-matmul'])
        with tarfile.open(fileobj=io.BytesIO(src)) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    out=base/member.name;out.parent.mkdir(exist_ok=True,parents=True)
                    out.write_bytes(archive.extractfile(member).read())
        old,flags=build(base,td,'base',args.cc)
        new,_=build(candidate,td,'candidate',args.cc)
        rows=[]
        def check(name,family,m,n,k,ap,la,bp,lb,ao,bo,want,saved):
            before=tuple(hashlib.sha256(a).hexdigest() for a in saved)
            dimargs=(m,n,k,ap,la,bp,lb)
            old_policy=old.qa_policy(*dimargs);new_policy=new.qa_policy(*dimargs)
            cert=new.qa_certificate(*dimargs)
            ub_a=new.qa_bound(m,k,ap,la);ub_b=new.qa_bound(k,n,bp,lb)
            witness_a=new.qa_witness(m,k,ap,la,ub_a);witness_b=new.qa_witness(k,n,bp,lb,ub_b)
            if not ub_a:stage='A-bound'
            elif not ub_b:stage='B-bound'
            elif not witness_a:stage='A-axis-witnesses'
            elif not witness_b:stage='B-axis-witnesses'
            elif not cert:stage='full-range-or-depth'
            else:stage='certified'
            coherent=new.qa_coherence(*dimargs) if cert else None
            assert old_policy==new_policy==want,(name,old_policy,new_policy,want)
            if cert:
                assert want==int(bool(coherent)),(name,cert,coherent,want)
            assert before==tuple(hashlib.sha256(a).hexdigest() for a in saved),name
            rows.append(dict(name=name,family=family,m=m,n=n,k=k,lda=la,ldb=lb,a_offset=ao,b_offset=bo,
                             expected=want,baseline=old_policy,candidate=new_policy,certificate=cert,
                             a_upper_hex=hex(ub_a),b_upper_hex=hex(ub_b),a_axis_witnesses=witness_a,
                             b_axis_witnesses=witness_b,first_certificate_stage=stage,
                             coherent_after_certificate='' if coherent is None else coherent))
        for idx,(name,family,a,b,declared) in enumerate(itertools.chain(hand_cases(),random_cases())):
            want=expected_policy(a,b)
            assert declared is None or want==declared,(name,want,declared)
            sa,ap,la,ao=parent(a,idx+1)
            sb,bp,lb,bo=parent(b,idx+19)
            check(name,family,len(a),len(b[0]),len(b),ap,la,bp,lb,ao,bo,want,(sa,sb))
        # A full-size eligible multiplication's inputs, without executing or timing a GEMM.
        m,n,k=1537,2306,2306
        arow=array('f',[1.])*k
        aa=arow*m
        bb=array('f');bpos=array('f',[1.])*n;bneg=array('f',[-1.])*n
        for p in range(k):bb.extend(bneg if (p*64//k)%2 else bpos)
        ap=ct.cast(aa.buffer_info()[0],P);bp=ct.cast(bb.buffer_info()[0],P)
        check('full-production-bounds','full-scale-policy',m,n,k,ap,k,bp,n,0,0,0,(aa,bb))
        aa[k+19]=1.5
        check('full-late-exact-fallback','full-scale-policy',m,n,k,ap,k,bp,n,0,0,0,(aa,bb))
        for n,k in ((3843,16),(16,3843),(3843,3843)):
            # The predicate checks its own scratch dimensions before accessing either pointer.
            assert old.qa_policy(1,n,k,None,k,None,n)==new.qa_policy(1,n,k,None,k,None,n)==2
        shapes=[(1536,2304,2304),(1537,2306,2306),(2561,3842,3842),(1535,2304,2304),
                (2562,3842,3842),(2561,3843,3842),(2561,3842,3843),(1921,2306,2306),
                (1922,2306,2306),(0,0,0),(-1,2304,2304),(2**31-1,2**31-1,2**31-1)]
        for m,n,k in shapes:
            d=m//2,n//3,k//3
            want=int(min(d)>=768 and max(d)<=1280 and 4*max(d)<=5*min(d))
            assert old.qa_shape(m,n,k)==new.qa_shape(m,n,k)==want,(m,n,k,want)
        stats=Counter((r['certificate'],r['candidate']) for r in rows)
        # Both certified outcomes, and every exact fallback outcome, must be covered.
        assert all(stats[pair] for pair in ((1,0),(1,1),(0,0),(0,1),(0,2))),stats
        index={r['name']:r for r in rows}
        stages=Counter(r['first_certificate_stage'] for r in rows)
        assert set(stages)=={'A-bound','B-bound','A-axis-witnesses','B-axis-witnesses',
                             'full-range-or-depth','certified'},stages
        for n0 in ('A-unsampled-0x1.8000000000000p+0','B-unsampled-0x1.8000000000000p+0',
                   'A-loose-upper','B-loose-upper','A-late-column-witness','B-late-row-witness',
                   'full-late-exact-fallback'):
            assert index[n0]['certificate']==0,n0
        with (args.output/'decisions.csv').open('w',newline='') as fd:
            writer=csv.DictWriter(fd,fieldnames=rows[0],lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
        report=dict(base=BASE,original_source_sha256=sha(base/'src/cob_hk15.inc'),
                    candidate_source_sha256=sha(candidate/'src/cob_hk15.inc'),
                    compiler=subprocess.check_output([args.cc,'--version'],text=True).splitlines()[0],
                    architecture=platform.machine(),os=platform.system(),flags=flags,base_case_count=len(rows),
                    receipt_sha256=sha(args.output/'decisions.csv'),production_shape_checks=len(shapes),
                    rejected_size_before_access=3,per_path={f'certificate={c}.policy={p}':v for (c,p),v in sorted(stats.items())},
                    per_family=dict(sorted(Counter(r['family'] for r in rows).items())),
                    certificate_stages=dict(sorted(stages.items())),
                    source_file_sha256={f:sha(Path(__file__).with_name(f)) for f in ('check.py','random_cases.py')})
        (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
if __name__=='__main__':main()
