"""Classify measured provider calls which actually crossed the public CBLAS ABI."""
import csv

# -1 denotes the respective X, Y or product scratch, 0..8 denotes a caller block.
A = (3,-1,1,-1,-1,-1,2,-1,-1,-1,-1,5,-1,-1,-1)
B = (-1,-1,-1,1,-1,2,-1,-1,-1,3,5,-1,7,-1,6)
C = (0,-1,-1,-1,-1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1)
def require(ok, reason):
    if not ok: raise ValueError(reason)

def validate_hk_leaf(layout, stdout, provider, method):
    m,n,k,lda,ldb,ldc=(int(layout[key]) for key in ("m","n","k","lda","ldb","ldc"))
    mh,nh,kh=m//2,n//3,k//3
    eligible=min(mh,nh,kh)>=768 and max(mh,nh,kh)<=1280 and 4*max(mh,nh,kh)<=5*min(mh,nh,kh)
    info=[line for line in stdout.splitlines() if line.startswith('HKLEAF,')]
    require(len(info)==1,'expected one public HK15 observation')
    fields=next(csv.reader(info))
    require(len(fields)==14,'invalid trace width')
    identity=fields[2]
    count,claimed,total,finished,right,bottom,depth,core,ok,work,mode,stored=map(int,[fields[1],*fields[3:]])
    require(count==1 and claimed==eligible and ok==1 and mode==0 and stored==1,'public/fringe witness failed')
    require((provider=='openblas' and identity.startswith('OpenBLAS 0.3.34 ')) or
            (provider=='mkl' and identity=='libmkl_intel_lp64.so.3') or
            (provider=='armpl' and (identity=='libarmpl_lp64.so' or identity.startswith('libarmpl_lp64.so.'))),'unverified serial provider')
    calls=[]
    names='index m n k lda ldb ldc a_block b_block c_block beta rowmajor_nt_alpha1'.split()
    for row in csv.reader(line for line in stdout.splitlines() if line.startswith('HKCALL,')):
        e=dict(zip(names,map(int,row[1:]),strict=True))
        require(e['index']==len(calls) and e['rowmajor_nt_alpha1']==1 and e['beta'] in (0,1),'invalid SGEMM ABI')
        calls.append(e)
    require(len(calls)==total,'missing native provider calls')
    if eligible:
        require(total==15 and work==1,'expected fifteen nonrecursive provider calls in one contiguous workspace')
        for i,e in enumerate(calls):
            require((e['m'],e['n'],e['k'])==(mh,nh,kh) and e['beta']==0,'invalid HK child geometry or beta')
            require((e['a_block'],e['b_block'],e['c_block'])==(A[i],B[i],C[i]),'wrong HK operator or operand pointer')
            require((e['lda'],e['ldb'],e['ldc'])==(lda if A[i]>=0 else kh, ldb if B[i]>=0 else nh,ldc if C[i]>=0 else nh),'incorrect HK child leading dimension')
        require((finished,right,bottom,depth,core)==(int(bool(m%2 or n%3 or k%3)),int(bool(n%3)),m%2,k%3,
                (m-m%2)*(n-n%3) if k%3 else 0),'invalid fringe completion')
    else:
        require((finished,right,bottom,depth,core)==(0,0,0,0,0),'unexpected HK fringe on bypass')
    return dict(qualified=True,route='hk15-direct-blas' if eligible else 'shape-bypass',method=method,provider=provider,
        provider_identity=identity,child_count=15 if eligible else 0,provider_calls=total,fringe_calls=right+bottom,
        workspace_ok=bool(work),fringe_finish=finished,depth_terms=depth,depth_outputs_checked=core,calls=calls)
