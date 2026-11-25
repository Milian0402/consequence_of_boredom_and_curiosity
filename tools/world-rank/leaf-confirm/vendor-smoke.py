"""Execute the four installed release worker pairs in fresh pinned processes on one cropped input."""
import csv
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ragged-hk'))
from vendor_route import validate_release

FIELDS = ('m','n','k','lda','ldb','ldc','ar','ac','br','bc','cr','cc')


def lines(args, cpu, reference, create=False):
    env = os.environ.copy()
    for name in ('RAGGED_REFERENCE','RAGGED_WRITE_REFERENCE'):
        env.pop(name,None)
    env['RAGGED_WRITE_REFERENCE' if create else 'RAGGED_REFERENCE'] = str(reference)
    for name in ('OPENBLAS_NUM_THREADS','GOTO_NUM_THREADS','OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS'):
        env[name] = '1'
    env['MKL_DYNAMIC'] = 'FALSE'
    return subprocess.check_output(['taskset','-c',str(cpu),*map(str,args)],env=env,text=True)


def single(log, tag):
    rows = [s.split(',')[1:] for s in log.splitlines() if s.startswith(tag+',')]
    if len(rows)!=1:
        raise ValueError(f'Expected one {tag} record, found {len(rows)}')
    return rows[0]


def smoke(inventory, oracle, work, out, cpu, geometry=(31,37,19,29,53,49,3,5,2,4,1,6)):
    with inventory.open() as f:
        methods = list(csv.DictReader(f))
    vendor = 'armpl' if os.uname().machine=='aarch64' else 'mkl'
    expected = {f'release-{p}-{mode}' for p in ('openblas',vendor) for mode in ('two','aligned-two')}
    if len(methods)!=4 or {r['method'] for r in methods}!=expected:
        raise ValueError('Missing released ordinary/trace worker')
    m,n,k,lda,ldb,ldc,ar,ac,br,bc,cr,cc = geometry
    layout = dict(zip(FIELDS,geometry))
    page = os.sysconf('SC_PAGE_SIZE')
    padded = (4*(cr+m+2)*ldc + page-1)//page*page//4 - m*n
    residues = [(4*(r*ld+c))%64 for r,c,ld in [(ar,ac,lda),(br,bc,ldb),(cr,cc,ldc)]]
    ref = work/'vendor-preflight.fp64'
    args = ['random',271713,0,*geometry]
    golden = single(lines([methods[0]['worker'],oracle,*args],cpu,ref,True),'GOLD')
    if int(golden[2])!=m*n or int(golden[3])!=23 or not 0<=float(golden[4])<=1e-9 or [int(x) for x in golden[5:]]!=[cpu,1]:
        raise ValueError('Full serial DGEMM oracle or its scalar FP64 spot checks failed')
    proof = []
    for r in methods:
        outputs = []
        for kind in ('worker','trace_worker'):
            log = lines([r[kind],'public',*args],cpu,ref)
            measured = single(log,'VIEW')
            hashes = single(log,'RAGGED')
            if (list(map(int,measured[:7]))!=[m*n,padded,*residues,cpu,1] or
                list(map(int,measured[10:]))!=[lda,ldb,ldc] or not 0<=float(measured[7])<=0.002 or
                hashes[:2]!=golden[:2] or len(hashes[2])!=64):
                raise ValueError(f'{r["method"]} {kind}: full output, parent guards, leading dimensions or thread policy failed')
            outputs.append((hashes[2],float(measured[7])))
            if kind=='trace_worker':
                route=validate_release(layout,log,method=r['method'])
        if outputs[0]!=outputs[1]:
            raise ValueError(f'{r["method"]}: ordinary and trace outputs differ')
        proof.append(dict(method=r['method'],**layout,cpu=cpu,threads=1,checked=m*n,guards=padded,
                          a_mod64=residues[0],b_mod64=residues[1],c_mod64=residues[2],max_error=outputs[0][1],
                          output_sha256=outputs[0][0],ordinary_trace_equal=1,serial_provider=route['provider_identity'],
                          route=route['route'],vendor_calls=route['calls'],core_leaves=route['core_calls']))
    with out.open('w',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=list(proof[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(proof)


if __name__=='__main__':
    inventory,oracle,work,out,cpu=sys.argv[1:]
    smoke(Path(inventory),Path(oracle),Path(work),Path(out)/'vendor-release-smoke.csv',int(cpu))
