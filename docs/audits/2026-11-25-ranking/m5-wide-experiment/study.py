from pathlib import Path
import subprocess as sp,csv,os,sys,hashlib
h=Path(__file__).resolve().parent;r=h.parents[1];w=r.parent
methods={'release':h/'release.dylib','candidate':h/'candidate.dylib','Accelerate':w/'ranking/apple-single.dylib','OpenBLAS':r/'openblas/openblas-mac.dylib','ArmPL':r/'armpl-mac/armpl-mtmm.dylib','MaxMulSK':w/'ranking/maxmulsk.dylib','MpGEMM':w/'cob/experiments-local/mpgemm/mpgemm.dylib','prepared-LIBXSMM':'prepared'}
assert all(p=='prepared' or p.is_file() for p in methods.values())
cases=list(csv.DictReader((h/'panel.csv').open()))
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
with (h/'participant-hashes.txt').open('w') as f:
 for name,path in methods.items():
  if path!='prepared':print(hashlib.sha256(path.read_bytes()).hexdigest(),name,file=f)
 print(hashlib.sha256((r/'libxsmm/upstream/lib/libxsmm.a').read_bytes()).hexdigest(),'prepared-LIBXSMM archive',file=f)
with (h/'times.csv').open('w') as out,(h/'times.log').open('w') as log:
 csv.writer(out).writerow('m n k pad seed block round position method calls first_seconds warm_seconds max_abs_error checked_cells threads_before threads_after process_to_caller_cpu code_bytes jit_setup_seconds'.split());out.flush()
 for block,seed in enumerate((856103219,1743857903),1):
  for row in (cases if block==1 else reversed(cases)):
   dims=[row[q] for q in ('m','n','k','pad')]
   cmd=[h/'bench',*dims,str(seed),str(block)]
   for name,path in methods.items():
    if int(dims[3]) and name in ('MaxMulSK','MpGEMM'):continue
    cmd.extend((name,path))
   print(block,row['family'],file=log,flush=True)
   sp.run(cmd,check=True,env=env,stdout=out,stderr=log)
