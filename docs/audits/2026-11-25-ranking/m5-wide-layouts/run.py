from pathlib import Path
import subprocess as sp,os,csv,sys,json,statistics as st,hashlib,shutil
h=Path(__file__).resolve().parent;rank=h.parents[1];old=rank/'frontier-wave/apple-thin';work=rank.parent
phase=sys.argv[1];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
seeds=(675893197,1382786813) if phase=='train' else (427297853,1670820329)
peers={'Accelerate':work/'ranking/apple-single.dylib','OpenBLAS':rank/'openblas/openblas-mac.dylib','ArmPL':rank/'armpl-mac/armpl-mtmm.dylib','MaxMulSK':work/'ranking/maxmulsk.dylib','MpGEMM':work/'cob/experiments-local/mpgemm/mpgemm.dylib','prepared-LIBXSMM':'prepared'}
base=old/'installed-on.dylib'
if phase=='train':
 cases=[dict(m=m,n=n,k=k,pad='0',family='anchor') for m,n,k in [('24','1280','131'),('39','2304','160'),('55','1536','384')]]
 methods={'incumbent':base,**{k:h/(k+'.dylib') for k in ('direct','vertical','superpanel')},**{k:v for k,v in peers.items() if k in ('Accelerate','prepared-LIBXSMM','MpGEMM')}}
elif phase in ('holdout','first'):
 chosen=(h/'selection.txt').read_text().strip();cases=list(csv.DictReader((h/'panel/panel.csv').open()));methods={'incumbent':base,'candidate':h/(chosen+'.dylib'),**peers}
else:raise ValueError(phase)
if phase=='first':
 exe=old/'first';ref=old/'production-first.csv';env['FRONTIER_FIRST_ONLY']='1'
else:
 exe=h/'bench';ref=old/'production-times.csv';env['LAYOUT_TRAIN']='1' if phase=='train' else '0'
with (h/(phase+'.csv')).open('w') as f,(h/(phase+'.log')).open('w') as log:
 csv.writer(f).writerow(next(csv.reader(ref.open())));f.flush()
 for block,seed in enumerate(seeds,1):
  for d in (cases if block==1 else reversed(cases)):
   dims=[str(d[q]) for q in ('m','n','k','pad')]
   use={name:path for name,path in methods.items() if not (int(dims[-1]) and name in ('MaxMulSK','MpGEMM'))}
   args=[exe,*dims,str(seed),str(block)]
   print(block,d['family'],file=log,flush=True)
   if phase=='first':
    for name,path in (use.items() if block==1 else reversed(use.items())):sp.run([*args,name,path],check=True,env=env,stdout=f,stderr=log)
   else:sp.run([*args,*[v for kv in use.items() for v in kv]],check=True,env=env,stdout=f,stderr=log)
if phase=='train':
 obs=list(csv.DictReader((h/'train.csv').open()));scores={}
 for method in ('direct','vertical','superpanel'):
  pts=[]
  for d in cases:
   allr=[x for x in obs if x['m']==d['m']]
   rt=lambda name:{(x['block'],x['round']):float(x['warm_seconds']) for x in allr if x['method']==name}
   inc=rt('incumbent');cand=rt(method);assert len(inc)==len(cand)==6
   pts.append(st.median(inc[i]/v for i,v in cand.items()))
  scores[method]=dict(warm=pts,gm=st.geometric_mean(pts))
 chosen=max(scores,key=lambda k:scores[k]['gm']);(h/'selection.txt').write_text(chosen+'\n');(h/'training-scores.json').write_text(json.dumps(scores,indent=2)+'\n')
 print(chosen,scores,flush=True)
