from pathlib import Path
import argparse,csv,subprocess as sp,os,hashlib,shutil
p=argparse.ArgumentParser(description='Rerun both compact adapters and three direct methods on the frozen M5 panel.')
for key in ['installed_cob','serial_accelerate','maxmulsk','mpgemm','libxsmm_checkout','new_output_dir']:p.add_argument(key,type=Path)
args=p.parse_args();source=Path(__file__).resolve().parent;h=args.new_output_dir.resolve();h.mkdir(parents=True,exist_ok=False)
old=h
for file in ('panel.csv','compact-adapter.c','guards.c','first.c','bench.c','common-bench.c','mtmm.h'):
 shutil.copyfile(source/file,h/file)
for n in ('first','bench'):
 sp.run(['cc','-O3','-std=c11','-I'+str(args.libxsmm_checkout.resolve()/'include'),h/(n+'.c'),args.libxsmm_checkout.resolve()/'lib/libxsmm.a','-o',h/n],check=True)
rows=list(csv.DictReader((h/'panel.csv').open()));assert len(rows)==6
methods={'candidate':args.installed_cob.resolve(),'Accelerate':args.serial_accelerate.resolve(),'prepared-LIBXSMM':'prepared','MaxMulSK-adapted':h/'MaxMulSK-adapted.dylib','MpGEMM-adapted':h/'MpGEMM-adapted.dylib'}
backends={'MaxMulSK':args.maxmulsk.resolve(),'MpGEMM':args.mpgemm.resolve()}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
for k,v in backends.items():
 var='COMPACT_'+k.upper();env[var]=str(v)
 sp.run(['cc','-O3','-std=c11','-dynamiclib','-DCOMPACT_BACKEND_ENV="'+var+'"','-I'+str(old),h/'compact-adapter.c','-o',methods[k+'-adapted']],check=True)
with (h/'hashes.txt').open('w') as f:
 for k,p in dict(backends,**methods).items():
  if p!='prepared':print(hashlib.sha256(p.read_bytes()).hexdigest(),k,file=f)
for first in (False,True):
 name='first' if first else 'times';exe=old/('first' if first else 'bench')
 with (h/(name+'.csv')).open('w') as out,(h/(name+'.log')).open('w') as log:
  csv.writer(out).writerow(next(csv.reader((source/(name+'.csv')).open())));out.flush()
  for block,seed in [(1,318741563),(2,1557601921)]:
   for s in (rows if block==1 else reversed(rows)):
    common=[exe,*[s[q] for q in ('m','n','k','pad')],str(seed),str(block)]
    if first:
     for k,v in (methods.items() if block==1 else reversed(methods.items())):
      sp.run([*common,k,v],check=True,env=dict(env,FRONTIER_FIRST_ONLY='1'),stdout=out,stderr=log)
    else:sp.run([*common,*[v for kv in methods.items() for v in kv]],check=True,env=env,stdout=out,stderr=log)
sp.run(['cc','-O3','-std=c11',h/'guards.c','-o',h/'guards'],check=True)
for k in ('candidate','MaxMulSK-adapted','MpGEMM-adapted'):
 with (h/('guards-'+k+'.csv')).open('w') as f:sp.run([h/'guards',methods[k]],check=True,env=env,stdout=f)
