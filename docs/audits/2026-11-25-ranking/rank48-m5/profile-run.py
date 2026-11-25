import csv,os,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent
os.environ['RANK_FINITE_LEAF']=str(h/'accelerate.dylib')
src=h/'bench/tools/world-rank/rank48/check.c'
text=src.read_text()
old='    gemm(m, n, k, a, lda, b, ldb, c, ldc);'
assert text.count(old)==1
(h/'profile-check.c').write_text(text.replace('#include "../vendor.h"','#include "vendor.h"').replace(old,'    for(int wave=0;wave<4;++wave) '+old.strip()))
subprocess.run(['clang','-O2','-std=c11','-I'+str(src.parent.parent),str(h/'profile-check.c'),'-o',str(h/'profile-check')],check=True)
p=Path(os.environ.get('RANK_APPLE_RESULT','results-qualified'))
if not p.is_absolute():p=h/p
with (p/'components.csv').open('w',newline='') as f:
 out=csv.writer(f);out.writerow(['series','call','rank','alloc_seconds','input_transforms_seconds','core_leaf_seconds','recombine_seconds','free_and_tails_seconds','wall_seconds'])
 for i,r in enumerate((48,49,49,48)):
  result=subprocess.check_output([h/'profile-check',h/f'profile{r}.dylib',str(r),'2039','2969','3367','11'],text=True)
  *phase,proof=result.splitlines()
  if len(phase)!=4 or not all(z.startswith('PHASE,') for z in phase) or proof.split(',')[-3:]!=['0','1','1']:raise ValueError(result)
  for call,z in enumerate(phase):out.writerow([i,call,*z.split(',')[1:]])
