"""Post-score balanced warm comparisons, using the same two installed product workers and serial FP64 goldens."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

work, raw = (Path(path).resolve() for path in sys.argv[1:3])
cpu = int(sys.argv[3]); assert os.sched_getaffinity(0) == {cpu}
seeds = (197906831, 884039527)
abi = 'm n k lda ldb ldc ar ac br bc cr cc'.split()
fields = 'family seed round position method checked guards cpu threads a_sha256 b_sha256 c_sha256 max_abs wall_s caller_s passed reason'.split()
out = raw/'guard-pairs'; out.mkdir(exist_ok=True)
read = lambda name: list(csv.DictReader((raw/name).open()))
references = {(r['family'],int(r['seed'])):r for r in read('goldens.csv') if r['mode']=='random'}
panel = [r for r in read('panel.csv') if r['family'].startswith(('fresh-','repeat-'))]
methods = ('ragged-hk15','guarded-hk15')
observations = {}
for shape in panel:
    for seed in seeds:
        name=shape['family']; reference=work/'goldens'/f'{name}-random-{seed}.f64'
        with reference.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==references[name,seed]['reference_sha256']
def require(test,msg):
    if not test: raise ValueError(msg)
def tagged(text,name):
    items = [line.split(',')[1:] for line in text.splitlines() if line.startswith(name+',')]
    require(len(items)==1,name+' evidence');return items[0]
with (out/'timings.csv').open('w') as stream:
    wr=csv.DictWriter(stream,fields,lineterminator='\n');wr.writeheader()
    for block,seed in enumerate(seeds):
        for rnd in range(3):
            for index,shape in enumerate(panel):
                family=shape['family'];base=references[family,seed]
                page=os.sysconf('SC_PAGESIZE')
                m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc=(int(shape[z]) for z in abi)
                guards=((cr+m+2)*lc*4+page-1)//page*(page//4)-m*n
                ordered=methods[::-1] if (index+rnd+block)%2 else methods
                for pos,entry in enumerate(ordered,1):
                    row=dict(family=family,seed=seed,round=rnd+1,position=pos,method=entry,passed=0,reason='')
                    exe=work/entry/'ordinary.worker'
                    ref=work/'goldens'/f'{family}-random-{seed}.f64'
                    try:
                        require(ref.is_file() and ref.stat().st_size==m*n*8,'missing shared FP64 golden')
                        run=subprocess.run([exe,'public','random',str(seed),'1',*(shape[z] for z in abi)],
                                env=dict(os.environ,RAGGED_REFERENCE=str(ref)),text=True,capture_output=True,timeout=150)
                        require(run.returncode==0,'worker exit '+str(run.returncode)+' '+run.stderr[-250:].strip())
                        v=tagged(run.stdout,'VIEW');h=tagged(run.stdout,'RAGGED')
                        require(len(v)==13 and len(h)==5 and (int(v[0]),int(v[1]),int(v[5]),int(v[6]))==(m*n,guards,cpu,1),'guard/thread/CPU mismatch')
                        require(tuple(map(int,v[2:5]))==tuple(4*i%64 for i in (ar*la+ac,br*lb+bc,cr*lc+cc)) and tuple(map(int,v[10:]))==(la,lb,lc),'layout mismatch')
                        err=float(v[7]);wall=float(v[9]);caller=float(h[4])
                        row.update(checked=v[0],guards=v[1],cpu=v[5],threads=v[6],a_sha256=h[0],b_sha256=h[1],c_sha256=h[2],max_abs=err,wall_s=wall,caller_s=caller)
                        require(h[:2]==[base['a_sha256'],base['b_sha256']] and len(h[2])==64 and all(c in '0123456789abcdef' for c in h[2]),'digest mismatch')
                        require(0<=err<=.002 and all(math.isfinite(x) and x>0 for x in (wall,caller)),'accuracy or duration')
                        row['passed']=1
                    except (ValueError,OSError,subprocess.TimeoutExpired) as exc:
                        row['reason']=str(exc)
                    wr.writerow(row);stream.flush()
                    observations[family,seed,rnd,entry]=row
report={}
for shape in panel:
    family=shape['family'];ratios=[];caller_ratios=[];pairs=[]
    for seed in seeds:
        for rnd in range(3):
            a,b=(observations[family,seed,rnd,entry] for entry in methods)
            valid=bool(a['passed'] and b['passed'] and a['c_sha256']==b['c_sha256'])
            ratio=a['wall_s']/b['wall_s'] if valid else None
            pairs.append(dict(seed=seed,round=rnd+1,both_qualified=bool(a['passed'] and b['passed']),bitwise_equal=valid,wall_ratio=ratio))
            if valid:ratios.append(ratio);caller_ratios.append(a['caller_s']/b['caller_s'])
    report[family]=dict(pairs=pairs,qualified=len(ratios)==6,median=statistics.median(ratios) if len(ratios)==6 else None,
        seed_medians=[statistics.median(ratios[:3]),statistics.median(ratios[3:])] if len(ratios)==6 else None,
        median_caller_ratio=statistics.median(caller_ratios) if len(ratios)==6 else None)
cohorts={}
for prefix in ('repeat-','fresh-'):
    rs=[r['median'] for f,r in report.items() if f.startswith(prefix)]
    cohorts[prefix[:-1]]=statistics.geometric_mean(rs) if len(rs)==3 and all(rs) else None
(out/'score.json').write_text(json.dumps(dict(cases=report,cohorts=cohorts),indent=2,allow_nan=False)+'\n')
