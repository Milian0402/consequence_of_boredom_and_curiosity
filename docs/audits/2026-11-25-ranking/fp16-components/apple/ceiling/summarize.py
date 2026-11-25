"""Check the matched instruction schedule and report its observed arithmetic rates."""
from pathlib import Path
import csv
import hashlib
import json
import re
from statistics import median

base=Path(__file__).resolve().parent
rows=list(csv.DictReader((base/'trials.csv').open()))
assert [(int(r['round']),int(r['position']),r['mode']) for r in rows] == [
    (i,j, 'fp16' if (i+j)%2 else 'fp32') for i in range(6) for j in range(2)]
assert len({(r['chips'],r['lanes'],r['fmopa']) for r in rows})==1
lanes=int(rows[0]['lanes'])
assert all(int(r['chips'])*8192==int(r['fmopa']) for r in rows)
def digest(file):
    return hashlib.sha256((base/file).read_bytes()).hexdigest()
asm=(base/'disassembly.txt').read_text()
for source in ('h','s'):
    tiles=re.findall(r'\bfmopa\s+za([0-3])\.s, p0/m, p0/m, z[0-3]\.'+source+r', z[0-3]\.'+source,asm)
    assert list(map(int,tiles))==[0,1,2,3]*8
pairs=[]
for i in range(6):
    times={r['mode']:float(r['seconds']) for r in rows if int(r['round'])==i}
    pairs.append(2*times['fp32']/times['fp16'])
s={'host':(base/'platform.txt').read_text().strip(),'work_units':'one FP32 FMA equals two arithmetic operations',
   'lanes_fp32':lanes,'fmopa_per_observation':int(rows[0]['fmopa']),
   'fp16_over_fp32_gflops_ratios':pairs,'paired_median_fp16_over_fp32':median(pairs),
   'fp16_over_fp32_min':min(pairs),'fp16_over_fp32_max':max(pairs),
   'median_gflops':{m:median(float(r['gflops']) for r in rows if r['mode']==m) for m in ('fp16','fp32')},
   'median_seconds':{m:median(float(r['seconds']) for r in rows if r['mode']==m) for m in ('fp16','fp32')},
   'checksums':{f:digest(f) for f in ('rings.S','trial.c','disassembly.txt','trials.csv','platform.txt')}}
print(json.dumps(s,indent=2))
