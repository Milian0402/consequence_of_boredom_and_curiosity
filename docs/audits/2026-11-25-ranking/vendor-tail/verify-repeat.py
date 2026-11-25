"""Reproduce the two sets of qualifications and check every cross-attempt ratio."""
import csv
import json
import math
from pathlib import Path
from statistics import geometric_mean, median
import subprocess
import sys
import tempfile

root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
rows=read(root/'cross-attempt-shapes.csv')
assert len(rows)==216
index={}
competitors={'openblas','matched-openblas','mkl','mkl-jit','matched-mkl','aocl','blasfeo','onednn-oneshot','onednn-cached-plan'}
with tempfile.TemporaryDirectory() as tmp:
    for attempt,side in [(1,'first'),(2,'second')]:
        data=root/f'attempt-{attempt}'
        out=Path(tmp)/side
        subprocess.run([sys.executable,'-B',root/'audit.py',data,out],check=True)
        for name in ('independent-score.json','independent-per-shape.csv'):
            assert (data/name).read_bytes()==(out/name).read_bytes()
        times={}
        for r in read(data/'timings.csv'):
            times.setdefault((r['family'],r['implementation']),{})[(int(r['block']),int(r['round']))]=float(r['seconds'])
        for row in rows:
            f,name,kind=row['family'],row['method'],row['comparison']
            peer=row[side+'_peer']
            if kind=='strongest_independent':
                assert peer==min(competitors,key=lambda p:(median(times[f,p].values()),p))
            elif kind=='matched_direct':
                assert peer=='matched-'+name.split('-')[2]
            elif kind=='matched_full':
                assert peer==name[:-5]+'full'
            else:
                assert peer=={'standalone_release':'stream-incumbent','stream_b':'stream-native'}[kind]
            key=(name,f,kind)
            ratio=median(times[f,peer][k]/t for k,t in times[f,name].items())
            assert math.isclose(ratio,float(row[side+'_speedup']),rel_tol=1e-13)
            index.setdefault(key,{})[side]=ratio
            if side=='second':
                d=index[key]
                assert int(row['wins_both_attempts'])==int(min(d.values())>1)
                assert int(row['losses_both_attempts'])==int(max(d.values())<1)
cohorts=json.loads((root/'cross-attempt.json').read_text())['cohorts']
for group in read(root/'cross-attempt-groups.csv'):
    n,k=group['method'],group['comparison']
    values=[index[n,f,k] for f in cohorts[group['cohort']]]
    for side in ['first','second']:
        assert math.isclose(geometric_mean(d[side] for d in values),float(group[side+'_gm']),rel_tol=1e-13)
    assert sum(min(d.values())>1 for d in values)==int(group['wins_both_attempts'])
print('Both raw audits and every within-host ratio and cross-attempt cohort geometric mean agree.')
