import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

root=Path(__file__).resolve().parent
def score():
    groups=defaultdict(dict)
    matrices={}
    with (root/'times.csv').open() as f:
        for row in csv.DictReader(f):
            shape=int(row['id'])
            dims=tuple(int(row[x]) for x in ('m','n','k'))
            assert matrices.setdefault(shape,dims)==dims
            assert int(row['verified_cells'])==dims[0]*dims[1] and int(row['reference_spots'])==96
            key=(shape,int(row['seed']),int(row['block']))
            method=row['method']
            assert method not in groups[key] and float(row['seconds'])>0
            groups[key][method]=(float(row['seconds']),int(row['position']))
    assert len(groups)==128
    shapes={}
    for shape,dims in sorted(matrices.items()):
        ratios=defaultdict(list)
        for (idx,seed,rep),g in groups.items():
            if idx!=shape:continue
            assert set(g)==set('OABF') and {val[1] for val in g.values()}==set(range(4))
            a,b,o,new=(g[key][0] for key in 'ABOF')
            for name,val in [('vs_v3',math.sqrt(a*b)/new),('vs_original',o/new),('v3_null',a/b)]:
                ratios[name].append(val)
        assert all(len(values)==16 for values in ratios.values())
        shapes[shape]={'m_n_k':dims,**{name:statistics.median(vals) for name,vals in ratios.items()}}
    result={'shapes':shapes,'cohorts':{}}
    for name,keys in [('previous',range(4)),('fresh',range(4,8))]:
        result['cohorts'][name]={r:statistics.geometric_mean(shapes[x][r] for x in keys) for r in ('vs_v3','vs_original','v3_null')}
    result['thresholds_met']=all(r['vs_v3']>=1.03 and r['vs_original']>=.95 for r in result['cohorts'].values()) and all(v['vs_v3']>=.97 for v in shapes.values())
    return result
if __name__=='__main__':print(json.dumps(score(),indent=2))
