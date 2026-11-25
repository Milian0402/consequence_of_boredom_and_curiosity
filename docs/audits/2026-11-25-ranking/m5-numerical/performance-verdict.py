"""Apply the performance gates in protocol.md to the two independent timing summaries."""
from pathlib import Path
import csv
import statistics
import sys

h=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def rows(name):
    with (h/name).open() as f:
        return list(csv.DictReader(f))
def shape(r):
    return tuple(r[x] for x in ['m','n','k','pad'])
public=rows('summary.csv')
direct={shape(r):r for r in rows('prepared-summary.csv') if r['buffers']=='4'}
anchor,=[r for r in public if r['class']=='anchor']
hold=[r for r in public if r['class']=='held-out']
assert len(hold)==8 and len(direct)==len(public)
def ratios(r):
    return tuple(map(float,[r['release_over_candidate'],r['paired_vs_peer'],direct[shape(r)]['paired_vs_direct']]))
a=ratios(anchor)
effects=[ratios(r) for r in hold]
geos=tuple(map(statistics.geometric_mean,zip(*effects)))
wins=sum(pub>1 and prep>1 for incumbent,pub,prep in effects)
minimum=min(r[0] for r in effects)
ok=min(a)>1 and min(geos)>1.05 and wins>=6 and minimum>=1/1.05
print('Anchor paired ratios vs incumbent, best public, and prepared:',*map(lambda x:f'{x:.5f}',a))
print('Held-out geometric means in the same order:',*map(lambda x:f'{x:.5f}',geos))
print(f'Held-out wins against both peers: {wins}/{len(hold)}; minimum speedup vs incumbent: {minimum:.5f}')
print('Performance gate: '+('passed' if ok else 'failed'))
raise SystemExit(0 if ok else 1)
