from pathlib import Path
from collections import Counter
import sys,csv,statistics as st,json,itertools,math
h=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda f:list(csv.DictReader(f.open()))
panel=read(h.parent/'panel/panel.csv');rows=read(h/'null.csv');assert len(rows)==504 and len(panel)==14
by=[]
for s in panel:
 t=[r for r in rows if all(r[k]==s[k] for k in ('m','n','k','pad'))];assert len(t)==36
 assert all(r['threads_before']==r['threads_after']=='1' and float(r['max_abs_error'])<.002 for r in t)
 for b in ['1','2']:
  orders=[tuple(x['method'] for x in sorted((r for r in t if r['block']==b and int(r['round'])==i),key=lambda r:int(r['position']))) for i in range(1,7)]
  assert sorted(orders)==sorted(itertools.permutations('ABD'))
  pos=Counter((r['method'],r['position']) for r in t if r['block']==b);assert len(pos)==9 and set(pos.values())=={2}
 times={n:{(r['block'],int(r['round'])):float(r['warm_seconds']) for r in t if r['method']==n} for n in 'ABD'}
 pair=lambda a,b,block=None:[v/times[b][k] for k,v in times[a].items() if block is None or k[0]==str(block)]
 aa=pair('A','B');x=[math.sqrt(v*times['B'][i])/times['D'][i] for i,v in times['A'].items()]
 p=st.median(x);z=st.median(aa)
 both=[st.median(pair(i,'D',b)) for b in (1,2) for i in 'AB']
 q1,_,q3=st.quantiles(aa,n=4,method='inclusive')
 by.append(dict(s,AA=z,AA_min=min(aa),AA_q1=q1,AA_q3=q3,AA_max=max(aa),AD=st.median(pair('A','D')),BD=st.median(pair('B','D')),gAB_over_D=p,s1_A_over_D=both[0],s1_B_over_D=both[1],s2_A_over_D=both[2],s2_B_over_D=both[3],null_alert=abs(z-1)>.03,corroborated=min(both)>1 and p-1>max(.03,abs(z-1))))
with (h/'null-shapes.csv').open('w') as f:
 w=csv.DictWriter(f,by[0]);w.writeheader();w.writerows(by)
fresh=[r for r in by if r['family'].startswith('fresh-')];ctl=[r for r in by if r['family'].startswith('control-')]
d={'null_geomean':st.geometric_mean(r['AA'] for r in by),'null_median_range':[min(r['AA'] for r in by),max(r['AA'] for r in by)],'null_round_range':[min(r['AA_min'] for r in by),max(r['AA_max'] for r in by)],'null_shape_alerts':[r['family'] for r in by if r['null_alert']],'fresh_corrected_gm':st.geometric_mean(r['gAB_over_D'] for r in fresh),'fresh_corrected_range':[min(r['gAB_over_D'] for r in fresh),max(r['gAB_over_D'] for r in fresh)],'corroborated_fresh':[r['family'] for r in fresh if r['corroborated']],'corroborated_bypasses':[r['family'] for r in ctl if r['corroborated']],'outside_m':next(r for r in ctl if r['family']=='control-outside-m'),'cells':sum(int(r['checked_cells']) for r in rows)+sum(int(r['checked_cells']) for r in rows if r['round']=='1'),'max_abs':max(float(r['max_abs_error']) for r in rows)}
(h/'null-summary.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
