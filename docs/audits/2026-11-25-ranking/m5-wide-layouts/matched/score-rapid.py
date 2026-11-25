from pathlib import Path
import sys,csv,statistics as st,json,itertools,math
h=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda f:list(csv.DictReader(f.open()))
panel=read(h.parent/'panel/panel.csv');rows=read(h/'rapid.csv');assert len(rows)==5040
out=[]
for s in panel:
 r=[x for x in rows if all(x[k]==s[k] for k in ('m','n','k','pad'))]
 assert len(r)==360 and all(x['threads_before']==x['threads_after']=='1' for x in r)
 for block in ('1','2'):
  for cycle in range(10):
   permutations=[tuple(x['method'] for x in sorted([v for v in r if v['block']==block and int(v['round'])==cycle*6+j+1],key=lambda a:int(a['position']))) for j in range(6)]
   assert sorted(permutations)==sorted(itertools.permutations('ABD'))
 vals={'family':s['family'],**{k:s[k] for k in ('m','n','k','pad')}}
 for clk,t in [('wall','warm_seconds'),('cpu','caller_seconds')]:
  w={n:{(int(x['block']),int(x['round'])):float(x[t]) for x in r if x['method']==n} for n in 'ABD'}
  aa=[w['A'][k]/v for k,v in w['B'].items()]
  corrected={i:math.sqrt(w['A'][i]*w['B'][i])/v for i,v in w['D'].items()}
  by=[st.median(v for (b,_),v in corrected.items() if b==block) for block in (1,2)]
  vals[clk+'_AA']=st.median(aa);vals[clk+'_AAgm']=st.geometric_mean(aa);vals[clk+'_corrected']=st.median(corrected.values());vals[clk+'_s1']=by[0];vals[clk+'_s2']=by[1]
 caller_share=[float(x['caller_seconds'])/float(x['warm_seconds']) for x in r]
 vals['caller_wall_median']=st.median(caller_share);vals['caller_wall_p05']=st.quantiles(caller_share,n=20,method='inclusive')[0]
 out.append(vals)
with (h/'rapid-shapes.csv').open('w') as f:
 w=csv.DictWriter(f,out[0]);w.writeheader();w.writerows(out)
fresh=[x for x in out if x['family'].startswith('fresh-')]
j={'observations':len(rows),'gm_corrected_wall':st.geometric_mean(x['wall_corrected'] for x in fresh),'gm_corrected_cpu':st.geometric_mean(x['cpu_corrected'] for x in fresh),'wall_AA_range':[min(x['wall_AA'] for x in out),max(x['wall_AA'] for x in out)],'cpu_AA_range':[min(x['cpu_AA'] for x in out),max(x['cpu_AA'] for x in out)],'AA_alerts':[x['family'] for x in out if abs(x['wall_AA']-1)>.03],'wins_over_3pct_both_seeds':[x['family'] for x in fresh if min(x['wall_s1'],x['wall_s2'])>1.03],'wall_corrected_range':[min(x['wall_corrected'] for x in fresh),max(x['wall_corrected'] for x in fresh)],'max_abs_error':max(float(x['max_abs_error']) for x in rows),'fp64_checked':sum(int(x['checked_cells']) for x in rows)+sum(int(x['m'])*int(x['n']) for x in rows if x['round']=='1')}
(h/'rapid-summary.json').write_text(json.dumps(j,indent=2)+'\n');print(j)
for v in out:print(v['family'],*[round(v[k],3) for k in ('wall_AA','wall_corrected','wall_s1','wall_s2')])
