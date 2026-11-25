from pathlib import Path
import csv,math,statistics as st,json,sys,itertools
h=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda name:list(csv.DictReader((h/'results'/(name+'.csv')).open()))
raw=read('samples');acc=read('accuracy');pool=read('pool');fault=read('faults')
assert len(raw)==648 and len(acc)==18 and len(pool)==6 and len(fault)==12
assert all(int(r['slots'])==int(r['unique_a'])==int(r['unique_b'])==int(r['unique_c']) and r['pages']==r['resident_pre']==r['resident_post']==r['epoch_min_resident'] and int(r['logical_bytes'])>=134217728 and r['same_alias']==r['readonly_ab']==r['shared_method_output']==r['threads']=='1' for r in pool)
assert all(float(x['max_abs'])<=.002 and x['guards_ok']=='1' and int(x['checked'])==int(x['slots'])*int(x['n'])*int(x['m']) for x in acc)
assert all(int(x['major_faults'])==0 and x['locked']=='1' for x in fault)
cases=list(csv.DictReader((h/'panel.csv').open()));results=[]
for c in cases:
 for mode in ('hot','rotate'):
  t=[x for x in raw if x['family']==c['family'] and x['mode']==mode]
  assert len(t)==108 and all(x['threads_before']==x['threads_after']==x['guards_ok']=='1' and int(x['calls'])==int(x['slots']) for x in t)
  for block in (1,2):
   for cycle in range(3):
    orders=[tuple(x['method'] for x in sorted((v for v in t if int(v['block'])==block and int(v['round'])==6*cycle+j),key=lambda a:int(a['position']))) for j in range(1,7)];assert sorted(orders)==sorted(itertools.permutations('ABD'))
  w={k:{(int(x['block']),int(x['round'])):int(x['wall_ns'])/int(x['calls']) for x in t if x['method']==k} for k in 'ABD'}
  u={k:{(int(x['block']),int(x['round'])):int(x['caller_ns'])/int(x['calls']) for x in t if x['method']==k} for k in 'ABD'}
  corr=lambda d,seed=None:st.median(math.sqrt(d['A'][i]*d['B'][i])/v for i,v in d['D'].items() if seed is None or i[0]==seed)
  null=st.median(w['A'][i]/v for i,v in w['B'].items())
  results.append(dict(c,mode=mode,slots=t[0]['slots'],A_over_B=null,incumbent_over_candidate=corr(w),caller_incumbent_over_candidate=corr(u),seed1=corr(w,1),seed2=corr(w,2),candidate_us=st.median(w['D'].values())/1000,incumbent_us=st.median([math.sqrt(w['A'][i]*w['B'][i]) for i in w['D']])/1000))
with (h/'scores.csv').open('w') as f:
 w=csv.DictWriter(f,results[0]);w.writeheader();w.writerows(results)
g={}
for mode in ('hot','rotate'):
 a=[x for x in results if x['mode']==mode];g[mode]={'gm':st.geometric_mean(x['incumbent_over_candidate'] for x in a),'caller_gm':st.geometric_mean(x['caller_incumbent_over_candidate'] for x in a),'wins':sum(x['incumbent_over_candidate']>1 for x in a),'seedwins':[sum(x[s]>1 for x in a) for s in ('seed1','seed2')],'minimum':min(x['incumbent_over_candidate'] for x in a),'null_alerts':sum(abs(x['A_over_B']-1)>.03 for x in a)}
v={'scores':g,'promote_to_holdouts':g['rotate']['gm']>=1.06 and min(g['rotate']['seedwins'])>=2 and g['hot']['gm']>=.985 and g['hot']['minimum']>=.95 and max(g[k]['null_alerts'] for k in g)<=1,'fp64_cells':sum(int(x['checked']) for x in acc),'max_abs':max(float(x['max_abs']) for x in acc),'timings':len(raw),'timed_calls':sum(int(x['calls']) for x in raw)}
(h/'scores.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
for x in results:print(x['family'],x['mode'],*[round(x[k],3) for k in ('A_over_B','incumbent_over_candidate','seed1','seed2')])
