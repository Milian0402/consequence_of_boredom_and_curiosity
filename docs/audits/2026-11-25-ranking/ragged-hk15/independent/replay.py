"""Independent raw-data replay of the registered ragged HK15 comparison."""
import argparse,csv,hashlib,json,math,re,statistics,zipfile
from collections import Counter,defaultdict
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('raw',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--original',type=Path);ap.add_argument('--receipt',type=Path)
args=ap.parse_args();root=args.raw;med=statistics.median;gm=statistics.geometric_mean
read=lambda name:list(csv.DictReader((root/name).open()))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if args.original:
 proof=json.loads(args.receipt.read_text());assert proof['sha']=='37ead44b9f3a144dada8a92b17bd447b138e6689' and proof['run']==45709409988 and proof['job_conclusion']=='success'
 assert sha(args.original)==proof['artifact_sha256'][7:]
 with zipfile.ZipFile(args.original) as z:
  assert z.testzip() is None
  for item in z.infolist():
   if not item.is_dir():assert z.read(item)==(root/item.filename).read_bytes(), item.filename
else:proof=None
assert sha(root/'panel.csv')=='6d9d898375f2f57eea76ae42d6ec04b0792d5440967da260b52c47b59c77dc42'
assert sha(root/'protocol.md')=='51ac3e957ad14d06eadcbea25ba17a60c75d5a237e2c2c64cf8d951ceacc5460'
assert (root/'production-sha.txt').read_text().splitlines()[0]=='22d925ed5a31f7246eb2269675e9a3e85d502410'
assert (root/'benchmark-sha.txt').read_text().strip()=='37ead44b9f3a144dada8a92b17bd447b138e6689'
architecture=dict((k.strip(),v.strip()) for k,v in (s.split(':',1) for s in (root/'lscpu.txt').read_text().splitlines() if ':' in s))
cpu=int((root/'pinned-cpu.txt').read_text());page=int((root/'page-bytes.txt').read_text())
registry=read('sources/methods.csv')
methods=[r['method'] for r in registry];roles={r['method']:r['role'] for r in registry}
assert len(methods)==len(set(methods)) and roles['ragged-hk15']=='candidate' and roles['released-hk15']=='baseline'
cand='ragged-hk15';base='released-hk15';seeds=(154983127,720184093)
fixtures={('random',seeds[0]),('random',seeds[1]),('cancel',seeds[0]),('scaled',seeds[1])}
pairs=[(s,r) for s in seeds for r in (1,2,3)]
cases=read('panel.csv');timings=read('timings.csv');accuracy=read('accuracy.csv');golds=read('goldens.csv');routes=read('routes.csv')
qual={(r['family'],r['method']):r['qualified']=='1' for r in read('qualified.csv')}
assert len(qual)==len(cases)*len(methods) and len(golds)==4*len(cases)
gold={(r['family'],r['mode'],int(r['seed'])):r for r in golds}
role_tiers={'standalone':{'baseline'},'independent_oneshot':{'independent-oneshot'},'vendor_assisted':{'vendor-assisted'},
            'independent_any':{'independent-oneshot','independent-retained'},'whole_any':set(roles.values())}
hosted=json.loads((root/'score.json').read_text())
summary={}; max_err=defaultdict(float);checked_cells={'accuracy':0,'warm':0}
for index,p in enumerate(cases):
 f=p['family'];M,N,K,la,lb,lc,ar,ac,br,bc,cr,cc=(int(p[x]) for x in 'm n k lda ldb ldc ar ac br bc cr cc'.split())
 mods=[4*(a*b+c)%64 for a,b,c in ((ar,la,ac),(br,lb,bc),(cr,lc,cc))]
 guards=((cr+M+2)*lc*4+page-1)//page*page//4-M*N
 rawtimes=defaultdict(dict);rawacc=defaultdict(dict)
 for mn,rs,table in [('accuracy',accuracy,rawacc),('warm',timings,rawtimes)]:
  for r in rs:
   if r['family']!=f:continue
   m=r['method'];s=int(r['seed']);j=int(r['round'])
   key=(r['mode'],s) if mn=='accuracy' else (s,j)
   assert key not in table[m]
   table[m][key]=r
   if r['passed']!='1':
    assert not qual[f,m];continue
   assert r['role']==roles[m] and all(r[k]==v for k,v in p.items())
   assert int(r['checked'])==M*N and int(r['cpu'])==cpu and int(r['threads'])==1 and int(r['guards'])==guards
   assert [int(r[x]) for x in ('a_mod64','b_mod64','c_mod64')]==mods
   g=gold[f,r['mode'],s]
   assert int(g['bytes'])==8*M*N and int(g['scalar_spots'])==23 and int(g['checked'])==M*N and float(g['max_spot_error'])<=1e-9
   assert all(r[z]==g[z] for z in ('a_sha256','b_sha256')) and re.fullmatch('[a-f0-9]{64}',r['c_sha256'])
   assert 0<=float(r['max_abs'])<=.002
   if m.startswith('onednn'):
    assert r['implementation'] and not r['implementation'].lower().startswith('ref') and int(r['cached'])==int(m.endswith('cached-plan'))
   max_err[m]=max(float(r['max_abs']),max_err[m]);checked_cells[mn]+=M*N
 eligible=[m for m in methods if qual[f,m]]
 viable=[m for m in methods if m in rawtimes]
 for seed,j in pairs:
  if not viable:continue
  shift=(index+j-1)%len(viable);order=viable[shift:]+viable[:shift]
  if seed==seeds[1]:order.reverse()
  assert [m for m in sorted(viable,key=lambda n:int(rawtimes[n][seed,j]['position']))]==order
 for m in eligible:
  assert set(rawacc[m])==fixtures and set(rawtimes[m])==set(pairs)
 assert {m for m in viable if all(v['passed']=='1' for v in rawtimes[m].values())}==set(eligible)
 for r in routes:
  if r['family']!=f or not qual[f,r['method']]:continue
  name=r['method'];assert r['verified']=='1' and r['ordinary_sha256']==r['trace_sha256']==rawacc[name]['random',seeds[0]]['c_sha256']
  counts=json.loads(r['receipt'])
  if name in (base,cand):
   active=(name==cand or not (M%2 or N%3 or K%3));fringe=active and bool(M%2 or N%3 or K%3)
   expected={'public_calls':1,'args_ok':1,'parents':active,'leaves':15*active,'fused_leaves':15*active,'finishes':fringe,'k_terms':(K%3)*fringe,'right':bool(N%3)*fringe,'bottom':bool(M%2)*fringe,'core_checked':(M-M%2)*(N-N%3)*bool(fringe and K%3)}
   assert all(counts[k]==v for k,v in expected.items())
  if name.startswith('fastmmw-'):
   split=1
   while (min(M,N,K)+split-1)//split>int(name.split('-')[1]):split*=2
   assert counts['core_multiple']==split and counts['leaves']==7**(split.bit_length()-1) and counts['padded']==int(any(dim%split for dim in (M,N,K)))
 times={m:[float(rawtimes[m][pair]['warm_s']) for pair in pairs] for m in eligible}
 medians={m:med(v) for m,v in times.items()}
 rankings={}
 for tier,roleset in role_tiers.items():
  field=[m for m in eligible if m!=cand and roles[m] in roleset]
  if not field or cand not in eligible:continue
  lead=min(field,key=lambda m:(medians[m],m));ratios=[x/y for x,y in zip(times[lead],times[cand])]
  k='standalone' if tier=='standalone' else tier
  rankings[k]=dict(peer=lead,ms=medians[lead]*1000,median=med(ratios),wins=sum(x>1 for x in ratios),seed_medians=list(map(med,(ratios[:3],ratios[3:]))),ratios=ratios)
  assert math.isclose(rankings[k]['median'],hosted['cases'][f]['pairs'][lead]['median'],rel_tol=1e-12)
  if tier!='standalone':
   assert lead==hosted['cases'][f]['tiers'][tier]['fastest_other']
 parity=None
 if index==0:
  parity=all(rawacc[cand][q]['c_sha256']==rawacc[base][q]['c_sha256'] for q in fixtures)
  assert parity
 winner=min(eligible,key=lambda m:(medians[m],m)) if eligible else None
 summary[f]=dict(eligible=len(eligible),candidate_ms=medians.get(cand,0)*1000,tiers=rankings,winner=winner,control_bitwise=parity)
fresh=[f for f in summary if f.startswith('fresh')]
agg={}
for tier in role_tiers:
 ranked=[summary[f]['tiers'][tier] for f in fresh if tier in summary[f]['tiers']]
 if len(ranked)!=3:continue
 agg[tier]=dict(geomean=gm(g['median'] for g in ranked),layout_wins=sum(g['median']>1 for g in ranked),pair_wins=sum(g['wins'] for g in ranked),min=min(g['median'] for g in ranked),seed_geomeans=[gm(g['seed_medians'][j] for g in ranked) for j in (0,1)])
 if tier!='standalone':assert math.isclose(agg[tier]['geomean'],hosted['field'][tier]['all_three_geomean'],rel_tol=1e-12)
baseagg=agg.get('standalone'); gate=bool(baseagg and all(qual[f,x] for f in summary for x in (base,cand)) and summary['repeat-divisible-winner']['control_bitwise'] and summary['repeat-divisible-winner']['tiers']['standalone']['median']>=.97 and baseagg['geomean']>=1.03 and baseagg['layout_wins']>=2 and baseagg['min']>=.98 and min(baseagg['seed_geomeans'])>1)
assert gate==hosted['primary']['passes']
record=dict(provenance=proof,architecture=architecture['Architecture'],cpu_model=architecture['Model name'],logical_cpu=cpu,eligible=sum(qual.values()),attempted_pairs=len(qual),references=len(golds),accuracy_rows=len(accuracy),time_rows=len(timings),passed_accuracies=sum(r['passed']=='1' for r in accuracy),passed_time_rows=sum(r['passed']=='1' for r in timings),checked_cells=checked_cells,max_abs=dict(max_err),primary_passed=gate,aggregates=agg,by_family=summary)
args.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('by_family','max_abs','provenance','aggregates')},separators=(',',':')))
