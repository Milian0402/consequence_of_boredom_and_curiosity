"""Independent raw-data replay of the registered ragged HK15 comparison."""
import argparse,csv,hashlib,json,math,re,statistics,zipfile
from collections import Counter,defaultdict
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('raw',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--original',type=Path);ap.add_argument('--receipt',type=Path);ap.add_argument('--run',type=int);ap.add_argument('--revision')
args=ap.parse_args();root=args.raw;med=statistics.median;gm=statistics.geometric_mean
read=lambda name:list(csv.DictReader((root/name).open()))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if args.original:
 proof=json.loads(args.receipt.read_text());assert proof['sha']==args.revision and proof['run']==args.run and proof['job_conclusion']=='success'
 assert sha(args.original)==proof['artifact_sha256'][7:]
 with zipfile.ZipFile(args.original) as z:
  assert z.testzip() is None
  for item in z.infolist():
   if not item.is_dir():assert z.read(item)==(root/item.filename).read_bytes(), item.filename
else:proof=None
assert sha(root/'panel.csv')=='204a37e5fd61c9530f01ec7a95e01ea2d09acf719784f198af6cdd6221609eff'
assert sha(root/'protocol.md')=='81d8142ae0feb265c092e9ed3ed26b523b777deb8a4b9ca9fd70a6cb563114ee'
assert (root/'production-sha.txt').read_text().splitlines()[0]=='f322a9ab58454762563b97323fdc5f29016cc39f'
if args.revision: assert (root/'benchmark-sha.txt').read_text().strip()==args.revision
architecture=dict((k.strip(),v.strip()) for k,v in (s.split(':',1) for s in (root/'lscpu.txt').read_text().splitlines() if ':' in s))
cpu=int((root/'pinned-cpu.txt').read_text());page=int((root/'page-bytes.txt').read_text())
registry=read('sources/methods.csv')
methods=[r['method'] for r in registry];roles={r['method']:r['role'] for r in registry}
assert len(methods)==len(set(methods)) and roles['ragged-hk15']=='candidate' and roles['released-hk15']=='baseline'
cand='ragged-hk15';base='released-hk15';seeds=(197906831,884039527)
fixtures={('random',seeds[0]),('random',seeds[1]),('cancel',seeds[0]),('scaled',seeds[1])}
pairs=[(s,r) for s in seeds for r in (1,2,3)]
cases=read('panel.csv');timings=read('timings.csv');accuracy=read('accuracy.csv');golds=read('goldens.csv');routes=read('routes.csv')
qual={(r['family'],r['method']):r['qualified']=='1' for r in read('qualified.csv')}
assert len(qual)==len(cases)*len(methods) and len(golds)==4*len(cases)
gold={(r['family'],r['mode'],int(r['seed'])):r for r in golds}
role_tiers={'standalone':{'baseline'},'independent_oneshot':{'independent-oneshot'},'vendor_assisted':{'vendor-assisted'},
            'independent_any':{'independent-oneshot','independent-retained'},'whole_any':set(roles.values())}
native='mkl' if architecture['Architecture']=='x86_64' else 'armpl'
released={f'release-{prov}-{opt}' for prov in ('openblas',native) for opt in ('aligned-two','two')}
required_direct={'openblas',native}|({'aocl','mkl-jit'} if native=='mkl' else set())
required_independent=required_direct|{'blasfeo','fastmmw-768','fastmmw-1536'}
if any(r['method']=='onednn-oneshot' and r['available']=='1' for r in registry):required_independent.add('onednn-oneshot')
tiers=dict(old_standalone={base},direct=required_direct,independent_oneshot={m for m in methods if roles[m]=='independent-oneshot'},
 independent_any={m for m in methods if roles[m].startswith('independent-')},released_winograd=released,
 whole_oneshot={m for m in methods if roles[m]!='independent-retained'},whole_any=set(methods))
required=dict(old_standalone={base},direct=required_direct,independent_oneshot=required_independent,
 independent_any=required_independent|{'onednn-cached-plan'},released_winograd=released,
 whole_oneshot=required_independent|released|{base,cand},whole_any=required_independent|released|{base,cand,'onednn-cached-plan'})
contenders=[cand,'hk15-openblas-leaf',f'hk15-{native}-leaf']
paired_rows={(r['family'],r['contender'],r['peer']):r for r in read('paired.csv')}
reported_pairs=set()
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
   if m.startswith('mkl-handle-'):
    retained=m.endswith('prepared'); assert r['implementation'] in ('explicit-generated','explicit-fallback')
    assert int(r['cached'])==int(retained) and int(r['plans'])==(1 if mn=='accuracy' or retained else 2)
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
 traced={r['method'] for r in registry if r.get('trace_worker_sha256') and r['method'] in eligible}
 assert traced=={r['method'] for r in routes if r['family']==f and qual[f,r['method']]}, 'routing evidence coverage'
 for r in routes:
  if r['family']!=f or not qual[f,r['method']]:continue
  name=r['method'];assert r['verified']=='1' and r['ordinary_sha256']==r['trace_sha256']==rawacc[name]['random',seeds[0]]['c_sha256']
  counts=json.loads(r['receipt'])
  if name in (base,cand):
   active=(name==cand or not (M%2 or N%3 or K%3));fringe=active and bool(M%2 or N%3 or K%3)
   expected={'public_calls':1,'args_ok':1,'parents':active,'leaves':15*active,'fused_leaves':15*active,'finishes':fringe,'k_terms':(K%3)*fringe,'right':bool(N%3)*fringe,'bottom':bool(M%2)*fringe,'core_checked':(M-M%2)*(N-N%3)*bool(fringe and K%3)}
   assert all(counts[k]==v for k,v in expected.items())
  if name.startswith('release-'):
   assert counts['provider_ok']==1 and counts['provider_identity']
   observations=[list(map(float,v.split(','))) for v in counts['leaf_observations'].split('|') if v]
   assert len(observations)==counts['calls']
   assert counts['calls']>=49 and counts['full_provider_forward']==0, 'registered released route required at all seven geometries'
   if counts['full_provider_forward']:
    assert counts['calls']==1 and tuple(observations[0][1:7])==(M,N,K,la,lb,lc)
   elif counts['calls']>=49:
    aligned=name.endswith('aligned-two');qa,qk=(64,32) if aligned else (4,4)
    mc,nc,kc=M-M%qa,N-N%qa,K-K%qk
    assert counts['core_calls']==counts['core_overwrites']==49 and counts['quadrants_ok']==counts['edges_ok']==1
    assert counts['right_strips']==int(N!=nc) and counts['bottom_strips']==int(M!=mc) and counts['depth_updates']==int(aligned and K!=kc)
    assert all(v[1:4]==[mc//4,nc//4,kc//4] and v[7:]==[1,0] for v in observations[:49])
   else: assert counts['route'] in ('native-no-vendor-calls','other-released-provider-path')
  if name.startswith('hk15-'):
   assert counts['child_count']==counts['provider_calls']==15 and counts['workspace_ok'] and counts['provider']==name.split('-')[1]
   assert counts['depth_terms']==K%3 and counts['fringe_calls']==int(bool(N%3))+M%2
   ops=counts['calls'];assert len(ops)==15 and len({z['index'] for z in ops})==15
   for op in ops:
    assert (op['m'],op['n'],op['k'])==(M//2,N//3,K//3) and op['beta']==0 and op['rowmajor_nt_alpha1']==1
    assert (op['lda'],op['ldb'],op['ldc'])==(la if op['a_block']>=0 else K//3,lb if op['b_block']>=0 else N//3,lc if op['c_block']>=0 else N//3)
   assert [op['a_block'] for op in ops]==[3,-1,1,-1,-1,-1,2,-1,-1,-1,-1,5,-1,-1,-1]
   assert [op['b_block'] for op in ops]==[-1,-1,-1,1,-1,2,-1,-1,-1,3,5,-1,7,-1,6]
   assert [op['c_block'] for op in ops]==[0,-1,-1,-1,-1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1]
  if name in ('acl','blis'):
   assert [counts[k] for k in ('m','n','k','lda','ldb','ldc')]==[M,N,K,la,lb,lc]
   assert counts['observations'] and all(x['calls']>0 for x in counts['observations'])
   if name=='blis':
    assert all(x['kernel'].startswith('bli_sgemm_armsve_asm_') and x['sve_bits']>=128 and x['kernel_mnk']>=M*N*K for x in counts['observations'])
   else:
    assert all(x['kernel'].startswith('arm_gemm::') and 'fp32' in x['kernel'].lower() or
               x['kernel'].startswith('arm_gemm::') and 'sgemm' in x['kernel'].lower() for x in counts['observations'])
  if name.startswith('mkl-handle-'):
   assert counts['caller_addresses_verified'] and counts['actual_geometry_verified'] and counts['creates']==counts['gets']==counts['calls']==1
   assert counts['implementation']==('explicit-generated' if counts['execution_status']==0 else 'explicit-fallback')
  if name.startswith('fastmmw-'):
   split=1
   while (min(M,N,K)+split-1)//split>int(name.split('-')[1]):split*=2
   assert counts['core_multiple']==split and counts['leaves']==7**(split.bit_length()-1) and counts['padded']==int(any(dim%split for dim in (M,N,K)))
 times={m:[float(rawtimes[m][pair]['warm_s']) for pair in pairs] for m in eligible}
 medians={m:med(v) for m,v in times.items()}; trial={}
 for contender in contenders:
  warm=times.get(contender); over={}
  if warm:
   for peer in eligible:
    if peer==contender:continue
    key=f,contender,peer; emitted=paired_rows[key];reported_pairs.add(key)
    reference=times[peer]; prs=[v/w for v,w in zip(reference,warm)]; ds=[1000*(v-w) for v,w in zip(reference,warm)]
    for q in range(6):
     ix=f'seed{1+q//3}_r{1+q%3}'
     assert float(emitted[ix+'_ratio'])==prs[q] and math.isclose(float(emitted[ix+'_delta_ms']),ds[q],abs_tol=1e-12)
    assert float(emitted['paired_median'])==med(prs) and math.isclose(float(emitted['median_delta_ms']),med(ds),abs_tol=1e-12)
    assert int(emitted['wins'])==sum(x>1 for x in prs)
  for name,universe in tiers.items():
   rivals=sorted((m for m in eligible if m in universe and m!=contender),key=lambda m:(medians[m],m))
   missing=sorted(required[name]-set(eligible))
   if not warm or not rivals:over[name]=dict(passes=False,missing=missing);continue
   best=rivals[0]; ref=times[best];ratios=[b/c for b,c in zip(ref,warm)];deltas=[1000*(b-c) for b,c in zip(ref,warm)]
   medratio=med(ratios);wins=sum(r>1 for r in ratios); blocks=[med(ratios[:3]),med(ratios[3:])]
   raw_win=medians[contender]<medians[best];passed=not missing and raw_win and wins>=5 and medratio>1.02
   over[name]=dict(peer=best,time_ms=medians[best]*1000,ratio=medratio,wins=wins,seed_medians=blocks,
       raw_won=raw_win,passes=passed,missing=missing,paired_differences_ms=deltas,
       leave_one_out_ratio_range=[min(med(ratios[:i]+ratios[i+1:]) for i in range(6)),max(med(ratios[:i]+ratios[i+1:]) for i in range(6))])
   production=hosted['cases'][f]['entries'][contender]
   calc=production['pairs'][best];decision=production['tiers'][name]
   assert all(math.isclose(calc['delta_ms'][i],deltas[i],rel_tol=1e-13,abs_tol=1e-12) for i in range(6))
   assert calc['ratios']==ratios and decision['fastest_other']==best and decision['passes']==passed and decision['missing_required']==missing
  trial[contender]=dict(ms=1000*medians[contender] if contender in medians else None,tiers=over)
 summary[f]=dict(eligible=eligible,whole_raw_winner=min(eligible,key=lambda m:(medians[m],m)) if eligible else None,entries=trial,
                 first_call_ms={m:1000*med(float(rawacc[m]['random',s]['first_s']) for s in seeds) for m in eligible},
                 methods_ms={m:1000*v for m,v in medians.items()})
assert set(paired_rows)==reported_pairs
aggs={}
for contender in contenders:
 aggs[contender]={}
 for cohort in ('repeat','fresh'):
  subset=[v['entries'][contender]['tiers'] for f,v in summary.items() if f.startswith(cohort+'-')];assert len(subset)==3
  group={}
  for name in tiers:
   sel=[v[name] for v in subset]
   complete=all('ratio' in v for v in sel)
   bgs=[gm(v['seed_medians'][i] for v in sel) for i in range(2)] if complete else []
   passed=complete and all(v['passes'] for v in sel) and min(bgs)>1
   group[name]=dict(passes=bool(passed),geomean=gm(v['ratio'] for v in sel) if complete else None,
                   seed_geomeans=bgs,raw_wins=sum(v.get('raw_won',False) for v in sel),paired_wins=sum(v.get('wins',0) for v in sel))
   peer=hosted['gates'][contender][cohort][name]
   assert peer['passes']==passed and ((not complete and peer['paired_geomean'] is None) or math.isclose(peer['paired_geomean'],group[name]['geomean'],rel_tol=1e-13))
  aggs[contender][cohort]=group
 aggs[contender]['full_field_passes_both']=all(aggs[contender][g]['whole_any']['passes'] for g in ('repeat','fresh'))
 assert aggs[contender]['full_field_passes_both']==hosted['gates'][contender]['full_field_passes_both']
for method in (base,cand):
 manifest=root/'standalone'/method/'products.sha256'
 if manifest.is_file():
  products={Path(filename.strip()).name:digest for digest,filename in (line.split(None,1) for line in manifest.read_text().splitlines())}
  source=next(row for row in registry if row['method']==method)
  assert products['ordinary.worker']==source['worker_sha256'] and products['trace.worker']==source['trace_worker_sha256']
record=dict(provenance=proof,architecture=architecture['Architecture'],model=architecture['Model name'],cpu=cpu,
    qualified=sum(qual.values()),method_cases=len(qual),references=len(golds),accuracy_rows=len(accuracy),timing_rows=len(timings),
    passed_accuracies=sum(r['passed']=='1' for r in accuracy),passed_timings=sum(r['passed']=='1' for r in timings),
    checked_cells=checked_cells,max_abs=dict(max_err),primary=cand,gates=aggs,by_family=summary)
if (root/'guard-pairs/timings.csv').is_file():
 guarded=read('guard-pairs/timings.csv');gresults={}
 greport=json.loads((root/'guard-pairs/score.json').read_text())
 original_methods=('ragged-hk15','guarded-hk15');qshapes=[s for s in cases if s['family'].startswith(('repeat-','fresh-'))]
 assert len(guarded)==len(qshapes)*12
 for j,shape in enumerate(qshapes):
  f=shape['family'];time_ratios=[];cpu_ratios=[];identical=0
  M,N,cr,lc=[int(shape[k]) for k in ('m','n','cr','ldc')]
  count=((cr+M+2)*lc*4+page-1)//page*page//4-M*N
  for b,seed in enumerate(seeds):
   for rnd in range(3):
    rr=sorted((r for r in guarded if r['family']==f and int(r['seed'])==seed and int(r['round'])==rnd+1),key=lambda v:int(v['position']))
    assert [r['method'] for r in rr]==list(original_methods[::-1] if (j+rnd+b)%2 else original_methods)
    if any(r['passed']!='1' for r in rr):continue
    for r in rr:
     assert int(r['threads'])==1 and int(r['cpu'])==cpu and int(r['checked'])==M*N and int(r['guards'])==count
     assert all(r[k]==gold[f,'random',seed][k] for k in ('a_sha256','b_sha256')) and 0<=float(r['max_abs'])<=.002
    matched=rr[0]['c_sha256']==rr[1]['c_sha256']
    assert matched==greport['cases'][f]['pairs'][b*3+rnd]['bitwise_equal']
    if matched:
     by_name={r['method']:r for r in rr};unguarded,guard=(by_name[n] for n in original_methods)
     ratio=float(unguarded['wall_s'])/float(guard['wall_s']);assert ratio==greport['cases'][f]['pairs'][b*3+rnd]['wall_ratio']
     time_ratios.append(ratio);cpu_ratios.append(float(unguarded['caller_s'])/float(guard['caller_s']));identical+=1
  gresults[f]=dict(identical_pairs=identical,wall_median=med(time_ratios) if identical==6 else None,
                   cpu_median=med(cpu_ratios) if identical==6 else None)
 gagg={}
 for group in ('repeat','fresh'):
  vals=[v['wall_median'] for f,v in gresults.items() if f.startswith(group)]
  gagg[group]=gm(vals) if all(vals) else None
  assert (gagg[group] is None and greport['cohorts'][group] is None) or math.isclose(gagg[group],greport['cohorts'][group],rel_tol=1e-12)
 record['post_score_guard']=dict(cases=gresults,cohorts=gagg)
if (root/'dyadic-stress/status.csv').is_file():
 records={}
 for run in read('dyadic-stress/status.csv'):
  method=run['method'];records[method]=dict(status=run['status'])
  if run['status']!='complete':continue
  rr=read('dyadic-stress/'+method+'.csv'); assert len(rr)==19
  assert len({(r['pattern'],r['shift']) for r in rr})==19 and all(int(r['cells'])==int(r['m'])*int(r['n']) for r in rr)
  records[method].update(fixtures=19,with_error_above_band=sum(int(r['over_0_002'])>0 for r in rr),
    with_nonfinite=sum(int(r['nonfinite'])>0 for r in rr),wrong_sign_cells=sum(int(r['negative_from_positive']) for r in rr),
    nonfinite_cells=sum(int(r['nonfinite']) for r in rr),exact_fixtures=sum(int(r['wrong'])==0 and int(r['nonfinite'])==0 for r in rr),
    failed_fixtures=[r['pattern']+'@'+r['shift'] for r in rr if int(r['over_0_002']) or int(r['nonfinite'])])
 record['out_of_cohort_exact_dyadic_stress']=records
args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(architecture['Model name'], 'admitted',sum(qual.values()),'of',len(qual),'primary complete field',aggs[cand]['full_field_passes_both'])
