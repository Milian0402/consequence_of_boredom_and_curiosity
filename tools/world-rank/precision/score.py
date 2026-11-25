"""Recompute bounded-fixture rankings and keep reduced-precision measurements separate."""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

from study import FIXTURES, PANEL, SEEDS, digest, read, require

def indexed(rows, key):
    out={}
    for row in rows:
        ind=key(row)
        require(ind not in out,'duplicate evidence '+str(ind))
        out[ind]=row
    return out

def pair(x,y):
    ratios=[b/a for a,b in zip(x,y)]
    differences=[1000*(b-a) for a,b in zip(x,y)]
    loomed=[st.median(ratios[:i]+ratios[i+1:]) for i in range(6)]
    return dict(peer_over_candidate=ratios,peer_minus_candidate_ms=differences,
                median=st.median(ratios),wins=sum(v>1 for v in ratios),ties=ratios.count(1),
                seed_medians=[st.median(ratios[:3]),st.median(ratios[3:])],
                leave_one_out_median_range=[min(loomed),max(loomed)],raw_median_win=st.median(x)<st.median(y))

def evaluate(out):
    require(digest(out/'panel.csv')==PANEL,'registered panel changed')
    p=read(out/'panel.csv');host=json.loads((out/'measurement.json').read_text())
    roster=indexed(read(out/'methods.csv'),lambda r:r['method'])
    admission=json.loads((out/'admission.json').read_text());routes=json.loads((out/'native-routes.json').read_text())
    q=indexed(read(out/'qualification.csv'),lambda r:(r['family'],r['method']))
    a=indexed(read(out/'accuracy.csv'),lambda r:(r['family'],r['method'],r['mode'],int(r['seed'])))
    t=indexed(read(out/'timings.csv'),lambda r:(r['family'],r['method'],int(r['seed']),int(r['round'])))
    gold=indexed(read(out/'goldens.csv'),lambda r:(r['family'],r['mode'],int(r['seed'])))
    slots=[(s,r) for s in SEEDS for r in (1,2,3)]
    required={'cob-released','openblas','blasfeo','onednn-oneshot'}
    required.update({'mkl','mkl-jit','aocl'} if host['architecture']=='x86_64' else {'armpl'})
    roles={m:x['role'] for m,x in roster.items()}
    mixed=[m for m,role in roles.items() if role=='mixed']
    target={m:{tier:[] for tier in ('fp32_oneshot','fp32_any','whole_any')} for m in mixed}
    cases={}
    for idx,spec in enumerate(p):
        f=spec['family'];m,n,k=(int(spec[v]) for v in ('m','n','k'))
        candidates={};record={};excluded={};first={};first_values={};warmvals={};errors={}
        require({nam for ff,nam in q if ff==f}==set(roster),'incomplete family roster')
        for mode,seed in FIXTURES:
            g=gold[f,mode,seed]
            require(int(g['checked'])==m*n and g['cpu']==str(host['cpu']) and
                    int(g['bytes_each'])==m*n*8 and g['signed_spots']==g['weight_spots']=='23','incomplete signed and weight golden')
        timed=[name for name in roster if q[f,name]['timed']=='1']
        for block,seed in enumerate(SEEDS):
            for rnd in (1,2,3):
                off=(idx+3*(rnd-1)+5*block)%max(1,len(timed))
                order=timed[off:]+timed[:off]
                if block:order.reverse()
                require({nm for ff,nm,s,r in t if (ff,s,r)==(f,seed,rnd)}==set(timed),'missing or extra warm observations')
                require(all(int(t[f,nm,seed,rnd]['position'])==i for i,nm in enumerate(order)),'warm order changed')
        for name in roster:
            qual=q[f,name];rs=[t[f,name,seed,r] for seed,r in slots] if name in timed else []
            accuracy=[a[f,name,mode,seed] for mode,seed in FIXTURES if (f,name,mode,seed) in a]
            if rs: require(len(accuracy)==4,'missing fixture from timed method')
            verified=[]
            for row in [*accuracy,*rs]:
                if row['structural']!='1':continue
                seed=int(row['seed']);g=gold[f,row['mode'],seed]
                require(row['a_sha256']==g['a_sha256'] and row['b_sha256']==g['b_sha256'] and
                        row['threads']=='1' and row['cpu']==str(host['cpu']) and int(row['checked'])==m*n,'input, CPU or count disagrees')
                la,lb,lc,ar,ac,br,bc,cr,cc=[int(spec[z]) for z in 'lda ldb ldc ar ac br bc cr cc'.split()]
                page=host['page'];guards=((cr+m+2)*lc*4+page-1)//page*(page//4)-m*n
                require(int(row['guards'])==guards and [int(row[z]) for z in ('a_mod64','b_mod64','c_mod64')]==
                        [4*x%64 for x in (ar*la+ac,br*lb+bc,cr*lc+cc)],'parent view differs')
                absolute=float(row['max_abs']);rho=float(row['max_rho']);gamm=float(row['gamma'])
                require(math.isclose(gamm,k/(16777216-k),rel_tol=1e-14),'gamma changed')
                numerical=math.isfinite(absolute) and absolute<=.002 and math.isfinite(rho) and all(int(row[z])==0 for z in ('nonfinite','over_abs','over_gamma'))
                require(row['numerical']==str(int(numerical)),'incorrect numerical admission')
                if roles[name] in ('mixed','precision-reference'):
                    proof=json.loads(row['route_evidence']);route=routes[f][name]
                    require(route['eligible'] and all(y['route']==1 for y in proof) and
                            [y['stage'] for y in proof]==(['FIRST','WARM'] if int(row['round']) else ['FIRST']),'missing measured native route evidence')
                if int(row['round']):
                    require(a[f,name,'random',seed]['c_sha256']==row['c_sha256'] and float(row['warm_s'])>0 and float(row['warm_cpu_s'])>0,'warm output or timer')
                verified.append(row)
            negatives=[r for r in [*accuracy,*rs] if r['structural']=='0' or r['numerical']=='0']
            errors[name]=dict(accuracy_records=len(accuracy),warm_records=len(rs),checked=sum(int(r['checked']) for r in verified),
                              nonfinite=sum(int(r['nonfinite']) for r in verified),over_abs=sum(int(r['over_abs']) for r in verified),
                              over_gamma=sum(int(r['over_gamma']) for r in verified),
                              max_abs=max((float(r['max_abs']) for r in verified if math.isfinite(float(r['max_abs']))),default=None),
                              max_rho=max((float(r['max_rho']) for r in verified if math.isfinite(float(r['max_rho']))),default=None),
                              failed_records=len(negatives))
            global_ok=(admission.get(name,{}).get('accuracy_qualified') is True and name in mixed
                       if roles[name] in ('mixed','precision-reference') else True)
            expected=bool(rs) and len(accuracy)==4 and all(r['structural']==r['numerical']=='1' for r in accuracy) and global_ok
            require((qual['eligible_strict']=='1')==expected,'incorrect qualification '+name)
            if len(rs)==6 and all(r['structural']=='1' for r in rs):
                vals=[float(r['warm_s']) for r in rs]
                warmvals[name]=st.median(vals)*1000
                first_values[name]=[float(r['first_s']) for r in rs]
                first[name]=st.median(first_values[name])*1000
                if expected and not negatives:
                    candidates[name]=vals
                    continue
            excluded[name]=qual['reason'] or (negatives[0]['reason'] if negatives else 'timing evidence incomplete')

        def matches(tier, peer):
            return {'fp32_oneshot':roles[peer] not in ('mixed','precision-reference','independent-retained'),
                    'fp32_any':roles[peer] not in ('mixed','precision-reference'),
                    'independent_oneshot':roles[peer]=='independent-oneshot',
                    'independent_any':roles[peer] in ('independent-oneshot','independent-retained'),
                    'whole_any':True}[tier]
        tiers=('fp32_oneshot','fp32_any','independent_oneshot','independent_any','whole_any')
        winners={tier:min((z for z in candidates if matches(tier,z)),key=lambda z:st.median(candidates[z]),default=None) for tier in tiers}
        for name in mixed:
            if name not in candidates:continue
            x=candidates[name];pairs={z:pair(x,values) for z,values in candidates.items() if name!=z}
            summaries={}
            for tier in tiers:
                chosen=min((z for z in pairs if matches(tier,z)),key=lambda z:st.median(candidates[z]),default=None)
                if not chosen:continue
                comp=pairs[chosen];needs=set(required)
                if tier in ('whole_any','fp32_any','independent_any'):needs.add('onednn-cached-plan')
                if tier.startswith('independent_'):needs.discard('cob-released')
                absent=sorted(needs-set(candidates));passes=not absent and comp['raw_median_win'] and comp['median']>1.02 and comp['wins']>=5
                summaries[tier]=dict(peer=chosen,passes=passes,missing_required=absent,**comp)
                if tier in ('fp32_oneshot','fp32_any'):
                    rivals=[z for z in pairs if matches(tier,z)]
                    # Leaders can differ at first call or between the two independent input blocks.
                    first_peer=min(rivals,key=lambda z:st.median(first_values[z]))
                    comparisons={'warm':(chosen,x,candidates[chosen]),'first':(first_peer,first_values[name],first_values[first_peer])}
                    for i in range(2):
                        sl=slice(3*i,3*i+3)
                        block_peer=min(rivals,key=lambda z:st.median(candidates[z][sl]))
                        comparisons['seed_'+str(SEEDS[i])]=(block_peer,x[sl],candidates[block_peer][sl])
                    landmarks={}
                    for measure,(peer,candidate_data,peer_data) in comparisons.items():
                        ratios=[b/a for a,b in zip(candidate_data,peer_data)]
                        median=st.median(ratios)
                        landmarks[measure]=dict(peer=peer,paired_ratios=ratios,median=median,
                           raw_ratio=st.median(peer_data)/st.median(candidate_data),
                           thresholds={str(th):dict(observations=sum(v>=th for v in ratios),
                               reaches=not absent and median>=th and st.median(peer_data)/st.median(candidate_data)>=th)
                               for th in (1.5,2.0)})
                    summaries[tier]['magnitude']=landmarks
                if tier in target[name]:target[name][tier].append(summaries[tier])
            record[name]=dict(tiers=summaries,pairs=pairs)
        cases[f]=dict(shape=[m,n,k],layout=spec['layout'],strict_qualified=list(candidates),exclusions=excluded,
                      fastest=winners,ranked_warm_ms={z:warmvals[z] for z in candidates},all_structural_warm_ms=warmvals,
                      first_ms=first,comparisons=record,accuracy=errors)
    summaries={}
    for name,tiers in target.items():
        summaries[name]={}
        for tier,all_rows in tiers.items():
            gm=st.geometric_mean(v['median'] for v in all_rows) if all_rows else None
            seeds=[st.geometric_mean(v['seed_medians'][i] for v in all_rows) for i in (0,1)] if all_rows else []
            passes=len(all_rows)==len(p) and all(v['passes'] for v in all_rows) and min(seeds)>1
            summaries[name][tier]=dict(covered=len(all_rows),registered=len(p),ratio_geomean=gm,
                                       seed_geomeans=seeds,raw_layout_wins=sum(v['raw_median_win'] for v in all_rows),
                                       paired_wins=sum(v['wins'] for v in all_rows),paired_observations=6*len(all_rows),
                                       layouts_over_gate=sum(v['passes'] for v in all_rows),passes=passes)
    for name,tiers in target.items():
        for tier in ('fp32_oneshot','fp32_any'):
            summaries[name][tier]['magnitude_layout_counts']={measure:{
                str(th):sum(r['magnitude'][measure]['thresholds'][str(th)]['reaches'] for r in tiers[tier])
                for th in (1.5,2.0)} for measure in ['warm','first',*('seed_'+str(s) for s in SEEDS)]}
    return dict(primary='mixed6',panel_sha256=PANEL,host=host,coverage={'goldens':len(gold),'accuracy':len(a),'warm':len(t)},
                roles=roles,cases=cases,gates=summaries)

if __name__=='__main__':
    root=Path(sys.argv[1]); summary=evaluate(root)
    (root/'score.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'primary':summary['primary'],'gates':summary['gates'],'coverage':summary['coverage']}))
