"""Offline secondary vendor-wide audit. Uses only Python's standard library and raw evidence."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median

PANEL='c05f30a91720c54086c0b6d5e858267501af8488e4af9614555e2f5e12c55bd6'
REGISTRATION='eac860a798287d2a11ce2ce1dac114129409c5124522fd60fc4f115d3c69e87b'
SEEDS=(348271759,710438957)
PROVIDERS=('openblas','mkl')
ASSISTED=[f'vendor-wide-{p}-{mode}' for p in PROVIDERS for mode in ('full','split')]
MATCHED=[f'matched-{p}' for p in PROVIDERS]
DIRECT=['openblas','mkl','mkl-jit','aocl','blasfeo','onednn-oneshot','onednn-cached-plan',*MATCHED]
STANDALONE=['stream-incumbent','stream-native']
ROSTER=set(DIRECT+STANDALONE+ASSISTED)
DIMS=('m','n','k','pad')
ROUNDS=[(b,r) for b in (1,2) for r in (1,2,3)]
QUAL={('random',1,SEEDS[0]),('random',2,SEEDS[1]),('scaled',0,SEEDS[0]),('cancel',0,SEEDS[0])}
TRACE=('selected','main_rows','tail_rows','vendor_calls','vendor_m','vendor_n','vendor_k','tail_completed')
MODES=('matched_direct','matched_public_full','standalone','stream','best_oneshot','best_any')
LIBS={'mkl':('libmkl_intel_lp64.so.3','libmkl_sequential.so.3','libmkl_core.so.3'),'openblas':('libcob_rank_openblas.so',)}


def check(test,message):
    if not test:raise ValueError(message)


def filesum(path):
    return sha256(path.read_bytes()).hexdigest()


def manifest(path):
    pairs={}
    for line in path.read_text().splitlines():
        digest,filename=line.split(maxsplit=1);filename=filename.lstrip('*')
        check(re.fullmatch('[0-9a-f]{64}',digest) and filename not in pairs,'invalid hash entry: '+line)
        pairs[filename]=digest
    return pairs


def records(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def indexed(items,field,expected):
    result={r[field]:r for r in items}
    check(set(result)==set(expected) and len(items)==len(expected),f'unexpected or missing {field}: {set(expected)^set(result)}')
    return result


def evidence(root):
    check(filesum(root/'registration.sha256')==REGISTRATION,'registration differs from pre-measurement source snapshot')
    registered=manifest(root/'registration.sha256')
    check(registered['panel.csv']==PANEL==filesum(root/'panel.csv'),'panel not frozen')
    check(registered['protocol.md']==filesum(root/'protocol.md'),'protocol mismatch')
    check(set((root/'freeze-check.txt').read_text().splitlines())=={name+': OK' for name in registered},'incomplete registered file verification')
    metadata=indexed(records(root/'methods.csv'),'implementation',ROSTER)
    builds=indexed(records(root/'builds.csv'),'implementation',ASSISTED+MATCHED)
    for name,row in metadata.items():
        if name in ASSISTED:role,provider,abi='assisted',name.split('-')[2],'direct'
        elif name in STANDALONE:role,provider,abi='standalone','cob','linked-public'
        elif name=='openblas':role,provider,abi='oracle','openblas','direct'
        else:role,provider,abi='direct',('onednn' if name.startswith('onednn-') else name.removeprefix('matched-')),'direct'
        check((row['role'],row['provider'],row['abi'])==(role,provider,abi),f'method metadata {name}')
        check(re.fullmatch('[0-9a-f]{64}',row['artifact_sha256']),f'artifact hash {name}')
    provider_records={}
    for prov in PROVIDERS:
        entries=manifest(root/(prov+'-provider.sha256'))
        by_basename={Path(p).name:p for p in entries}
        need=set(LIBS[prov])|({'libopenblas.a'} if prov=='openblas' else set())
        check(set(by_basename)==need and len(entries)==len(need),f'{prov} provider hash manifest members')
        provider_records[prov]={b:entries[path] for b,path in by_basename.items()}
        for mode in ('direct','full','split'):
            name=f'matched-{prov}' if mode=='direct' else f'vendor-wide-{prov}-{mode}'
            b=builds[name]
            check(b['provider']==prov and b['mode']==mode and b['timed_sha256']==metadata[name]['artifact_sha256'],f'{name} build/config mismatch')
            trace=b['trace_sha256']
            check((mode=='direct' and not trace) or (mode!='direct' and re.fullmatch('[a-f0-9]{64}',trace) and trace!=b['timed_sha256']),f'{name} trace/timed fingerprints')
            for suffix in ([''] if mode=='direct' else ['','.trace']):
                text=(root/(name+suffix+'.ldd')).read_text()
                check('not found' not in text and not re.search(r'libmkl_(?:intel|gnu|tbb)_thread',text),f'{name} unresolved or threaded linkage')
                deps=dict(re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)',text,flags=re.M))
                for lib in LIBS[prov]:
                    check(lib in deps and Path(deps[lib])==Path(by_basename[lib]),f'{name}{suffix} {lib} differs from hashed provider path')
                if prov=='openblas':check(not any('libmkl_' in dep for dep in deps),'mixed OpenBLAS/MKL linkage')
                else:check(not any('openblas' in dep.lower() for dep in deps),'mixed MKL/OpenBLAS linkage')
    symbols={l.split()[-1] for l in (root/'openblas-bindings.txt').read_text().splitlines() if re.search(r'\s[TW]\s',l)}
    check({'sgemm_','cblas_sgemm','openblas_get_parallel','openblas_get_num_threads','openblas_get_config'}<=symbols,'missing OpenBLAS binding')
    verified=defaultdict(set);current=None
    for line in (root/'worker.log').read_text(errors='replace').splitlines():
        if line.startswith('worker '):
            match=re.match(r'worker (\S+) block=(\d+) seed=',line)
            current=match.group(1) if match else None
        elif line.startswith('qualified single-thread LP64 SGEMM binding ') and current in builds:
            verified[current].add(line.split()[-1])
    for name in builds:
        check(verified[name]=={LIBS[builds[name]['provider']][0]},f'{name} runtime private-namespace provider confirmation missing or inconsistent')
    return dict(provider_artifacts=provider_records,provider_runtime={n:sorted(v) for n,v in verified.items()},hashes={x:filesum(root/x) for x in ['registration.sha256','methods.csv','builds.csv','routes.csv','accuracy.csv','timings.csv']})


def audit(root):
    provenance=evidence(root)
    panel=records(root/'panel.csv');cases={r['family']:tuple(int(r[k]) for k in DIMS) for r in panel}
    ranking=json.loads((root/'ranking.json').read_text())
    methods=list(ranking['implementations'])
    check(len(cases)==12 and len(methods)==len(ROSTER) and set(methods)==ROSTER,'participant/panel coverage')
    check(ranking['panel_sha256']==PANEL and ranking['phase']=='stream-b' and tuple(ranking['seeds'])==SEEDS and ranking['rounds_per_seed']==3 and ranking['full_output_tolerance']==.002,'frozen schedule')
    check(not ranking['errors'],'study recorded qualification or timing errors')
    def key(row):
        f,n=row['family'],row['implementation']
        check(f in cases and n in ROSTER and tuple(int(row[k]) for k in DIMS)==cases[f],'bad dimensions/participant')
        return f,n
    cpus=set();dispatch={};routes=records(root/'routes.csv')
    for row in routes:
        f,n=entry=key(row);m,w,k,_=cases[f]
        check(n in ASSISTED and entry not in dispatch,'unexpected or repeated route')
        eligible=m in range(48,257) and 512<=w<=4096 and 96<=k<=768 and w>=4*m and m*k<=98304
        tail=(m%16 if 0<m%16<=4 and n.endswith('-split') else 0) if eligible else 0
        main=m-tail if eligible else 0
        want=(1,main,tail,1,main,w,k,int(tail>0)) if eligible else (0,)*8
        check(tuple(int(row[k]) for k in TRACE)==want and row['passed']=='1' and not row.get('error'),f'route {entry}')
        check(int(row['timed_checked'])==int(row['trace_checked'])==m*w and row['threads']==row['guards']==row['inputs_unchanged']=='1',f'route correctness {entry}')
        cpus.add(row['cpu']);dispatch[entry]=want
    check(len(routes)==len(cases)*len(ASSISTED),'route coverage')
    checks=defaultdict(set);values=defaultdict(dict);counts=defaultdict(dict)
    accuracy=records(root/'accuracy.csv');times=records(root/'timings.csv')
    max_error=0.;checked=0
    for row in accuracy:
        f,n=entry=key(row);token=(row['mode'],int(row['block']),int(row['seed']))
        check(token in QUAL and token not in checks[entry] and int(row['stress'])==int(token[1]==0) and row['phase']=='stream-b',f'fixture identity {entry}')
        detail=dict(piece.split('=',1) for piece in row['detail'].split(',') if '=' in piece)
        e=float(row['max_abs_error'])
        check(row['verified']=='1' and row['nonfinite']==row['guard_changes']==detail['input_changes']=='0' and math.isfinite(e) and 0<=e<=.002,f'FP64 or guard fail {entry}')
        check(math.isfinite(float(detail['first_sgemm_seconds'])) and float(detail['first_sgemm_seconds'])>0,f'invalid first time {entry}')
        if n.startswith('onednn-'):check(bool(detail.get('onednn_impl')),f'missing oneDNN identity for {entry}')
        check(int(row['checked'])==cases[f][0]*cases[f][1] and detail['threads']==row['threads']=='1' and row['cpu']==detail['cpu'],f'fixture threads, CPU or extent {entry}')
        cpus.add(row['cpu']);checks[entry].add(token);checked+=int(row['checked']);max_error=max(max_error,e)
    order_map={f:i for i,f in enumerate(cases)}
    for row in times:
        f,n=entry=key(row);b,r=rid=int(row['block']),int(row['round'])
        check(rid in ROUNDS and rid not in values[entry] and int(row['seed'])==SEEDS[b-1] and row['stress']=='0' and row['mode']=='random' and row['phase']=='stream-b',f'timing pair {entry}')
        duration=float(row['seconds']);batch=int(row['calls']);position=int(row['position'])
        check(math.isfinite(duration) and duration>0 and batch>0 and row['threads']=='1' and row['resident'] in ('0','1'),f'bad duration, batch or thread count {entry}')
        # Invert the recorded rotation/reversal to recover each implementation's original index.
        pos=position-1
        check(0<=pos<len(methods),'impossible method position')
        if (r+b)%2:pos=len(methods)-1-pos
        idx=(pos+order_map[f]+r-1)%len(methods)
        check(methods[idx]==n,f'wrong method order {entry} {rid}')
        cpus.add(row['cpu']);values[entry][rid]=duration;counts[entry][rid]=batch
    check(len(cpus)==1 and next(iter(cpus)).isdigit(),'worker CPU consistency')
    for f in cases:
        for n in methods:
            check(checks[f,n]==QUAL and set(values[f,n])==set(ROUNDS),f'incomplete fixtures/timing {f} {n}')
        for pair in ROUNDS:check(len({counts[f,n][pair] for n in methods})==1,f'unmatched pair batching {f} {pair}')
    summaries=[];cohorts={}; ratios={}
    for f,dims in cases.items():
        typical={n:median(values[f,n].values()) for n in methods}
        best=lambda ns:min(ns,key=lambda n:(typical[n],n))
        for c in ASSISTED:
            prov=c.split('-')[2]
            comparators={'matched_direct':'matched-'+prov,'matched_public_full':'vendor-wide-'+prov+'-full',
                         'standalone':'stream-incumbent','stream':'stream-native',
                         'best_oneshot':best([n for n in DIRECT if n!='onednn-cached-plan']),'best_any':best(DIRECT)}
            row={'family':f,**dict(zip(DIMS,dims)),'candidate':c,'median_ms':typical[c]*1000,'tail_rows':dispatch[f,c][2]}
            for label,n in comparators.items():
                z=[values[f,n][pair]/values[f,c][pair] for pair in ROUNDS]
                ratios[f,c,label]=z
                row[label+'_name']=n;row[label+'_paired']=median(z);row[label+'_wins']=sum(x>1 for x in z)
                row[label+'_seed1']=median(z[:3]);row[label+'_seed2']=median(z[3:])
            summaries.append(row)
    groups={'original_five':[f for f in cases if f.startswith('fresh-') and not f.startswith('fresh-tail-')],
            'added_four':[f for f in cases if f.startswith('fresh-tail-')]}
    check([len(g) for g in groups.values()]==[5,4],'fresh shape groups')
    def gm(items):return math.exp(math.fsum(map(math.log,items))/len(items))
    for c in ASSISTED:
        sets={}
        for group,families in groups.items():
            norm={}
            for label in MODES:
                zs=[ratios[f,c,label] for f in families];medians=[median(z) for z in zs]
                norm[label]={'geomean_paired':gm(medians),'seed_gms':[gm([median(z[:3]) for z in zs]),gm([median(z[3:]) for z in zs])],
                             'shape_wins':sum(z>1 for z in medians),'paired_wins':sum(z>1 for arr in zs for z in arr),'minimum':min(medians)}
            sets[group]=dict(shapes=families,comparisons=norm)
        sets['tail_beats_both']=[f for f in cases if dispatch[f,c][2] and median(ratios[f,c,'matched_direct'])>1 and median(ratios[f,c,'matched_public_full'])>1]
        cohorts[c]=sets
    result={'valid':True,'cpu':next(iter(cpus)),'seeds':SEEDS,'participants':len(methods),'shapes':len(cases),'timed_rounds':len(times),'accuracy_fixtures':len(accuracy),
            'fp64_cells':checked,'max_abs_error':max_error,'public_trace_rows':len(dispatch),'provenance':provenance,'candidates':cohorts}
    return result,summaries


def main():
    parser=argparse.ArgumentParser();parser.add_argument('artifact',type=Path);parser.add_argument('out',type=Path);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        source=args.artifact
        if not (source/'methods.csv').is_file():
            paths=[p.parent for p in source.rglob('mkl-provider.sha256') if (p.parent/'methods.csv').is_file()]
            check(len(paths)==1,'could not uniquely locate vendor-wide result directory')
            source=paths[0]
        summary,rows=audit(source)
    except (ValueError,OSError,KeyError,IndexError,ZeroDivisionError) as exc:
        (args.out/'independent-score.json').write_text(json.dumps({'valid':False,'error':str(exc)},indent=2)+'\n')
        raise SystemExit(str(exc))
    with (args.out/'independent-per-shape.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (args.out/'independent-score.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
