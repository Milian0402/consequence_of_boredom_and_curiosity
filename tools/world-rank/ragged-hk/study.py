"""Serial whole-public-call study of the five preregistered HK15 views."""
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import vendor_route

SEEDS=(154983127,720184093)
FIXTURES=(('random',SEEDS[0]),('random',SEEDS[1]),('cancel',SEEDS[0]),('scaled',SEEDS[1]))
METHOD_KEYS='method role worker plugin trace_worker trace_plugin'.split()
ABI='m n k lda ldb ldc ar ac br bc cr cc'.split()
QUAL='checked guards a_mod64 b_mod64 c_mod64 cpu threads max_abs first_s warm_s first_cpu_s warm_cpu_s a_sha256 b_sha256 c_sha256 implementation plans cached'.split()
PANEL_KEYS=['family','layout',*ABI]
ROW_KEYS=[*PANEL_KEYS,'method','role','mode','seed','round','position',*QUAL,'passed','reason']
HERE=Path(__file__).resolve().parent

def need(test,message):
    if not test:raise ValueError(message)

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def writer(path,fields):
    f=path.open('w',newline='');wr=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');wr.writeheader()
    def emit(row):wr.writerow(row);f.flush()
    return emit

def digest_string(value):
    return re.fullmatch(r"[0-9a-f]{64}",value) is not None

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def mk(name,role,worker,plug='public',trace='',traceplug='public'):
    return dict(zip(METHOD_KEYS,[name,role,str(worker),str(plug),str(trace),str(traceplug)]))

def methods(vendor,work,out):
    native='mkl' if platform.machine()=='x86_64' else 'armpl'
    ordinary=work/'released-hk15/ordinary.worker'
    labels=[mk(n,r,work/n/'ordinary.worker',trace=work/n/'trace.worker') for n,r in [('released-hk15','baseline'),('ragged-hk15','candidate')]]
    vr=work/'vendor-release.csv';built={r['method']:r for r in read(vr)} if vr.exists() else {}
    for p in ['openblas',native]:
        for level in ['aligned-two','two']:
            n=f'release-{p}-{level}';r=built.get(n,{})
            labels.append(mk(n,'vendor-assisted',r.get('worker',''),trace=r.get('trace_worker','')))
    peers=['openblas',native,*(['mkl-jit','aocl'] if native=='mkl' else []),'fastmmw-768','fastmmw-1536','blasfeo']
    if native=='mkl': peers+=['tblis-ab-1','tblis-ab-2']
    for n in peers:
        trace=(vendor/(n+'.trace.so')) if n.startswith('fastmmw') else ''
        labels.append(mk(n,'independent-oneshot',ordinary,vendor/(n+'.so'),ordinary if trace else '',trace))
    for policy,role in [('oneshot','independent-oneshot'),('cached-plan','independent-retained')]:
        labels.append(mk('onednn-'+policy,role,ordinary,vendor/'onednn'/('onednn-f32-'+policy+'.so')))
    (out/'sources').mkdir(exist_ok=True)
    file=writer(out/'sources/methods.csv',['method','role','available','reason','worker_sha256','adapter_sha256','trace_worker_sha256','trace_adapter_sha256'])
    for obj in labels:
        valid=Path(obj['worker']).is_file() and (obj['plugin']=='public' or Path(obj['plugin']).is_file());why=''
        if obj['method'].startswith('onednn-') and not (vendor/'onednn/qualified').is_file():valid=False;why='native strict qualifications unavailable'
        if obj['method'].startswith('tblis') and not ((vendor/'tblis-status.txt').exists() and (vendor/'tblis-status.txt').read_text().startswith('qualified:')):valid=False;why='native Haswell backend not qualified'
        if obj['method'].startswith(('fastmmw-','release-')) or obj['role'] in ('baseline','candidate'):
            if not Path(obj['trace_worker']).is_file() or (obj['trace_plugin']!='public' and not Path(obj['trace_plugin']).is_file()):valid=False;why='paired trace product missing'
        if not valid and not why:why='build or installed evidence unavailable'
        obj.update(available=valid,reason=why)
        def digest(k):return sha(Path(obj[k])) if Path(obj[k]).is_file() else ''
        file(dict(method=obj['method'],role=obj['role'],available=int(valid),reason=why,worker_sha256=digest('worker'),adapter_sha256=digest('plugin'),trace_worker_sha256=digest('trace_worker'),trace_adapter_sha256=digest('trace_plugin')))
    return labels

def line(text,tag):
    vals=[x.split(',')[1:] for x in text.splitlines() if x.startswith(tag+',')];need(len(vals)==1,'missing or duplicate '+tag+' evidence');return vals[0]

def engine(vendor,work,out,cpu):
    need(os.sched_getaffinity(0)=={cpu},'controller not pinned');pan=read(out/'panel.csv')
    need(sha(out/'panel.csv')=='6d9d898375f2f57eea76ae42d6ec04b0792d5440967da260b52c47b59c77dc42','panel changed')
    roster=methods(vendor,work,out);files=work/'goldens';page=os.sysconf('SC_PAGESIZE')
    acc=writer(out/'accuracy.csv',ROW_KEYS);warm=writer(out/'timings.csv',ROW_KEYS)
    routes=writer(out/'routes.csv',['family','method','role','verified','route','ordinary_sha256','trace_sha256','receipt','reason'])
    gold=writer(out/'goldens.csv',['family','mode','seed','a_sha256','b_sha256','reference_sha256','bytes','checked','scalar_spots','max_spot_error','cpu','threads'])
    qualified=writer(out/'qualified.csv',['family','method','role','qualified','reason'])
    refs={};ok={};first={};issues={}; traces={}
    def call(exe,plug,p,mode,seed,count,env):
        result=subprocess.run([exe,plug,mode,str(seed),str(count),*(p[k] for k in ABI)],env=dict(os.environ,**env),capture_output=True,text=True,timeout=140)
        if result.returncode:raise ValueError(f'exit={result.returncode} '+result.stderr[-450:].replace('\n',' '))
        return result.stdout
    for p in pan:
        fam=p['family'];m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc=(int(p[x]) for x in ABI)
        for mode,seed in FIXTURES:
            key=fam,mode,seed;path=files/f'{fam}-{mode}-{seed}.f64';need(not path.exists(),'golden reuse')
            s=call(roster[0]['worker'],str(vendor/'openblas.so'),p,mode,seed,0,{'RAGGED_WRITE_REFERENCE':str(path)})
            aa,bb,checked,spots,err,c,t=line(s,'GOLD');need(int(checked)==m*n and int(spots)==23 and int(c)==cpu and int(t)==1 and 0<=float(err)<=1e-9 and digest_string(aa) and digest_string(bb) and path.stat().st_size==8*m*n,'incomplete gold')
            refs[key]=(path,aa,bb);gold(dict(family=fam,mode=mode,seed=seed,a_sha256=aa,b_sha256=bb,reference_sha256=sha(path),bytes=path.stat().st_size,checked=checked,scalar_spots=spots,max_spot_error=err,cpu=c,threads=t))
    def observe(p,obj,mode,seed,count,trace=False):
        m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc=(int(p[x]) for x in ABI)
        parent=((cr+m+2)*lc*4+page-1)//page*page//4;path,aa,bb=refs[p['family'],mode,seed]
        s=call(obj['trace_worker' if trace else 'worker'],obj['trace_plugin' if trace else 'plugin'],p,mode,seed,count,{'RAGGED_REFERENCE':str(path)})
        v=line(s,'VIEW');h=line(s,'RAGGED');need(len(v)==13 and len(h)==5,'worker ABI mismatch')
        checked,guards,am,bm,cm,logical,nt=map(int,v[:7]);need((checked,guards,logical,nt)==(m*n,parent-m*n,cpu,1),'CPU count, thread or guards differ')
        need((am,bm,cm)==tuple(4*i%64 for i in (ar*la+ac,br*lb+bc,cr*lc+cc)) and tuple(map(int,v[10:]))==(la,lb,lc),'caller leading dimensions or addresses differ')
        need(h[0]==aa and h[1]==bb and digest_string(h[2]),'input SHA or output SHA mismatch')
        err,ft,wt=map(float,v[7:10]); cf,cw=map(float,h[3:]);need(all(math.isfinite(x) and x>0 for x in ([ft,cf,wt,cw] if count else [ft,cf])),'invalid timer')
        impl='';plans='';cached='';name=obj['method']
        if name.startswith('onednn-'):
            parts=line(s,'DNN');impl=','.join(parts[:-2]);plans,cached=map(int,parts[-2:]);need(impl and not impl.lower().startswith('ref') and cached==int(name.endswith('cached-plan')) and plans==(1 if cached else 1+count),'strict oneDNN executable/plan count mismatch')
        r=dict(zip(QUAL,[checked,guards,am,bm,cm,logical,nt,err,ft,wt,cf,cw,h[0],h[1],h[2],impl,plans,cached]));return r,s
    def row(p,obj,mode,seed,j,pos,emit):
        base=dict(p,method=obj['method'],role=obj['role'],mode=mode,seed=seed,round=j,position=pos)
        try:
            receipt,_=observe(p,obj,mode,seed,int(j!=0));err=receipt['max_abs'];passed=math.isfinite(err) and 0<=err<=.002
            reason='' if passed else f'FP64 max_abs={err} exceeds 0.002 or nonfinite';emit(dict(base,**receipt,passed=int(passed),reason=reason));return passed,reason,receipt
        except (ValueError,OSError,subprocess.TimeoutExpired) as ex:
            emit(dict(base,passed=0,reason=str(ex)));return False,str(ex),{}
    for p in pan:
        fam=p['family'];m,n,k=map(int,(p['m'],p['n'],p['k']))
        print('Qualifying',fam,flush=True)
        for obj in roster:
            name=obj['method'];key=fam,name;issues[key]=[]
            if not obj['available']:issues[key].append(obj['reason']);ok[key]=False;continue
            for mode,seed in FIXTURES:
                yes,msg,r=row(p,obj,mode,seed,0,0,acc)
                if not yes:issues[key].append(msg)
                first[key,mode,seed]=r.get('c_sha256','')
            ok[key]=not issues[key]
            if obj['trace_worker']:
                route='';why='';proof={};trace_digest='';ordinary=first[key,'random',SEEDS[0]]
                try:
                    r,s=observe(p,obj,'random',SEEDS[0],0,True);trace_digest=r['c_sha256'];need(0<=r['max_abs']<=.002 and trace_digest==ordinary,'ordinary and trace full output or FP64 differs')
                    if name.startswith('release-'):
                        provider=name.split('-')[1];proof=vendor_route.validate_release(p,s,provider,method=name);route=proof['route']
                    elif name.startswith('fastmmw-'):
                        cutoff=int(name.split('-')[-1]);split=1;leaves=1
                        while (min(m,n,k)+split-1)//split>cutoff:split*=2;leaves*=7
                        leaf,pad=map(int,line(s,'FAST'));need(leaf==leaves and pad==int(any(d%split for d in (m,n,k))),'FastMMW route differs')
                        proof=dict(cutoff=cutoff,leaves=leaf,padded=pad,core_multiple=split);route='upstream-serial-winograd'
                    else:
                        proof={a:int(b) for a,b in (s.split('=') for s in line(s,'HK'))};rem=(m%2,n%3,k%3);parent=int(name=='ragged-hk15' or not any(rem));finish=int(parent and any(rem));mc=m-m%2;nc=n-n%3
                        child_m,child_n,child_k=mc//2,nc//3,k//3
                        surface=child_m*child_n+child_m*child_k+child_n*child_k
                        lower=int(parent and 960*surface<=3*child_m*child_n*child_k<1024*surface)
                        expected=dict(public_calls=1,parents=parent,leaves=15*parent,lowered=lower,fused_leaves=15*parent,finishes=finish,right=int(finish and n%3!=0),bottom=int(finish and m%2!=0),k_terms=finish*(k%3),core_checked=(mc*nc if finish and k%3 else 0),args_ok=1)
                        need(all(proof.get(k)==v for k,v in expected.items()),'Observed HK/edge coordinates or arithmetic differ '+str(proof));route='hk15-ragged' if finish else ('hk15-divisible' if parent else 'released-fallback')
                except (ValueError,OSError,subprocess.TimeoutExpired) as ex:why=str(ex);ok[key]=False;issues[key].append(why)
                routes(dict(family=fam,method=name,role=obj['role'],verified=int(not why),route=route,ordinary_sha256=ordinary,trace_sha256=trace_digest,receipt=json.dumps(proof,separators=(',',':')),reason=why))
        if fam=='repeat-divisible-winner':
            for mode,seed in FIXTURES:
                a=first.get(((fam,'released-hk15'),mode,seed));b=first.get(((fam,'ragged-hk15'),mode,seed));
                if not a or a!=b:ok[fam,'ragged-hk15']=False;issues[fam,'ragged-hk15'].append('Divisible output differs from original')
    for ib,seed in enumerate(SEEDS):
        for j in range(3):
            shapes=pan[j:]+pan[:j]
            if ib:shapes=list(reversed(shapes))
            for p in shapes:
                fam=p['family'];viable=[obj for obj in roster if ok[fam,obj['method']]]
                if not viable:continue
                off=(j+pan.index(p))%len(viable);order=viable[off:]+viable[:off]
                if ib:order=order[::-1]
                for pos,obj in enumerate(order,1):
                    key=fam,obj['method'];yes,msg,r=row(p,obj,'random',seed,j+1,pos,warm)
                    if not yes:issues[key].append(msg)
                print('Warm',seed,j+1,fam,flush=True)
    for p in pan:
        for obj in roster:
            key=p['family'],obj['method'];qualified(dict(family=key[0],method=key[1],role=obj['role'],qualified=int(ok[key] and not issues[key]),reason=' | '.join(dict.fromkeys(issues[key]))))

if __name__=='__main__':engine(*(Path(x).resolve() for x in sys.argv[1:4]),int(sys.argv[4]))
