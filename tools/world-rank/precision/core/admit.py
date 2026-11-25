"""Join exact arithmetic admissions to native per-depth, per-backend execution witnesses."""
import argparse,csv,ctypes,hashlib,json,os,platform,subprocess,sys
from pathlib import Path

MODES={'mixed1':1,'mixed3':3,'mixed6':6,'mixed6-bands':16,'mixed9':9}
COORDS='m n k lda ldb ldc ar ac br bc cr cc'.split()
ROOT=Path(__file__).resolve().parents[1]

def put(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def capture(args,log,timeout=600):
    try:
        proc=subprocess.run(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        log.write_bytes(proc.stdout);return proc.returncode
    except (OSError,subprocess.TimeoutExpired) as e:
        log.write_text(str(e)+'\n');return 125

def main(work,out,cpu):
    assert os.sched_getaffinity(0)=={cpu}
    candidates=work/'candidates'; accuracy=out/'precision-accuracy';proof=out/'native-bf16';accuracy.mkdir(exist_ok=True);proof.mkdir(exist_ok=True)
    panel=list(csv.DictReader((out/'panel.csv').open())); family={r['family']:{} for r in panel}; registrations={}
    def one_family(prefix,backend):
        fused=prefix=='fused'
        names={'fused6-amx':6} if fused else {prefix+s:m for s,m in MODES.items()}
        core=candidates/('fused6-amx.so' if fused else ('arm-core.so' if prefix else 'onednn-core.so'))
        tag='fused-amx' if fused else 'arm' if prefix else 'onednn'
        qualified=accuracy/tag;qualified.mkdir(exist_ok=True)
        usable=core.is_file();why='' if usable else 'registered provider binary unavailable'
        if usable and fused:
            probe=subprocess.run([sys.executable,'-c','import ctypes,sys; sys.exit(int(ctypes.CDLL(sys.argv[1]).amx_fused6_init()!=0))',str(core)],check=False)
            if probe.returncode:usable=False;why='AMX BF16 tile CPU/OS not available'
        verdicts={}
        if usable:
            err=capture([sys.executable,'-B',ROOT/'accuracy/evaluate.py','--library',core,'--modes','6' if fused else '1,3,6,9,16','--label',tag,'--out',qualified],qualified/'test.log')
            if err==0 and (qualified/'verdicts.json').exists():verdicts=json.loads((qualified/'verdicts.json').read_text())
            else:why=f'adversarial evaluator exit {err}'
        for name,mode in names.items():
            rec=verdicts.get(str(mode),{})
            registrations[name]=dict(available=usable and (candidates/(name+'.so')).is_file(),accuracy_qualified=bool(mode not in (1,3) and rec.get('matched_direct_accuracy')),category='precision-reference' if mode in (1,3) else 'strict-candidate',reason=why or ('complete original and direct-quality gates' if rec.get('matched_direct_accuracy') else 'outside registered FP32 accuracy gates'),accuracy_report=f'precision-accuracy/{tag}/verdicts.json' if verdicts else None)
        for g in panel:
            key=g['family'];m,n,k=map(int,(g['m'],g['n'],g['k']));cache={}
            if usable and not fused:
                for d in (k,2*k,3*k):
                    path=proof/f'{tag}-{key}-k{d}.json';log=path.with_suffix('.log')
                    src=ROOT/('acl-backend' if prefix else 'backend')/'qualify.py'
                    exe=backend if prefix else backend/'bf16-probe'
                    status=capture([sys.executable,'-B',src,exe,path,m,n,d,d,n,n],log)
                    raw=json.loads(path.read_text()) if path.is_file() else {}
                    cache[d]=dict(eligible=raw.get('eligible') is True and status==0,implementation=raw.get('implementation',''),route=raw.get('route',''),reason=raw.get('reason',f'witness exit {status}'),proof='native-bf16/'+path.name)
            for name,mode in names.items():
                depths=[] if fused else [3*k,2*k,k] if mode==16 else [k]
                rec=dict(eligible=False,route='',impls=[],reason=why or 'native path not proven',required_depths=depths,reported_native=[])
                if usable and not fused:
                    x=[cache[d] for d in depths];rec.update(eligible=all(v['eligible'] for v in x),route=' / '.join(v['route'] for v in x),impls=[v['implementation'] for v in x],reported_native=[v['eligible'] for v in x],reason=' / '.join(v['reason'] for v in x),witnesses=[v['proof'] for v in x]);
                if usable and fused:
                    path=proof/f'{tag}-{key}.json';log=path.with_suffix('.log')
                    status=capture([work/'fused-probe',candidates/'fused6-amx.trace.so',*(g[c] for c in COORDS)],log)
                    try:r=json.loads(next(x for x in log.read_text().splitlines() if x.startswith('{')))
                    except (ValueError,StopIteration):r={'eligible':False,'reason':f'AMX route witness exit {status}'}
                    put(path,r);rec.update(eligible=status==0 and r.get('eligible') is True,route=r.get('route',''),reason=r['reason'],witnesses=['native-bf16/'+path.name]);
                family[key][name]=rec
    one_family('',work/'onednn-bf16')
    if platform.machine()=='aarch64':one_family('arm-',work/'arm-bf16')
    else:one_family('fused',work)
    for f,data in [('admission.json',registrations),('native-routes.json',family)]:
        put(work/f,data);put(out/f,data)

if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),int(sys.argv[3]))
