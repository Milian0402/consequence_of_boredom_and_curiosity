"""Bind the two completed rotation measurements to frozen files and their process order."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from score import METHODS, MODES, PANEL, SEEDS, process_order, read_csv, require

FREEZES = {
    "primary": ("freeze.sha256", "ad542d3932cc263fd910c8eb9abbf42a89a59e70cb14f5fe01cfd731caf31c38"),
    "wired": ("wired-freeze.sha256", "472116009ea951a49c39aed99440e2df02effa8d0d5308e6d0a61d889c7272b4")}
WIRED_LAUNCH = "f82f843ee16753f5a3064016742de958d4d05d054c4b687bfce56f6506f18970"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bind(root, raw, vintage, binary_root=None):
    name, signature = FREEZES[vintage]
    require(sha(root/name)==signature, 'Freeze manifest changed')
    manifest = dict(line.split('  ',1)[::-1] for line in (root/name).read_text().splitlines())
    executable = 'wired' if vintage=='wired' else 'worker'
    for file,digest in manifest.items():
        if file==executable and not (root/file).exists():
            continue
        require(sha(root/file)==digest, 'Unbound source or executable '+file)
    if vintage=='wired':
        require(sha(root/'run-wired.py')==WIRED_LAUNCH,'Wired launcher differs from captured executable recipe')
    require(sha(raw/'panel.csv')==manifest['panel.csv'], 'Raw panel differs from frozen panel')
    methods=json.loads((root/'methods.json').read_text())
    require(tuple(methods)==METHODS and methods['A']==methods['B']!=methods['D'] and
            all(re.fullmatch('[0-9a-f]{64}',h) for h in methods.values()), 'Frozen method roster and alias')
    if binary_root:
        h=binary_root.resolve(); apple=h.parent; world=apple.parents[1]; work=world.parent
        packages={'A':apple/'matched/reproduction/incumbent.dylib','B':apple/'matched/reproduction/incumbent.dylib',
            'D':apple/'compact2/shipped.dylib','Accelerate':work/'ranking/apple-single.dylib',
            'OpenBLAS':world/'openblas/openblas-mac.dylib','ArmPL':world/'armpl-mac/armpl-mtmm.dylib',
            'MaxMulSK':work/'ranking/maxmulsk.dylib','MpGEMM':work/'cob/experiments-local/mpgemm/mpgemm.dylib',
            'prepared-LIBXSMM':world/'libxsmm/upstream/lib/libxsmm.a',
            'KleidiAI-fixed':world/'layout-wave/kleidiai/build/kleidiai-fixed.dylib',
            'KleidiAI-elastic':world/'layout-wave/kleidiai/build/kleidiai-elastic.dylib'}
        require(sha(h/executable)==manifest[executable], 'Local timed executable differs from frozen binary')
        for method,path in packages.items():
            require(sha(path)==methods[method], 'Linked static archive or ordinary-call library changed for '+method)
    expected=process_order(); marks=[]; jits=[]; opens=[]
    for line in (raw/'run.log').read_text().splitlines():
        part=line.split()
        if part and part[0] in ('starting','finished'):
            require(len(part)==4 and float(part[3])>0, 'Malformed process marker')
            marks.append((part[0],int(part[1]),part[2])); continue
        if line.startswith('jit '):
            require(len(part)==3 and int(part[2])>0, 'Empty JIT code')
            jits.append((part[1],int(part[2])))
        if line.startswith('OpenBLAS core='):
            require('threads=1' in line, 'OpenBLAS thread count'); opens.append(line)
    require(marks==[(event,b,f) for f,b in expected for event in ('starting','finished')] and
            [f for f,n in jits]==[f for f,b in expected] and len(opens)==len(expected),
            'Missing/different worker process, JIT, or OpenBLAS record')
    pool=read_csv(raw/'pool.csv'); miss=[int(r['pages'])-int(r['epoch_min_resident']) for r in pool]
    faults=None
    if vintage=='wired':
        rows=read_csv(raw/'faults.csv')
        desired=[(f,b,MODES[(i+list(PANEL).index(f)+b-1)%2]) for f,b in expected for i in range(2)]
        require([(r['family'],int(r['block']),r['mode']) for r in rows]==desired, 'Fault record chronology')
        require(all(r['locked']=='1' and int(r['seed'])==SEEDS[int(r['block'])] and
                    min(int(r['major_faults']),int(r['minor_faults']))>=0 for r in rows),'Missing locked/fault record')
        require(max(miss)==0 and all(int(r['resident_pre'])==int(r['resident_post'])==int(r['pages']) for r in pool),
                'Wired arena pages missing at an observation')
        faults={k:sum(int(r[k]) for r in rows) for k in ('major_faults','minor_faults')}
    return dict(vintage=vintage,manifest_sha256=signature,protocol_sha256=manifest['protocol.md'],
        panel_sha256=manifest['panel.csv'],worker_sha256=manifest[executable],methods_sha256=methods,
        source_hashes=manifest,raw_sha256={p.name:sha(p) for p in raw.iterdir() if p.suffix in ('.csv','.log')},
        local_worker_and_libraries_checked=bool(binary_root),finished_processes=len(expected),prepared_jit_code_bytes=jits,
        registered_epoch_order_verified=True,arena_pools_with_nonresident_snapshot=sum(bool(x) for x in miss),
        maximum_missing_pages_at_snapshot=max(miss),faults=faults)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('frozen_sources',type=Path)
    p.add_argument('raw',type=Path);p.add_argument('vintage',choices=tuple(FREEZES))
    p.add_argument('--native-binary-root',type=Path)
    args=p.parse_args()
    print(json.dumps(bind(args.frozen_sources,args.raw,args.vintage,args.native_binary_root),indent=2))
