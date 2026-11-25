"""Verify every ACL trace event against the ordinary composite's actual mode and input."""
from collections import Counter
import json
from pathlib import Path
import sys


def audit(root):
    read=lambda name:json.loads((root/name/'scores.json').read_text())
    ordinary,trace=read('ordinary'),read('trace')
    field=('case','seed','mode','m','n','k','output_sha256','returncode','route','primitive_calls','reason','plans')
    witness=[dict(x.split('=',1) for x in line.split(',')[1:]) for line in
             (root/'trace.log').read_text().splitlines() if line.startswith('ACL_BF16_KERNEL,')]
    events=iter(witness)
    n_mixed=n_fallback=hits=0
    for a,b in zip(ordinary['cases'],trace['cases'],strict=True):
        assert all(a[k]==b[k] for k in field) and a['returncode']==0
        if a['route']!=1:
            n_fallback+=a['route']==2
            continue
        n_mixed+=1
        m,n,k=a['m'],a['n'],a['k']
        q=int(a['mode'])
        depths=[3*k,2*k,k] if q==16 else [k]*q
        assert len(depths)==a['primitive_calls']
        assert all(p['native']==1 and p['implementation']=='a64_interleaved_bf16fp32_mmla_8x12' for p in a['plans'])
        assert [p['k'] for p in a['plans']]==(depths if q==16 else [k])
        for d in depths:
            e=next(events)
            ab,bb,kp=(m+7)//8,(n+11)//12,((d+3)//4)*4
            count=((ab+7)//8)*((bb+7)//8)
            expect=dict(m=m,n=n,k=d,ablocks=ab,bblocks=bb,kp=kp,calls=count,
                        abytes=2*ab*8*kp,bbytes=2*bb*12*kp,cbytes=min(8,ab)*min(8,bb)*8*12*4,allocations=2)
            assert e['name']=='a64_interleaved_bf16fp32_mmla_8x12'
            assert all(int(e[k])==v for k,v in expect.items()),(a['case'],a['mode'],e)
            hits+=count
    assert next(events,None) is None
    assert ordinary['verdicts']==trace['verdicts']
    summary=dict(emulation='QEMU -cpu neoverse-n2',physical_host_qualification=False,
                 registration=ordinary['registration'],ordinary_dso=ordinary['library_sha256'],
                 trace_dso=trace['library_sha256'],matrix_methods=len(ordinary['cases']),mixed=n_mixed,
                 rescues=n_fallback,trace_events=len(witness),confirmed_kernel_calls=hits,
                 counts={m:dict(normal_gamma_exceeded=sum(x['gamma_exceeded'] for x in ordinary['cases']
                            if x['category']=='normal' and x['mode']==m),verdict=v) for m,v in ordinary['verdicts'].items()})
    (root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Verified',len(ordinary['cases']),'matrices and',hits,'ACL kernel invocations.')


if __name__=='__main__':
    audit(Path(sys.argv[1]))
