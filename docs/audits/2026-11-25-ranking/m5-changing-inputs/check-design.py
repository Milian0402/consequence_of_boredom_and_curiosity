"""Offline synthetic schedule and corruption regression. No multiplication or machine timings."""
import itertools
import math
from pathlib import Path
import tempfile
import score


def fixtures():
    samples=[];accuracy=[];pools=[]
    for f,shape in score.PANEL.items():
        m,n,k=shape;size=4*(m*k+k*n+m*n);slots=(score.THRESHOLD+size-1)//size;pages=(slots*size+16383)//16384
        for block,seed in score.SEEDS.items():
            base=dict(family=f,m=m,n=n,k=k,pad=0,block=block,seed=seed,slots=slots)
            pools.append(dict(base,logical_bytes=slots*size,arena_bytes=pages*16384,page_bytes=16384,pages=pages,
                resident_pre=pages,resident_post=pages,epoch_min_resident=pages,threads=1,readonly_ab=1,shared_method_output=1,same_alias=1,
                unique_a=slots,unique_b=slots,unique_c=slots))
            for label in score.METHODS:accuracy.append(dict(base,method=label,checked=m*n*slots,max_abs=.0001,guards_ok=1))
            labels=list(score.METHODS)
            for mode in score.MODES:
                for round_idx in range(11):
                    # Alternate ends of the cyclic list, independently expressed from the verifier's closed form.
                    indices=[0,1,10,2,9,3,8,4,7,5,6]
                    if block==2:indices=indices[::-1]
                    order=[labels[(i+round_idx)%11] for i in indices]
                    start=(seed+17*round_idx)%slots
                    step=0 if mode=='hot' else next(j%slots for j in itertools.count((seed+2*round_idx+1)%slots) if math.gcd(j%slots,slots)==1)
                    for pos,label in enumerate(order,1):
                        ns={'A':12,'B':12,'D':8,'Accelerate':10,'OpenBLAS':16}.get(label,23)
                        if mode=='rotate':ns*=2 if label=='Accelerate' else 1.5
                        samples.append(dict(base,mode=mode,round=round_idx+1,position=pos,method=label,calls=slots,
                            start_slot=start,step=step,wall_ns=ns*slots,caller_ns=ns*slots*.9,threads_before=1,threads_after=1,guards_ok=1))
    # The native driver visits the second block in reverse shape order and alternates
    # which mode is measured first for each geometry and seed.
    def process(r):
        ordinal=list(score.PANEL).index(r['family'])
        return (r['block'], ordinal if r['block']==1 else -ordinal)
    accuracy.sort(key=lambda r:(*process(r), score.METHODS.index(r['method'])))
    pools.sort(key=process)
    def chronology(r):
        ordinal=list(score.PANEL).index(r['family'])
        mode_order=(score.MODES.index(r['mode'])+ordinal+r['block']-1)%2
        return (*process(r), mode_order, r['round'], r['position'])
    samples.sort(key=chronology)
    return samples,accuracy,pools


def main():
    samples,accuracy,pools=fixtures();rows={'samples.csv':samples,'accuracy.csv':accuracy,'pool.csv':pools}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for file,records in rows.items():score.dump_csv(root/file,records)
        results=score.run(root,root);groups=results['groups']
        for g in groups:
            expected={'hot':10/8,'rotate':20/12}[g['mode']]
            assert abs(g['gm_peer_over_shipped']-expected)<1e-12 and g['wins']==g['shapes']
        assert results['qualification']['timing_batches']==4356
        for file,field,value in [('pool.csv','same_alias',0),('accuracy.csv','max_abs','nan'),('samples.csv','step',12),('samples.csv','position',10)]:
            records=rows[file];old=records[0][field];records[0][field]=value;score.dump_csv(root/file,records)
            try:score.run(root,root)
            except ValueError:pass
            else:raise RuntimeError('Failed to reject corrupted '+field)
            records[0][field]=old;score.dump_csv(root/file,records)
        # Preserve every row and valid within-epoch trial key but swap the first two epochs.
        wrong=samples[121:242]+samples[:121]+samples[242:]
        score.dump_csv(root/'samples.csv',wrong)
        try:score.run(root,root)
        except ValueError:pass
        else:raise RuntimeError('Accepted swapped hot/rotate epochs')
        print('Checked complete synthetic design and five corruptions, including reversed mode order.')

if __name__=='__main__':main()
