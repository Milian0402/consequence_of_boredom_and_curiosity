"""Run the full 2^6 design using a new guarded single-thread process per measurement."""
import csv
import os
from pathlib import Path
import subprocess
import sys

work,vendor,out=map(Path,sys.argv[1:4])
cpu=int(sys.argv[4])
if os.sched_getaffinity(0)!={cpu}: raise ValueError("factorial study must run on one logical CPU")
with (out/"panel.csv").open() as f: panel=list(csv.DictReader(f))
families=list(dict.fromkeys(r["family"] for r in panel)); cells={(r["family"],int(r["mask"])):r for r in panel}
if families!=["repeat-win-deeper","repeat-win-ragged","fresh-short"] or len(cells)!=len(panel) or len(cells)!=192:
    raise ValueError("Incomplete factorial panel")
methods=["stream","mkl"];seeds=[73194513,38641729]
raw_fields="method family mask mode seed round place sequence calls checked guards a_mod64 b_mod64 c_mod64 cpu threads error first_s per_call_s a_hash b_hash c_hash ao_observed bo_observed co_observed lda_observed ldb_observed ldc_observed".split()
abi="m n k lda ldb ldc ao bo co".split()
metric_fields=raw_fields[9:]
stderr=(out/"worker-diagnostics.txt").open("w")
def run(row,method,mode,seed,count,trace=False):
    product="public" if method=="stream" else str(vendor/"mkl.so")
    executable=work/("stream.trace.worker" if trace else "stream.worker")
    result=subprocess.run([executable,product,mode,str(seed),str(count),*(row[k] for k in abi)],
                          stdout=subprocess.PIPE,stderr=stderr,text=True,check=True)
    lines=result.stdout.splitlines()
    values=next(x for x in lines if x.startswith("FACTOR,")).split(",")[1:]
    if len(values)!=len(metric_fields): raise ValueError("Incomplete output record")
    rec=dict(zip(metric_fields,values))
    for key in ("ao","bo","co","lda","ldb","ldc"):
        if rec[key+"_observed"]!=row[key]:raise ValueError(("view differs",key,row,rec))
    if trace:
        proof=dict(s.split("=",1) for s in next(x for x in lines if x.startswith("FRONTIER,")).split(",")[1:])
        return rec,proof
    return rec
with (out/"raw.csv").open("w",newline="") as raw,(out/"routes.csv").open("w",newline="") as tracefile:
    w=csv.DictWriter(raw,fieldnames=raw_fields,lineterminator="\n");w.writeheader()
    trace=csv.DictWriter(tracefile,fieldnames="family mask cpu public_calls avx2 avx512 packed_avx2 packed_avx512 args_ok alignment_ok refused checked a_mod64 b_mod64 c_mod64 a_hash b_hash c_hash".split(),lineterminator="\n");trace.writeheader()
    def save(row,method,mode,seed,j,place,sequence,calls):
        rec=run(row,method,mode,seed,calls)
        w.writerow(dict(rec,method=method,family=row["family"],mask=row["mask"],mode=mode,seed=seed,round=j,place=place,sequence=sequence,calls=calls))
        return rec
    for row in panel:
        for mode,seed in (("random",seeds[0]),("random",seeds[1]),("cancel",seeds[0]),("scaled",seeds[1])):
            for method in methods:
                rec=save(row,method,mode,seed,-1,-1,-1,0)
                if (method,mode,seed)==("stream","random",seeds[0]):
                    observed,proof=run(row,method,mode,seed,0,True)
                    for key in ("checked","guards","a_mod64","b_mod64","c_mod64","cpu","threads","error","a_hash","b_hash","c_hash"):
                        if rec[key]!=observed[key]:raise ValueError(("trace changed result",row,key))
                    trace.writerow(dict(family=row["family"],mask=row["mask"],**proof,**{k:rec[k] for k in ("checked","cpu","a_mod64","b_mod64","c_mod64","a_hash","b_hash","c_hash")}))
        raw.flush();tracefile.flush()
    for j in range(6):
        block,t=divmod(j,3);order=[]
        for slot in range(64):
            for h in range(3):
                i=(h+t)%3
                mask=(37*slot+11*i+17*t)%64
                order.append((slot,i,cells[families[i],mask]))
        if block:order.reverse()
        for sequence,(slot,i,row) in enumerate(order):
            first=(i+slot+t+block)%2
            for place,index in enumerate((first,1-first)):
                save(row,methods[index],"random",seeds[block],j,place,sequence,16)
        raw.flush()
stderr.close()
