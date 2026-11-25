import csv
import subprocess
import sys
from pathlib import Path

build, vendors, out = map(Path, sys.argv[1:4])
cpu = int(sys.argv[4])
methods = ['default', 'stream', 'packed', 'openblas', 'mkl', 'mkl-jit', 'aocl']
seeds = [73194513, 38641729]
with (out/'panel.csv').open() as f:
    shapes = list(csv.DictReader(f))
fields = 'method family layout mode seed round place calls checked guards a_mod64 b_mod64 c_mod64 cpu threads error first_s per_call_s'.split()
raw = (out/'raw.csv').open('w')
writer = csv.DictWriter(raw, fields, lineterminator='\n')
writer.writeheader()
routes = (out/'routes.csv').open('w')
route_fields = 'method family layout public_calls avx2 avx512 packed_avx2 packed_avx512 args_ok alignment_ok refused'.split()
route_writer = csv.DictWriter(routes,route_fields,lineterminator='\n')
route_writer.writeheader()
log = (out/'worker.log').open('w')

def invoke(s, method, mode, seed, calls, trace=False):
    product = method in methods[:3]
    worker = build/f"{method if product else 'default'}.{'trace' if trace else 'worker'}"
    plugin = 'public' if product else str(vendors/f'{method}.so')
    args = [worker,plugin,mode,str(seed),str(calls),*(s[key] for key in 'm n k lda ldb ldc ar ac br bc cr cc'.split())]
    command = subprocess.run(args,capture_output=True,text=True)
    log.write(' '.join(map(str,args))+'\n'+command.stdout+command.stderr); log.flush()
    if command.returncode:
        raise RuntimeError(f"worker failed {method} {s['family']} {s['layout']} {mode}: {command.stderr[-600:]}")
    lines = command.stdout.splitlines()
    row = next(x for x in lines if x.startswith('VIEW,')).split(',')[1:]
    data = dict(zip(fields[8:],row,strict=True))
    if int(data['cpu']) != cpu or int(data['threads']) != 1:
        raise ValueError('unpinned worker')
    if trace:
        detail = next(x for x in lines if x.startswith('FRONTIER,')).split(',')[1:]
        route_writer.writerow(dict(method=method,family=s['family'],layout=s['layout'],**dict(x.split('=') for x in detail)))
        routes.flush()
    return data

def record(s,method,mode,seed,r,place,calls):
    data=invoke(s,method,mode,seed,calls)
    writer.writerow(dict(method=method,family=s['family'],layout=s['layout'],mode=mode,seed=seed,round=r,place=place,calls=calls,**data))
    raw.flush()

failures={}
def operation(s,method,fn,*args):
    key=(s['family'],s['layout'],method)
    if key in failures: return
    try:
        fn(s,method,*args)
    except (ValueError,RuntimeError,StopIteration) as exc:
        failures[key]=str(exc)
        with (out/'failures.csv').open('a') as f:
            csv.writer(f).writerow([*key,str(exc)])

for s in shapes:
    for method in ['stream','packed']:
        operation(s,method,invoke,'random',seeds[0],0,True)
    for method in methods:
        for mode,seed in [('random',seeds[0]),('random',seeds[1]),('cancel',seeds[0]),('scaled',seeds[1])]:
            operation(s,method,record,mode,seed,-1,-1,0)

for r in range(6):
    # The second group reverses the complete layout and method sequence for
    # each corresponding round, including the aligned/cropped visit order.
    order=shapes[r%3:]+shapes[:r%3]
    if r>=3: order=order[::-1]
    for s in order:
        j=(shapes.index(s)+2*(r%3))%len(methods)
        roster=methods[j:]+methods[:j]
        if r>=3: roster=roster[::-1]
        for place,method in enumerate(roster):
            operation(s,method,record,'random',seeds[r//3],r,place,16)
