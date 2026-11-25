"""Run the diagnostic lifecycle and separate plan-only clocks on the registered physical views."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

panel,vendor,build,out=[Path(x).resolve() for x in sys.argv[1:]]
out.mkdir(parents=True,exist_ok=True)
affinity=os.sched_getaffinity(0)
if len(affinity)!=1: raise ValueError("one pinned CPU required")
cpu=min(affinity)
here=Path(__file__).resolve().parent
methods=("mkl-handle-oneshot","mkl-handle-prepared")
geometry="m n k lda ldb ldc ar ac br bc cr cc".split()
fields_route="""method family layout cpu ar ac br bc cr cc m n k lda ldb ldc stage creates gets calls destroys
create_status execution_status destroy_status ordinary_status trace_status status_name order transa transb alpha beta checked
guards ordinary_error trace_error callable callable_owner callable_permissions same_handle same_kernel operands_checked
a_mod64 b_mod64 c_mod64 threads passed""".split()
def save(name,rows,fields):
    with (out/name).open("w",newline="") as target:
        w=csv.DictWriter(target,fieldnames=fields);w.writeheader();w.writerows(rows)
def run(args):
    return subprocess.run(args,text=True,capture_output=True)
def reason(p):
    tail=p.stderr.strip().splitlines()
    return f"diagnostic_rc_{p.returncode}: "+(tail[-1][:280] if tail else 'no diagnostic message')
with panel.open() as f:cases=list(csv.DictReader(f))
routes=[];faults=[];keys=[];costs=[];errors=[];plan_errors=[];exclusions=[];owners=set()
for method in methods:
    command=[sys.executable,str(here/"route-worker.py"),str(build),method]
    ctl=run(command)
    if ctl.returncode:
        for r in cases:
            exclusions.append(dict(method=method,family=r["family"],layout=r["layout"],reason="method-control: "+reason(ctl)))
        continue
    control=json.loads(ctl.stdout)
    for t in control["faults"]:faults.append(dict(method=method,cpu=cpu,**t))
    for t in control["keys"]:keys.append(dict(method=method,cpu=cpu,**t))
    for r in cases:
        identity=dict(method=method,family=r["family"],layout=r["layout"])
        p=run(command+[r[k] for k in geometry])
        if p.returncode:
            exclusions.append(dict(identity,reason=reason(p)))
            continue
        results=json.loads(p.stdout)
        if isinstance(results,dict) and results.get("excluded"):
            results.pop("excluded")
            errors.append(dict(identity,**results))
            exclusions.append(dict(identity,reason=f"{results['stage']}: {results['reason']} status={results['create_status']}/{results['execution_status']}"))
            continue
        for s in results:
            routes.append(dict(identity,cpu=cpu,**{k:r[k] for k in geometry[6:]},**s))
cost_fields="sample status create_ns get_ns destroy_ns clock_pair_ns callable destroy_status".split()
for r in cases:
    p=run([str(build/"handle-plan-cost"),*(r[k] for k in geometry[:6]),"16"])
    owners.update(s for s in p.stderr.splitlines() if s.startswith("plan-cost "))
    here_cost=[]
    for line in p.stdout.splitlines():
        tag,*v=line.split(",")
        if tag=='COST' and len(v)==len(cost_fields):
            here_cost.append(dict(family=r["family"],layout=r["layout"],cpu=cpu,**{k:r[k] for k in geometry[:6]},
                                  **dict(zip(cost_fields,v,strict=True))))
    costs.extend(here_cost)
    if p.returncode or len(here_cost)!=16:
        plan_errors.append(dict(family=r["family"],layout=r["layout"],samples=len(here_cost),returncode=p.returncode,
                                reason="invalid-or-uncallable-plan-samples" if len(here_cost)==16 else "incomplete-plan-diagnostics"))
save('handle-routes.csv',routes,fields_route)
save('handle-faults.csv',faults,"method cpu scenario status creates gets calls destroys recovered".split())
save('handle-keys.csv',keys,"method cpu key step status creates gets calls destroys".split())
save('handle-plan-cost.csv',costs,["family","layout","cpu",*geometry[:6],*cost_fields])
save('handle-route-exclusions.csv',exclusions,"method family layout reason".split())
save('handle-route-errors.csv',errors,"method family layout stage reason ordinary_status trace_status create_status execution_status destroy_status creates gets calls destroys".split())
save('handle-plan-errors.csv',plan_errors,"family layout samples returncode reason".split())
(out/'handle-api-owners.txt').write_text('\n'.join(sorted(owners))+'\n')
