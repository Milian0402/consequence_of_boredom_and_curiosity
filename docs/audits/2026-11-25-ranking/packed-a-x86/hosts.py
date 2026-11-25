"""Retain individual hosted results and distinguish model replication from cross-vendor transfer."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from measurements import demand

def summary(inputs):
    hosts=defaultdict(list);vendors=defaultdict(list);seen=set()
    for p in sorted(inputs,key=lambda x:(x["run"],x["attempt"],x["replica"])):
        key=p["run"],p["attempt"],p["replica"]
        demand(key not in seen,"Duplicate job in multi-host comparison")
        seen.add(key)
        receipt=p.get("receipt",{})
        demand(receipt.get("artifact_id") and receipt.get("artifact_sha256"),"Original host receipt is required for multi-host statements")
        row=dict(run=p["run"],attempt=p["attempt"],replica=p["replica"],job_id=receipt["job_id"],
                 artifact_id=receipt["artifact_id"],artifact_sha256=receipt["artifact_sha256"],model=p["model"],
                 vendor=p["cpu_vendor"],isa=p["isa"],primary=p["gates"]["passes"],matched=p["gates"].get("matched"),
                 external_gates={k:v["passes"] for k,v in p["gates"].get("field",{}).items()},available=list(p["methods"]),
                 omitted=p["omitted"])
        hosts[p["model"]].append(row);vendors[p["cpu_vendor"]].append(row)
    both=set(vendors)=={"GenuineIntel","AuthenticAMD"}
    groups={m:dict(first_registered_replica=rs[0]["replica"],first_registered_run=rs[0]["run"],status="all passed" if all(r["primary"] for r in rs)
                   else "mixed" if any(r["primary"] for r in rs) else "all missed",observations=rs) for m,rs in hosts.items()}
    return dict(models=groups,cpu_model_count=len(groups),distinct_native_vendors=sorted(vendors),
                both_native_vendors_observed=both,all_native_vendor_gates_pass=both and all(r["primary"] for rs in vendors.values() for r in rs),
                all_fresh_field_thresholds_pass=both and all(r["external_gates"] and all(r["external_gates"].values()) and not r["omitted"] for rs in vendors.values() for r in rs),
                interpretation="Ordered by registered run, attempt and replica. Timings and seed blocks from one machine are dependent. Same-model replicas are reported separately.")

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("reports",nargs="+",type=Path);cli.add_argument("--output",type=Path)
    a=cli.parse_args();data=json.dumps(summary([json.loads(p.read_text()) for p in a.reports]),indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:a.output.write_text(data)
    else:print(data,end="")
