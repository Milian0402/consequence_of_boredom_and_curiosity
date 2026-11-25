"""Score prespecified provider choices using complete, paired, qualified observations."""
import csv
import json
import os
from pathlib import Path
import platform
import socket
from statistics import median
import sys

dst=Path(sys.argv[1])
def read(name):
    with (dst/name).open() as f:
        return list(csv.DictReader(f))
def shape(r):
    return tuple(int(r[k]) for k in ("m","n","k","pad"))

trained=json.loads((dst/"training-hosts.json").read_text())
arch=platform.machine()
host_lines=(dst/"host.txt").read_text().splitlines()
model=next(r.split(":",1)[1].strip() for r in host_lines if r.startswith("Model name:"))
if "AMD EPYC" in model and arch=="x86_64":
    policy="trained_amd_epyc"
elif "neoverse-n2" in model.lower() and arch=="aarch64":
    policy="trained_neoverse_n2"
elif "intel" in model.lower() and arch=="x86_64":
    policy="intel_extrapolation_from_amd"
else:
    policy="no_trained_family"
fixed=trained[arch]["method"] if policy!="no_trained_family" else None
eligibility=read("eligibility.csv")
direct={r["implementation"] for r in eligibility if r["role"] in ("direct","external_winograd","external_strassen") and r["status"]=="included"}
valid={(shape(r),r["implementation"]) for r in read("case-eligibility.csv") if r["qualified"]=="1"}
times={}
for r in read("timings.csv"):
    key=shape(r),r["implementation"]
    samples=times.setdefault(key,{})
    pair=int(r["block"]),int(r["round"])
    if pair in samples: raise ValueError(f"duplicate timing: {key}, {pair}")
    samples[pair]=float(r["seconds"])
pairs=[(b,r) for b in (1,2) for r in (1,2,3)]
def compare(s,method,peer):
    if (s,method) not in valid or (s,peer) not in valid:return None
    ratios=[times[s,peer][p]/times[s,method][p] for p in pairs]
    return dict(peer=peer,ratios=ratios,median=median(ratios),wins=sum(x>1 for x in ratios))
data={"implementation_phase":"release-replay","cpu_model":model,"cpu_arch":arch,"hostname":socket.gethostname(),
      "training":trained[arch],"different_training_vm_hostname":socket.gethostname()!=trained[arch]["hostname"],
      "policy":policy,"fixed":fixed,"run_id":os.environ.get("GITHUB_RUN_ID"),"cases":[]}
native="mkl" if arch=="x86_64" else "armpl"
for row in read("panel.csv"):
    s=shape(row)
    eligible={name for shape_id,name in valid if shape_id==s}
    for name in eligible:
        if set(times[s,name])!=set(pairs):raise ValueError(f"unexpected valid timing coverage: {s} {name}")
    peers=eligible & direct-{"onednn-cached-plan"}
    if not peers: raise ValueError(f"no complete independent comparator: {s}")
    fastest=min(sorted(peers),key=lambda name:median(times[s,name].values()))
    summaries={}
    for provider in ("openblas",native):
        for mode in ("two","aligned-two"):
            method=f"release-{provider}-{mode}"
            summaries[method]={}
            for description,peer in (("same_provider_ordinary",f"release-{provider}-two"),
                ("matching_prototype",f"cob-{provider}-{mode}"),("direct_provider",provider),
                ("strongest_independent",fastest),("prior_release","release-previous"),
                ("prior_standalone","cob-hk15-parent-960")):
                summaries[method][description]=compare(s,method,peer)
    data["cases"].append(dict(family=row["family"],shape=s,best_independent=fastest,
        median_seconds={n:median(times[s,n].values()) for n in sorted(eligible)},
        fixed_vs_strongest=compare(s,fixed,fastest) if fixed else None,configurations=summaries))
odd=[c["fixed_vs_strongest"] for c in data["cases"] if c["family"] in ("fresh-deep-odd","fresh-wide-odd")]
data["fixed_won_both_new_odds"]=len(odd)==2 and all(x and x["median"]>1 and x["wins"]>=5 for x in odd) if fixed else None
(dst/"release-score.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
