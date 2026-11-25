"""Reconcile hosted scores and omissions with independently reconstructed observations."""
import json
from measurements import close, demand, details, rows


def cross_check(root,run,methods,cases,gates,score_required=True):
    ranking=json.loads((root/"ranking.json").read_text())
    by_family={c["family"]:c for c in cases}
    excluded={c["family"]+"/"+n:v["reason"] for c in cases for n,v in c["exclusions"].items()}
    errors={(c["family"],n,f["mode"],f["block"],f["seed"]) for c in cases for n,v in c["exclusions"].items() for f in v["failures"]}
    received=[(r["family"],r.get("implementation"),r.get("mode"),r.get("block"),r.get("seed")) for r in ranking["errors"]]
    demand(len(set(received))==len(received) and set(received)==errors and
           all("failed full-output validation" in r["error"] for r in ranking["errors"]),"Unexplained coordinator errors")
    demand(int((root/"study-status.txt").read_text().strip())==int(bool(errors)),"Reported phase exit disagrees with actual measurement errors")
    for r in rows(root/"accuracy.csv"):
        f,n=r["family"],r["implementation"]
        if n.startswith("libxsmm-"):
            meta=by_family[f]["routes"][n]
            demand(meta["admitted"] and meta["code_bytes"]==int(details(r)["xsmm_jit_bytes"])>0,"Timed LIBXSMM kernel does not match per-geometry native preflight")
    recorded=root/"frontier-score.json"
    if not gates["mandatory_qualified"] and not recorded.exists():
        return
    hosted=json.loads(recorded.read_text())
    demand(hosted["run"]==run and hosted["method_order"]==list(methods) and hosted["excluded"]==excluded,"Hosted cohort, identity or exclusions")
    for c in cases:
        f=c["family"];w=c["winners"];m=c["medians_us"]
        pub={n for n in m if methods[n]["role"] in ("incumbent","candidate","assisted")}
        project=min(pub,key=lambda n:(m[n],n))
        demand(hosted["leaders"][f]=={"field":w["whole"],"independent":w["external_oneshot"],"including_retained":w["external_any"],"project":project},
               "Hosted field or retained-plan ranking differs for "+f)
        raw=hosted["comparisons"][f];pairs=c["entries"]["stream-packed-a"]["pairs"]
        demand(set(raw)==set(pairs),"Incomplete hosted frontier competitors")
        for n,x in raw.items():
            p=pairs[n]
            demand(x["peer"]==n and x["wins"]==p["wins"] and close(x["speedup"],p["median"])
                   and x["ratios"]==p["ratios"] and x["seed_medians"]==p["seed_medians"] and
                   close(x["candidate_us"],m["stream-packed-a"]) and close(x["peer_us"],m[n]),"Hosted pair mismatch "+str((f,n)))
    q=gates["matched"];g=hosted["standalone_gate"]
    demand(g["passed"]==gates["passes"] and g["geometric_mean"]==q["geomean"] and q["minimum"]==g["minimum"]
           and q["seed_geomeans"]==g["seed_gms"] and q["wins"]==g["shape_wins"],"Hosted threshold diverges")
