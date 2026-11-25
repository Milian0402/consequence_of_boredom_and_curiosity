"""Predict a fixed schedule using only the two calibration layouts on a verified native host."""
import argparse
import csv
import json
from pathlib import Path
from statistics import geometric_mean, median

TRAIN = ("deep-odd-prior", "wide-odd-prior")
TEST = ("deep-odd-followup", "wide-odd-followup", "near-square-followup")
PAIRS = [(seed, i) for seed in (31048579, 196173023) for i in (1, 2, 3)]


def paired(family, old, method):
    return [family[old][key]/family[method][key] for key in PAIRS]


def select(layouts, original, signed, direct):
    alternatives = [(direct, [median(paired(layouts[f], original, direct)) for f in TRAIN]),
                    (signed, [median(paired(layouts[f], original, signed)) for f in TRAIN])]
    best, best_score = original, 1.015
    for name, values in alternatives:
        score = geometric_mean(values)
        if min(values) >= 0.995 and score > best_score:
            best, best_score = name, score
    return best


def read(root, name):
    with (root / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(root):
    info = json.loads((root / "signed-score.json").read_text())
    direct = "mkl" if info["native_provider"] == "mkl" else "openblas"
    original, signed = "original-" + direct, "signed-" + direct
    layouts = {family: {} for family in (*TRAIN, *TEST)}
    roles = {r["implementation"]:r["role"] for r in read(root,"eligibility.csv") if r["status"]=="included"}
    eligible = {(r["family"], r["implementation"]) for r in read(root, "case-eligibility.csv") if r["qualified"] == "1"}
    for r in read(root, "timings.csv"):
        fam, name = r["family"], r["implementation"]
        if (fam, name) not in eligible:
            continue
        obs = layouts[fam].setdefault(name,{})
        key = int(r["seed"]),int(r["round"])
        if key in obs or int(r["calls"])!=1 or int(r["threads"])!=1:
            raise ValueError("invalid timing")
        obs[key] = float(r["seconds"])
    for fam,methods in layouts.items():
        if any(set(obs)!=set(PAIRS) for obs in methods.values()):
            raise ValueError(f"incomplete timings on {fam}")
    chosen=select(layouts,original,signed,direct)
    recorded=sum(sum(layouts[f][n].values()) for f in TRAIN for n in (original,signed,direct))
    def result(f):
        methods=layouts[f]
        peers = [name for name in methods if roles[name] in ("direct","external_winograd") and name!="onednn-cached-plan"]
        fastest=min(peers,key=lambda n:median(methods[n].values()))
        if chosen not in methods:
            return {"family":f,"qualified":False}
        vs=paired(methods,original,chosen)
        external=paired(methods,fastest,chosen)
        return {"family":f,"qualified":True,"chosen_ms":1000*median(methods[chosen].values()),
                "speedup_vs_release":median(vs),"wins_vs_release":sum(r>1 for r in vs),
                "fastest_external":fastest,"speedup_vs_external":median(external),
                "wins_vs_external":sum(r>1 for r in external),"identical_external":fastest==chosen}
    rows=[result(f) for f in TEST]; gains=[r["speedup_vs_release"] for r in rows if r["qualified"]]
    return {"model":info["model"],"release":original,"chosen":chosen,
            "calibration_timed_multiply_seconds":recorded,
            "calibration_ratios":{n:[median(paired(layouts[f],original,n)) for f in TRAIN] for n in (direct,signed)},
            "followups":rows,"meets_public_retest_rule":len(gains)==3 and min(gains)>=0.98 and geometric_mean(gains)>1.02}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("verified_signed_artifact",type=Path)
    print(json.dumps(run(cli.parse_args().verified_signed_artifact),indent=2,allow_nan=False))
