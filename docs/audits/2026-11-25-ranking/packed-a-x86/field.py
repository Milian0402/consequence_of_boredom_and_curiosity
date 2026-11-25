"""Compute separate standalone, independent, retained-plan and vendor-assisted rankings."""
from statistics import geometric_mean,median
from measurements import compare,demand

def score_cases(panel,roles,designated,times,first,allowed,exclusions,checks):
    results=[]
    for f,s in panel.items():
        eligible=sorted(allowed[f])
        medians={n:1e6*median(times[f,n].values()) for n in eligible}
        cold={n:[first[f,n][b]*1e6 for b in (1,2)] for n in roles if len(first[f,n])==2}
        groups={
            "direct":[n for n in eligible if roles[n]=="direct"],
            "external_oneshot":[n for n in eligible if roles[n] in ("direct","external")],
            "external_any":[n for n in eligible if roles[n] in ("direct","retained","external")],
            "whole_oneshot":[n for n in eligible if roles[n]!="retained"],
            "whole":eligible}
        win={name:min(g,key=lambda n:(medians[n],n)) for name,g in groups.items() if g}
        demand(set(win)==set(groups),f"No qualified independent method at {f}")
        order=sorted(eligible,key=lambda n:(medians[n],n))
        entries={}
        for n in designated:
            if n not in eligible:continue
            pairs={p:compare(times[f,n],times[f,p]) for p in eligible if n!=p}
            entries[n]={"rank":order.index(n)+1,"pairs":pairs}
        numerical={n:dict(max_abs=max(r["max_abs"] for r in checks[f,n].values()),
                          signed_min=min(r["signed_min"] for r in checks[f,n].values()),
                          signed_max=max(r["signed_max"] for r in checks[f,n].values())) for n in roles if checks[f,n]}
        results.append(dict(family=f,shape=s,qualified=eligible,exclusions=exclusions[f],medians_us=medians,
                            first_us=cold,winners=win,entries=entries,numerical=numerical))
    return results


def aggregate(pairs):
    values=[p["median"] for p in pairs]
    jack=[geometric_mean(values[:i]+values[i+1:]) for i in range(len(values))] if len(values)>1 else values
    return dict(geomean=geometric_mean(values),seed_geomeans=[geometric_mean(p["seed_medians"][i] for p in pairs) for i in (0,1)],
                wins=sum(v>1 for v in values),observation_wins=sum(p["wins"] for p in pairs),minimum=min(values),maximum=max(values),
                leave_one_layout_out_geomeans=jack,leave_one_layout_out_geomean_range=[min(jack),max(jack)])


def decisions(cases,contract):
    fresh=[c for c in cases if c["family"].startswith("fresh-")]
    new,old=contract["candidate"],contract["baseline"]
    threshold=contract["primary"]
    demand(len(fresh)==threshold["fresh_cases"],"Frozen fresh layout count differs")
    mandatory_ok=all(new in c["entries"] and old in c["qualified"] for c in cases)
    if not mandatory_ok:
        return {"passes":False,"mandatory_qualified":False,
                "failed_cases":[c["family"] for c in cases if new not in c["entries"] or old not in c["qualified"]]}
    matched=aggregate([c["entries"][new]["pairs"][old] for c in fresh])
    component=dict(geomean=matched["geomean"]>=threshold["geomean_at_least"],minimum=matched["minimum"]>=threshold["minimum_at_least"],
                   layout_wins=matched["wins"]>=threshold["case_wins_at_least"],both_seed_geomeans=min(matched["seed_geomeans"])>threshold["each_seed_greater_than"])
    field={}
    for group in ("external_oneshot","external_any"):
        peers=[c["winners"][group] for c in fresh]
        paired=[c["entries"][new]["pairs"][p] for c,p in zip(fresh,peers)]
        mark=contract["field"]
        case_pass=[v["median"]>mark["every_fresh_paired_median_greater_than"] and v["wins"]>=mark["every_fresh_observations_won_at_least"] for v in paired]
        field[group]=dict(summary=aggregate(paired),raw_wins=sum(c["medians_us"][new]<c["medians_us"][p] for c,p in zip(fresh,peers)),
                          cases_passed=sum(case_pass),passes=all(case_pass))
    return dict(passes=all(component.values()),criteria=component,mandatory_qualified=True,matched=matched,field=field)


def assisted(cases,underlying):
    new="host-pad8";control="host-direct";out={}
    fresh=[r for r in cases if r["family"].startswith("fresh-")]
    if not all(new in c["routes"] for c in fresh):
        return {"passes":False,"reason":"host_pad8_not_available"}
    active=[c for c in fresh if c["routes"][new]["padded"]]
    if not active or any(new not in c["entries"] or not {control,underlying}<=set(c["qualified"]) for c in fresh):
        return {"passes":False,"reason":"incomplete_vendor_assisted_qualifications"}
    sets={p:[c["entries"][new]["pairs"][p] for c in active] for p in (control,underlying)}
    for peer,v in sets.items():
        out[peer]=aggregate(v)
    both=sum(all(sets[peer][i]["median"]>1.02 and sets[peer][i]["wins"]>=5 for peer in sets) for i in range(len(active)))
    return dict(passes=both>=2 and all(summary["geomean"]>1.02 and summary["minimum"]>=.97 for summary in out.values()),
                underlying=underlying,fresh_padded=[c["family"] for c in active],fresh_forwarded=[c["family"] for c in fresh if c not in active],
                fresh_qualified_wins_against_both=both,comparisons=out)
