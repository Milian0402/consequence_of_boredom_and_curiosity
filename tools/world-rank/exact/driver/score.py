"""Recompute per-fixture exact-domain SGEMM comparisons from all original observations."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

SEEDS=(188441917,3079128031)
S8_SEEDS=(2341279043,916583729)
FIELDS=("seconds cpu_seconds cpu threads checked guards guard_changes nonfinite over_abs over_gamma "
        "not_exact max_abs max_rho output_sha256").split()
STAGES=("FIRST","WARM")
CORPUS_OUTPUT={
    "optdigits":"ec47b13672085cc2096a990dd166c76e2e29843c7f79ea88d1d483db94b05db0",
    "optdigits-images":"eec353b47234bfe0fa3544331dc16aceee6a9b1ee3b7165b51e30fe565a9e825",
}

def require(valid, reason):
    if not valid:
        raise ValueError(reason)

def order_for(methods, case, block, rnd):
    off=(case+3*(rnd-1)+5*block)%max(1,len(methods))
    order=methods[off:]+methods[:off]
    return order if block==0 else order[::-1]

def s8_route(row, p, binding, stage):
    record = row["traces"]["S8_ROUTE"][stage]
    route, reason, count, amax, bmax, plans, products, fallback, native, impl = record
    counters = tuple(map(int, (route,count,amax,bmax,plans,products,fallback,native)))
    route,count,amax,bmax,plans,products,fallback,native = counters
    spec, k = p["spec"], p["k"]
    cap = {"optdigits":16,"optdigits-images":1}.get(spec,63)
    total = (p["m"]+p["n"])*k
    available = binding["checked_int8_native"]
    require(type(available) is bool, "missing signed instruction witness")
    require(binding.get("checked_probe_passed") is True, "independent checked INT8 full-call proof failed")
    if k<=128 or k>4096:
        expected = "depth-shallow" if k<=128 else "depth-outside"
        require(reason==expected and counters==(2,0,-1,-1,0,0,1,0), "checked INT8 shape-only FP32 bypass")
    elif route==3 and not available:
        require(reason=="backend-unavailable" and (plans,products,fallback,native)==(0,0,1,0),
                "checked INT8 provider unavailability")
        require((count,amax,bmax)==(0,-1,-1) or
                (spec not in ("early","late") and (count,amax,bmax)==(total,cap,cap)),
                "checked INT8 unavailable scan stage")
    elif spec in ("early","late"):
        invalid = (min(k,64),-1,-1) if spec=="early" else (total,cap,-1)
        require(reason=="integer-domain" and counters==(2,*invalid,0,0,1,0), "checked INT8 fractional FP32 fallback")
    else:
        require(available and reason=="native-exact" and counters==(1,total,cap,cap,1,1,0,1)
                and k*cap*cap<=2**24 and bool(impl) and impl==binding["int8_implementation"],
                "checked signed INT8 native route, exact bound or primitive identity")


def parse_route(row, p, binding, stage):
    traces=row["traces"]; name=row["method"]; mode=p["spec"]
    integer=mode not in ("early","late")
    if name=="s8-dispatch":
        s8_route(row,p,binding,stage)
    elif name=="exact-dispatch":
        r,reason,count,amax,bmax,fast,fallback,terms,native,impl=traces["ROUTE"][stage]
        r,count,amax,bmax,fast,fallback,terms,native=map(int,(r,count,amax,bmax,fast,fallback,terms,native))
        if not binding["native"]:
            require((r,reason,count,amax,bmax,fast,fallback,terms,native)==(3,"native-disabled",0,-1,-1,0,1,0,0),"unavailable-native fallback")
        else:
            expected_count=min(64,p["k"]) if mode=="early" else (p["m"]+p["n"])*p["k"]
            require(count==expected_count,"actual logical scan count")
            bound=1 if mode=="optdigits-images" else 16 if mode=="optdigits" else 63
            expect=(-1,-1) if mode=="early" else (bound,-1) if mode=="late" else (bound,bound)
            require((amax,bmax)==expect,"observed input maxima")
            if integer:
                require((r,reason,fast,fallback,terms,native)==(1,"native-exact",1,0,1,1) and
                        bool(impl) and impl==binding["implementation"],"native plan identity")
            else:
                require((r,reason,fast,fallback,terms,native)==(2,"integer-domain",0,1,0,0),"value rejection")
    elif name=="raw-bf16":
        route,terms,plans,native,impl=traces["BF16"][stage]
        require((route,terms,plans,native)==("1","1","1","1") and impl==binding["implementation"],"raw native BF16 proof")
    elif name=="onednn-int8":
        terms,impl=traces["INT8"][stage]
        require(integer and terms=="1" and bool(impl) and "ref" not in impl.lower(),"native one-product INT8 proof")
        if binding.get("int8_implementation"):
            require(impl==binding["int8_implementation"],"INT8 execution differs from preflight")
    elif name=="bitset-popcount":
        words,=traces["BINARY"][stage]
        require(mode=="optdigits-images" and int(words)==(p["k"]+63)//64,"binary full-repack proof")
    elif name.startswith("onednn-"):
        impl,plans,cached=traces["DNN"][stage]
        keep=name.endswith("cached-plan")
        require(bool(impl) and "ref" not in impl.lower() and int(cached)==int(keep) and
                int(plans)==(1 if stage=="FIRST" or keep else 2),"oneDNN strict plan")

def compare(candidate,peer):
    ratios=[b/a for a,b in zip(candidate,peer)]
    med=median(ratios);wins=sum(x>1 for x in ratios);raw=median(peer)/median(candidate)
    return dict(paired_ratios=ratios,paired_median=med,wins=wins,ties=ratios.count(1),seed_medians=[median(ratios[:3]),median(ratios[3:])],
                raw_median_ratio=raw,passes=med>1.02 and wins>=5,raw_median_win=raw>1)

def compute(raw, records):
    checked = "s8-dispatch" in raw["methods"]
    seeds = S8_SEEDS if checked else SEEDS
    primary = "s8-dispatch" if checked else "exact-dispatch"
    slots = [(s,r) for s in seeds for r in (1,2,3)]
    require(raw["seeds"]==list(seeds),"seed schedule differs")
    indexed={}
    for r in records:
        key=(r["family"],r["method"],r["seed"],r["round"])
        require(key not in indexed,"duplicate observation")
        indexed[key]=r
    source=raw["methods"]; consumed=set(); cases={}
    for i,p in enumerate(raw["panel"]):
        f=p["family"];active=[n for n in source if n not in raw["skips"][f]]
        policy=raw["bindings"][f]
        modes=raw["goldens"][f]
        for block,seed in enumerate(seeds):
            for rnd in (1,2,3):
                for pos,n in enumerate(order_for(active,i,block,rnd)):
                    key=(f,n,seed,rnd)
                    require(indexed[key]["position"]==pos,"method order differs")
                    consumed.add(key)
        if p["spec"] in ("optdigits","optdigits-images"):
            require(modes[str(seeds[0])]["inputs"]==modes[str(seeds[1])]["inputs"],"real inputs changed between scheduling blocks")
        evidence={};numerical=[];exact=[]; structural=[]
        for n in active:
            rows=[indexed[f,n,s,r] for s,r in slots]
            issues=[];badnums=0;inexact=0;checks=0;peak=0.
            times={s:[] for s in STAGES}
            anchors={}
            for row in rows:
                try:
                    require("failure" not in row,row.get("failure","worker failure"))
                    require(row["inputs"]==modes[str(row["seed"])]["inputs"],"input differs")
                    for stage in STAGES:
                        v=row["stages"][stage]
                        require(math.isfinite(v["seconds"]) and math.isfinite(v["cpu_seconds"]) and min(v["seconds"],v["cpu_seconds"])>0,"invalid clock")
                        require(v["threads"]==1 and v["cpu"]==raw["cpu"] and v["checked"]==p["m"]*p["n"],"CPU, thread or coverage")
                        guarded=(p["cr"]+p["m"]+2)*p["ldc"]*4;page=raw["page_bytes"]
                        require(v["guards"]==(guarded+page-1)//page*(page//4)-v["checked"] and v["guard_changes"]==0,"output bounds")
                        require(all(0<=v[k]<=v["checked"] for k in ("nonfinite","over_gamma","not_exact","over_abs")),"invalid evidence count")
                        require(v["output_sha256"]==anchors.setdefault(row["seed"],v["output_sha256"]),"inconsistent output bits")
                        if p["spec"] in CORPUS_OUTPUT and v["not_exact"]==0:
                            require(v["output_sha256"]==CORPUS_OUTPUT[p["spec"]],"known integer corpus bytes differ")
                        parse_route(row,p,policy,stage)
                        fallback_bf16 = n=="exact-dispatch" and (not policy["native"] or p["spec"] in ("early","late"))
                        fallback_s8 = n=="s8-dispatch" and row["traces"]["S8_ROUTE"][stage][0]!="1"
                        if fallback_bf16 or fallback_s8:
                            # Compare against the independently measured identical backing DSO on this exact input.
                            match=indexed[f,"fp32-policy",row["seed"],row["round"]]
                            require(match["stages"][stage]["output_sha256"]==v["output_sha256"],"FP32 backing bytes differ")
                        times[stage].append(v["seconds"]);checks+=v["checked"];inexact+=v["not_exact"]
                        badnums+=int(v["max_abs"] is None or v["max_abs"]>.002 or v["max_rho"] is None or
                                     v["max_rho"]>p["k"]/(16777216.-p["k"]) or bool(v["over_abs"] or v["over_gamma"] or v["nonfinite"]))
                        if v["max_abs"] is not None: peak=max(peak,v["max_abs"])
                except (ValueError,KeyError,IndexError,TypeError) as e:
                    issues.append(str(e))
            stable=not issues and all(len(t)==6 for t in times.values())
            if stable:structural.append(n)
            if stable and not badnums:numerical.append(n)
            if stable and not badnums and inexact==0:exact.append(n)
            evidence[n]=dict(checked_values=checks,inexact_values=inexact,failed_numerical_calls=badnums,
                             max_abs=peak,structural_issues=sorted(set(issues))[:6],times_s=times,
                             medians_ms={s:1000*median(t) for s,t in times.items() if len(t)==6})
        def fp32(n):
            return n not in ("s8-dispatch","exact-dispatch","raw-bf16","onednn-int8","bitset-popcount") and source[n]["role"] not in ("integer-specialist","binary-specialist")
        def retained(n):
            return "retain" in source[n]["role"] or n.endswith("cached-plan")
        def independent(n):
            return fp32(n) and n not in ("fp32-policy","cob-standalone") and not n.startswith("release-")
        leaders={};comparisons={}
        integer=p["spec"] not in ("early","late")
        eligible_s8_domain=integer and 128<p["k"]<=4096
        expected_native=(policy["checked_int8_native"] and eligible_s8_domain) if checked else policy["native"] and integer
        strict_primary=(expected_native if checked else integer)
        for stage in STAGES:
            groups={"fp32_oneshot":[n for n in numerical if fp32(n) and not retained(n)],
                    "fp32_any":[n for n in numerical if fp32(n)],
                    "fp32_exact":[n for n in exact if fp32(n)],
                    "fp32_exact_oneshot":[n for n in exact if fp32(n) and not retained(n)],
                    "independent_fp32_oneshot":[n for n in numerical if independent(n) and not retained(n)],
                    "independent_fp32_any":[n for n in numerical if independent(n)],
                    "integer_specialists":[n for n in exact if source[n]["role"] in ("integer-specialist","binary-specialist")],
                    "all_qualified":[n for n in numerical if n in exact or n not in ("raw-bf16","onednn-int8","bitset-popcount","exact-dispatch") or (n=="exact-dispatch" and not integer)]}
            if checked:
                groups["all_qualified"]=[n for n in numerical if fp32(n) or n in exact or
                                           (n==primary and not expected_native) or
                                           (n=="exact-dispatch" and (not integer or not policy["native"]))]
                groups["checked_policies"]=[n for n in groups["all_qualified"] if n in (primary,"exact-dispatch","fp32-policy")]
            leaders[stage]={group:min(pool,key=lambda n:evidence[n]["medians_ms"][stage],default=None) for group,pool in groups.items()}
            if primary not in numerical or (strict_primary and primary not in exact): continue
            candidate=evidence[primary]["times_s"][stage]
            peers={n:compare(candidate,evidence[n]["times_s"][stage]) for n in numerical if n!=primary}
            against={}
            for group,pool in groups.items():
                choices=[n for n in pool if n in peers]
                if choices:
                    peer=min(choices,key=lambda n:evidence[n]["medians_ms"][stage])
                    against[group]=dict(peer=peer,**peers[peer])
            comparisons[stage]=dict(peers=peers,strongest=against)
        cases[f]=dict(spec=p["spec"],dimensions=[p[k] for k in ("m","n","k")],expected_native=expected_native,
                      binding=policy,skips=raw["skips"][f],qualified_numerical=numerical,qualified_exact=exact,
                      structural=structural,evidence=evidence,leaders=leaders,candidate_comparisons=comparisons)
        if checked:
            eligible = primary in numerical and (not expected_native or primary in exact)
            if policy.get("checked_probe_passed") is not True:
                status="checked-probe-failed"
            elif eligible_s8_domain and not policy["checked_int8_native"]:
                status="native-unavailable"
            elif not eligible:
                status="primary-failed-qualification"
            elif expected_native:
                status="verified-native"
            else:
                status="verified-fallback"
            cases[f].update(checked_native_domain=eligible_s8_domain,primary_status=status)
    require(set(indexed)==consumed,"extra measurement rows")
    result=dict(seeds=list(seeds),cases=cases,observations=len(indexed))
    if checked:
        gaps={f:c["primary_status"] for f,c in cases.items() if c["primary_status"] not in ("verified-native","verified-fallback")}
        fresh=cases.get("fresh3584")
        large=None
        if fresh is not None:
            peer=fresh["candidate_comparisons"].get("WARM",{}).get("strongest",{}).get("fp32_any")
            exact_native=fresh["primary_status"]=="verified-native" and primary in fresh["qualified_exact"]
            large=dict(qualified=bool(peer and exact_native and peer["paired_median"]>=2 and
                                      peer["raw_median_ratio"]>=2 and peer["wins"]>=5),
                       exact_native=exact_native,opponent=peer)
        result.update(candidate_method=primary,primary_complete=not gaps,primary_gaps=gaps,fresh_large_effect=large)
    return result

def write_results(out, result):
    (out/"score.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    with (out/"comparisons.csv").open("w",newline="") as file:
        writer=csv.writer(file);writer.writerow(["family","stage","tier","peer","paired_median","wins","raw_median_ratio","passes","seed1_median","seed2_median",*["pair_"+str(i) for i in range(1,7)]])
        for f,c in result["cases"].items():
            for stage,s in c["candidate_comparisons"].items():
                for tier,row in s["strongest"].items():
                    writer.writerow([f,stage,tier,row["peer"],row["paired_median"],row["wins"],row["raw_median_ratio"],int(row["passes"]),*row["seed_medians"],*row["paired_ratios"]])

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("output",type=Path);args=cli.parse_args();out=args.output
    raw=json.loads((out/"measurement.json").read_text())
    records=[json.loads(s) for s in (out/"observations.jsonl").read_text().splitlines()]
    write_results(out,compute(raw,records))
