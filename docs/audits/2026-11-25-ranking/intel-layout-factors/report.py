"""Report the six preregistered storage-layout effects and their paired uncertainty."""
import argparse,csv,json,math
from pathlib import Path
from statistics import geometric_mean
from score import FACTORS,FAMILIES,METHODS

def ratio(v):return f"{v:.4f}×"

def render(d):
    by={(r["family"],r["method"],r["factor"],r["other"]):r for r in d["effects"]}
    end={(r["family"],r["method"]):r for r in d["endpoints"]}
    h=d["historical_endpoints"]
    label={"repeat-win-deeper":"108×1728×320","repeat-win-ragged":"137×2179×347","fresh-short":"43×2291×137"}
    names={"stream":"COB","mkl":"oneMKL","stream_over_mkl":"COB/oneMKL"}
    out=["# Which matrix layout changes affect multiplication time?","",
         f"The experiment ran on {d['model']} using COB’s released stream-B kernel and ordinary sequential oneMKL. For each input and the output, it switches the row stride and the starting address independently. Each of three shapes uses the same logical input matrices across 64 storage layouts. A ratio of 0.90× means the changed layout took 10% less time, and 1.10× means it took 10% more.","",
         "## Whole-layout change","",
         "| Shape M×N×K | Method | Current cropped/original | Earlier cropped/original | Six current rounds |",
         "| --- | --- | ---: | ---: | ---: |"]
    summary=[]
    for n in METHODS:
        pooled=[r for r in d["descriptive_equal_geometry_means"] if r["kind"]=="main" and r["method"]==n]
        primary=max(pooled,key=lambda r:abs(math.log(r["geometric_mean"])))
        a=primary["geometries_above_one"];b=primary["geometries_below_one"]
        phrase=(f"longer on all three" if a==3 else f"shorter on all three" if b==3 else f"longer on {a} and shorter on {b}")
        term=primary["factor"].replace("_"," ").replace("origin","start offset")
        summary.append(f"For {names[n]}, **{term}** produced the largest main effect averaged equally across the three shapes, {ratio(primary['geometric_mean'])} the original time. It ran {phrase}.")
    mk=[by[f,"mkl",key,""] for f in FAMILIES for key in ("B_stride","B_origin")]
    if all(r["rounds_below_one"]==6 for r in mk):
        out[4:4]=["**Both changes to B made oneMKL faster on all three shapes.** The larger row stride improved elapsed time by "+
                  ", ".join(f"{(1-by[f,'mkl','B_stride','']['ratio'])*100:.1f}%" for f in FAMILIES)+
                  " in table order. Moving B’s start independently improved it by "+
                  ", ".join(f"{(1-by[f,'mkl','B_origin','']['ratio'])*100:.1f}%" for f in FAMILIES)+
                  ". All six round estimates on every shape agreed on the direction of both effects.",""]
    else:out[4:4]=[" ".join(summary),""]
    for f in FAMILIES:
        for n in METHODS:
            a=end[f,n];p=h[f][n]
            out.append(f"| {label[f]} | {names[n]} | {ratio(a['ratio'])} | {ratio(p['prior']['ratio'])} | {ratio(a['min_ratio'])}–{ratio(a['max_ratio'])} |")
    out+=["","The two historical endpoints are exact matches in shape, stride and relative start address. The historical CPU was "
         +h[FAMILIES[0]][METHODS[0]]["historical_model"]+".","",
         "## Individual layout features","",
         "The high setting uses the cropped layout’s stride or start offset. Within each round, each contrast holds the other five features fixed across 32 matched pairs. The table gives the geometric mean of their time ratios over six rounds.","",
         "| Shape | Method | A stride | B stride | C stride | A start | B start | C start |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in FAMILIES:
        for n in (*METHODS,"stream_over_mkl"):
            vals=[ratio(by[f,n,x,""]["ratio"]) for x in FACTORS]
            out.append(f"| {label[f]} | {names[n]} | "+" | ".join(vals)+" |")
    out+=["","The COB/oneMKL values describe how a layout change alters the time ratio of the two implementations. Values above one shift the relative performance in oneMKL’s favor.","",
         "## Stride and start address on the same operand","",
         "Each number is the ratio of the stride effect at the nonzero starting offset to the stride effect at zero offset. The other four features are averaged over all 16 settings.","",
         "| Shape | Method | A stride × A start | B stride × B start | C stride × C start |",
         "| --- | --- | ---: | ---: | ---: |"]
    for f in FAMILIES:
        for n in METHODS:
            out.append(f"| {label[f]} | {names[n]} | "+" | ".join(ratio(by[f,n,FACTORS[i],FACTORS[i+3]]["ratio"]) for i in range(3))+" |")
    cond={(r["family"],r["method"],r["stride"],r["origin_level"]):r for r in d["conditional_stride_effects"]}
    f=max(FAMILIES,key=lambda f:abs(math.log(by[f,"mkl","B_stride","B_origin"]["ratio"])))
    out+=["",f"The strongest within-B interaction on oneMKL occurred at {label[f]}. Raising B’s stride changed elapsed time to "
           f"{ratio(cond[f,'mkl','B_stride',0]['ratio'])} with its original starting address, and to {ratio(cond[f,'mkl','B_stride',1]['ratio'])} "
           "with the cropped address. An optimization that changes both needs a whole-call timing on each combination."]
    out+=["","## Effects that changed direction",""]
    for n in METHODS:
        main=[v for v in d["descriptive_equal_geometry_means"] if v["method"]==n and v["kind"]=="main"]
        across=[v["factor"].replace("_"," ") for v in main if min(v["geometries_above_one"],v["geometries_below_one"])>0]
        rows=[v for v in d["effects"] if v["method"]==n and v["kind"]=="main"]
        mixed=sum(v["rounds_above_one"]>0 and v["rounds_below_one"]>0 for v in rows)
        answer=", ".join(across) if across else "none of the six feature contrasts"
        out.append(f"For {names[n]}, {answer} changed direction across shapes. Individual rounds disagreed in direction for {mixed} of the 18 shape–feature combinations.")
        out.append("")
    out+=["","The complete results include the remaining 12 feature interactions, the slope at each starting offset, all six round estimates, and both seed-block estimates. With three fixed shapes on one CPU, the observed ranges describe sensitivity to layout and trial order.","",
          "## Verification","",
          f"The audit checked {d['coverage']['independent_fp64_checked_positions']:,} output positions against independent FP64 products over {d['coverage']['preflight_fixtures']:,} preflight observations and {d['coverage']['warm_observations']:,} timed observations. Each timing includes 16 complete calls, totaling {d['coverage']['warm_complete_calls']:,} warm calls. Input fingerprints matched across layouts and methods. The largest observed absolute output error was {max(d['coverage']['max_abs'].values()):.6g} against a limit of 0.002. [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/{d['native']['run']}) · [Benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/{d['native']['benchmark']}/tools/world-rank/factorial)"]
    return "\n".join(out)+"\n"

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("score",type=Path);cli.add_argument("out",type=Path);a=cli.parse_args()
    a.out.write_text(render(json.loads(a.score.read_text())))
