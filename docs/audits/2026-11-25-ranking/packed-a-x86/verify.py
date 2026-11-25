"""Independently score a frozen native frontier host; optionally authenticate its original GitHub ZIP."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile
from collections import Counter
from field import score_cases,decisions,assisted
from hosted import cross_check
from measurements import demand,measure,rows,shape,single
from native import NEW,PRIOR,controls,host,peers,products,sha,valid_sha,witness


def verify_receipt(raw,record,payload,run):
    receipt=json.loads(record.read_text())
    digest=hashlib.sha256(payload.read_bytes()).hexdigest()
    demand(receipt["run"]==run["run"] and receipt["attempt"]==run["attempt"] and receipt["benchmark_sha"]==run["benchmark"] and
           receipt["source_sha"]==run["benchmark"] and receipt["replica"]==run["replica"] and receipt["branch"]=="bench/native-linux-gemm" and
           receipt["job"]=="Frontier ranking on x86-64, replica "+str(run["replica"]) and
           receipt["artifact"]==f"frontier-x86-{run['run']}-{run['attempt']}-{run['replica']}" and
           receipt["artifact_sha256"]==digest and receipt["bytes"]==payload.stat().st_size and receipt["artifact_id"]>0,
           "Hosted artifact digest/run/commit mismatch")
    with ZipFile(payload) as zipfile:
        demand(zipfile.testzip() is None, "Corrupt original hosted ZIP")
        names=[p.filename for p in zipfile.infolist() if not p.is_dir()]
        demand(len(set(names))==len(names)==receipt["members"] and all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names),"Ambiguous raw archive")
        bases=[n[:-len("run.json")] for n in names if n=="run.json" or n.endswith("/run.json")]
        matching=[b for b in bases if json.loads(zipfile.read(b+"run.json"))==run]
        demand(len(matching)==1,"Cannot identify source run in archive")
        prefix=matching[0]
        root=raw
        for part in PurePosixPath(prefix).parts:
            root=root.parent
        for n in names:
            f=root/n
            demand(f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root.resolve()) and
                   sha(zipfile.read(n))==sha(f.read_bytes()),"Raw data differs from hosted member "+n)
    return {"run":run["run"],"attempt":run["attempt"],"artifact_id":receipt["artifact_id"],"artifact_sha256":digest,"job_id":receipt["job_id"]}


def verify(root,contract,receipt=None,archive=None):
    run=json.loads((root/"run.json").read_text());phase=contract["phase"]
    demand(run["production"]==contract["release"] and run["benchmark"]==contract["benchmarks"].get(str(run["run"])) and
           min(run["attempt"],run["run"])>0 and run["replica"]>=0,"Native Actions source or job")
    for f,h in run["source_files"].items():
        demand(Path(f).name==f and valid_sha(h) and h==sha((root/f).read_bytes()),"Native manifest SHA differs "+f)
    demand(set(run["source_files"])>={"panel.csv","protocol.md","novelty.json","methods.csv","frontier-sources.json"} and
           run["source_files"]["panel.csv"]==contract["panel_sha256"] and run["source_files"]["protocol.md"]==contract["protocol_sha256"],
           "Frozen panel or test rule changed")
    demand(all(run["source_files"].get(n)==h for n,h in contract["source"]["registered_files"].items()),"Preregistered files differ between native and frozen source")
    data=rows(root/"panel.csv"); panel={r["family"]:shape(r) for r in data}
    demand(len(data)==len(panel)==len(set(panel.values()))==12,"Panel rows are duplicated or missing")
    mdata=rows(root/"methods.csv"); registry=single(mdata,lambda r:r["implementation"]); order=list(registry)
    roles={n:r for n,r in contract["roles"].items() if n in registry}
    omitted=single(rows(root/"omissions.csv"),lambda r:r["implementation"])
    absent=set(contract["roles"])-set(roles)
    demand(order==list(roles) and set(omitted)==absent and
           absent<={"libxsmm-dispatch","libxsmm-retained","host-direct","host-pad8"} and
           all(r["reason"] for r in omitted.values()),"Unexplained change to registered cohort, order or mandatory peers")
    methods={};rank=json.loads((root/"ranking.json").read_text())
    demand(rank["panel_sha256"]==contract["panel_sha256"] and not rank.get("small_vendors"),"Unexpected shape-filtered cohort")
    for n,r in registry.items():
        role=roles[n]; rawrole="standalone" if role in ("incumbent","candidate") else "vendor-assisted" if role=="assisted" else "independent"
        policy="retained" if role=="retained" else "one-shot"
        demand(r["role"]==rawrole and r["policy"]==policy and valid_sha(r["sha256"]),f"Role or hashed binary differs for {n}")
        variant,path=rank["implementations"][n]
        base=(n+".worker" if rawrole=="standalone" else "onednn-f32-"+n.removeprefix("onednn-")+".so" if n.startswith("onednn-") else n+".so")
        demand(Path(path).name==base and variant==("linked-public" if rawrole=="standalone" else "direct"),"Wrong public interface "+n)
        methods[n]=dict(role=role,policy=policy,sha256=r["sha256"])
    model,vendor,isa=host(root); binding=products(root,contract,methods,{n:r["sha256"] for n,r in registry.items()})
    valid,routes,cpu=witness(root,panel,isa)
    parity=controls(root,panel,routes)
    ext,traces,underlying=peers(root,panel,methods,vendor,cpu,contract)
    valid.update(ext)
    times,first,qualified,excluded,checks,coverage=measure(root,phase,panel,roles,contract["seeds"],valid,order)
    demand(cpu==coverage["logical_cpu"]==run["cpu"],"Trace and ordinary workers had different native CPU bindings")
    selected={n for n in roles if roles[n] in ("incumbent","candidate","assisted")}
    cases=score_cases(panel,roles,sorted(selected),times,first,qualified,excluded,checks)
    for c in cases:
        f=c["family"]
        c["routes"]={**routes[f],**traces.get(f,{})}
    gate=decisions(cases,contract);gate["vendor_assisted"]=assisted(cases,underlying)
    cross_check(root,run,methods,cases,gate)
    result=dict(phase=phase,model=model,architecture="x86_64",cpu_vendor=vendor,isa=isa,run=run["run"],attempt=run["attempt"],
                omitted={n:r["reason"] for n,r in omitted.items()},
                replica=run["replica"],revision=run["benchmark"],release=run["production"],seeds=contract["seeds"],methods=methods,cases=cases,
                gates=gate,coverage=coverage,source=binding,control_equivalence=parity)
    if receipt:
        demand(archive is not None,"Official verification requires the original archive")
        result["receipt"]=verify_receipt(root,receipt,archive,run)
    return result


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw",type=Path);cli.add_argument("--contract",type=Path,default=Path(__file__).with_name("contract.json"))
    cli.add_argument("--receipt",type=Path);cli.add_argument("--archive",type=Path);cli.add_argument("--output",type=Path)
    args=cli.parse_args()
    if bool(args.receipt)!=bool(args.archive):cli.error("Supply both --receipt and --archive to verify original hosted files")
    answer=json.dumps(verify(args.raw,json.loads(args.contract.read_text()),args.receipt,args.archive),indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.output:args.output.write_text(answer)
    else:print(answer,end="")
