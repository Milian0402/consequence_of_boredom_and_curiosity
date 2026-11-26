"""Run isolated complete-call comparisons for the exact-domain dispatcher."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from score import FIELDS, SEEDS, compute, order_for, write_results

DIMS = "m n k lda ldb ldc ar ac br bc cr cc".split()
def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(path):
    with path.open("rb") as f:
        h=hashlib.sha256()
        while chunk:=f.read(1024*1024):
            h.update(chunk)
        return h.hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def one(stdout,tag):
    lines=[v.split("\t")[1:] for v in stdout.splitlines() if v.startswith(tag+"\t")]
    need(len(lines)==1, "missing or multiple "+tag+" records")
    return lines[0]

def stage_records(stdout,tag):
    found={}
    for line in stdout.splitlines():
        if line.startswith(tag+"\t"):
            _,stage,*vals=line.split("\t")
            need(stage not in found,"duplicate "+tag+" "+stage)
            found[stage]=vals
    return found

def plugin_path(value, work):
    p=Path(value)
    return (p if p.is_absolute() else work/p).resolve()

def worker(worker_path, plugin, action, panel, seed, corpus, ref, weight, env):
    params=[str(panel[k]) for k in DIMS]
    args=[str(worker_path),action,str(plugin),panel["spec"],str(corpus),str(seed),str(ref),str(weight),*params,"2"]
    p=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=720)
    reason=p.stderr.strip().splitlines()
    if p.returncode:
        # Keep diagnostics useful without publishing machine-local paths.
        tail=reason[-1].split("exact worker: ")[-1] if reason else "no diagnostic"
        tail=re.sub(r"/[^\s]+","<path>",tail)[:220]
        raise ValueError(f"worker exit {p.returncode}: {tail}")
    return p.stdout

def input_record(stdout,cpu,p):
    a,b,cpu_id,*mods=one(stdout,"INPUT")
    need(all(re.fullmatch("[0-9a-f]{64}",v) for v in (a,b)),"invalid input digest")
    expected=[4*(p[row]*p[ld]+p[col])%64 for row,ld,col in
              (("ar","lda","ac"),("br","ldb","bc"),("cr","ldc","cc"))]
    need(int(cpu_id)==cpu and list(map(int,mods))==expected,"worker affinity or parent origins")
    return [a,b]

def run(work,out,cpu,panel_file,corpus):
    need(os.sched_getaffinity(0)=={cpu},"controller must be pinned to one native Linux CPU")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"panel.csv").resolve()!=panel_file.resolve():
        shutil.copyfile(panel_file,out/"panel.csv")
    panel=read_csv(panel_file);registry=read_csv(work/"methods.csv")
    need(len(set(p["family"] for p in panel))==len(panel)>0,"duplicate case")
    roster={m["method"]:m["role"] for m in registry}
    need(len(roster)==len(registry) and {"fp32-policy","exact-dispatch","openblas"}<=set(roster),"incomplete or duplicate method roster")
    paths={m["method"]:plugin_path(m["plugin"],work) for m in registry}
    native=json.loads((work/"native.json").read_bytes())
    for p in panel:
        for dim in DIMS:
            p[dim]=int(p.get(dim,0))
        need(p["family"] in native,"missing shape witness")
        need(p["spec"] in ("exact","early","late","optdigits","optdigits-images"),"unknown input class")
    wk=work/"exact.worker"; need(wk.is_file(),"missing executable")
    env={**os.environ, **{k:"1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","BLIS_NUM_THREADS","GOTO_NUM_THREADS")}, "MKL_DYNAMIC":"FALSE", "ONEDNN_DEFAULT_FPMATH_MODE":"STRICT", "DNNL_DEFAULT_FPMATH_MODE":"STRICT"}
    for k in ("ONEDNN_VERBOSE","DNNL_VERBOSE","ONEDNN_JIT_DUMP"):
        env.pop(k,None)
    methods={n:{"role":r,"plugin_sha256":sha(paths[n]) if paths[n].is_file() else None} for n,r in roster.items()}
    raw=dict(cpu=cpu,page_bytes=os.sysconf("SC_PAGESIZE"),seeds=list(SEEDS),worker_sha256=sha(wk),panel_sha256=sha(panel_file),methods=methods,panel=panel,bindings={},goldens={},skips={})
    records=[]
    with (out/"observations.jsonl").open("w",buffering=1) as stream:
        def emit(row):
            records.append(row)
            stream.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
        for case_id,p in enumerate(panel):
            f=p["family"]; proof=native[f]
            good=proof["eligible"]
            need(type(good) is bool,"native eligibility must be boolean")
            fallback=plugin_path(proof.get("fallback_plugin") or env.get("EXACT_FALLBACK_PLUGIN") or paths["fp32-policy"],work)
            fast=plugin_path(env.get("EXACT_FAST_PLUGIN") or paths.get("raw-bf16","absent"),work)
            need(fallback.is_file(),"missing backing FP32 policy")
            if good: need(fast.is_file(),"missing prequalified native BF16 plugin")
            binding=dict(native=good,implementation=proof.get("implementation",""),reason=proof.get("reason",""),
                         int8_eligible=bool(proof.get("int8_eligible",False)),int8_implementation=proof.get("int8_implementation",""),
                         fallback_method=proof.get("fallback_method","fp32-policy"),fallback_sha256=sha(fallback),
                         fast_sha256=sha(fast) if fast.is_file() else None)
            raw["bindings"][f]=binding
            case_paths={**paths,"fp32-policy":fallback}
            here={**env,"EXACT_FALLBACK_PLUGIN":str(fallback),"EXACT_FAST_PLUGIN":str(fast),"EXACT_NATIVE_ENABLED":str(int(good))}
            integral=p["spec"] in ("exact","optdigits","optdigits-images")
            skips={}; active=[]
            for name,role in roster.items():
                if not case_paths[name].is_file(): why="registered binary unavailable"
                elif name=="raw-bf16" and not good: why="native BF16 unavailable for layout"
                elif role=="integer-specialist" and not integral: why="fractional input outside INT8 domain"
                elif role=="integer-specialist" and not binding["int8_eligible"]: why="native INT8 unavailable for layout"
                elif role=="binary-specialist" and p["spec"]!="optdigits-images": why="outside registered binary image workload"
                else: active.append(name);continue
                skips[name]=why
            raw["skips"][f]=skips
            raw["goldens"][f]={}
            # Each pair of files is discarded after the three rounds for that input.
            for block,seed in enumerate(SEEDS):
                with tempfile.TemporaryDirectory(prefix="exact-golden-",dir=work) as tmp:
                    ref=Path(tmp)/"signed";weight=Path(tmp)/"weight"
                    gold=worker(wk,paths["openblas"],"gold",p,seed,corpus,ref,weight,here)
                    hashes=input_record(gold,cpu,p)
                    cells=p["m"]*p["n"]
                    count,observed,spots=map(int,one(gold,"GOLD"))
                    need(count==cells and observed==cpu and spots>=min(32,cells) and ref.stat().st_size==weight.stat().st_size==8*cells,"incomplete checked golden")
                    raw["goldens"][f][str(seed)]=dict(inputs=hashes,signed_sha256=sha(ref),weight_sha256=sha(weight),
                                                       outputs=cells,spot_checks=spots)
                    dump(out/"measurement.json",raw)
                    for rnd in (1,2,3):
                        for pos,name in enumerate(order_for(active,case_id,block,rnd)):
                            row=dict(family=f,method=name,seed=seed,round=rnd,position=pos)
                            try:
                                stdout=worker(wk,case_paths[name],"measure",p,seed,corpus,ref,weight,here)
                                need(input_record(stdout,cpu,p)==hashes,"logical input differs from golden")
                                checks=stage_records(stdout,"CHECK")
                                need(list(checks)==["FIRST","WARM"],"missing timed result")
                                stages={}
                                for stage,values in checks.items():
                                    need(len(values)==len(FIELDS),"incomplete output metrics")
                                    vals={k:(str(v) if k=="output_sha256" else float(v) if k in ("seconds","cpu_seconds","max_abs","max_rho") else int(v)) for k,v in zip(FIELDS,values)}
                                    for key in ("seconds","cpu_seconds"):
                                        need(math.isfinite(vals[key]) and vals[key]>0,"invalid timer")
                                    need(vals["cpu"]==cpu and vals["threads"]==1 and vals["checked"]==cells,"bad CPU or output count")
                                    need(re.fullmatch("[0-9a-f]{64}",vals["output_sha256"]),"invalid output digest")
                                    if not math.isfinite(vals["max_rho"]): vals["max_rho"]=None
                                    if not math.isfinite(vals["max_abs"]): vals["max_abs"]=None
                                    stages[stage]=vals
                                extra={tag:stage_records(stdout,tag) for tag in ("ROUTE","BF16","DNN","INT8","BINARY")}
                                row.update(inputs=hashes,stages=stages,traces={k:v for k,v in extra.items() if v})
                            except (OSError,ValueError,subprocess.TimeoutExpired) as e:
                                row["failure"]=str(e)
                            emit(row)
                        print(f"{f}: block {block+1}/2 round {rnd}/3",flush=True)
    result=compute(raw,records)
    write_results(out,result)
    return result

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work",type=Path)
    cli.add_argument("out",type=Path)
    cli.add_argument("cpu",type=int)
    cli.add_argument("--panel",required=True,type=Path)
    cli.add_argument("--corpus",required=True,type=Path)
    a=cli.parse_args()
    run(a.work.resolve(),a.out.resolve(),a.cpu,a.panel.resolve(),a.corpus.resolve())
