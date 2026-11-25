"""Build the pinned and proposed native paths in a disposable directory and compare all results."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REF = "f8d543176674683704860fa88cb62535f0f8a1aa"
SCHEDULE = Path("tools/world-rank/finite/hk233.inc")


def change(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"fixture hook {old!r} not unique in {path}")
    path.write_text(text.replace(old,new))


def tiny_thresholds(directory):
    header = directory / "src/cob_hk15.inc"
    text = header.read_text()
    if text.count("*768") != 3 or not re.search(r"\b960\b", text):
        raise ValueError("HK threshold locations changed")
    header.write_text(re.sub(r"\b960\b", "32", text.replace("*768", "*8")))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("repository", type=Path)
    cli.add_argument("candidate", type=Path)
    cli.add_argument("stage", choices=("tiny", "full", "panel"))
    args = cli.parse_args()
    compiler = os.environ.get("CC", "cc")
    paths = ["include", "src", str(SCHEDULE), "tools/world-rank/challenger-flags.txt"]
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        work = Path(temp)
        baseline = work/"old"
        candidate = work/"candidate"
        baseline.mkdir()
        subprocess.run(["git","-C",args.repository,"archive","--output",work/"base.tar",REF,*paths],check=True)
        subprocess.run(["tar","-xf",work/"base.tar","-C",baseline],check=True)
        for name in paths:
            p = args.candidate/name
            if p.is_dir():
                shutil.copytree(p, candidate/name)
            else:
                (candidate/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,candidate/name)
        for rel in ["src/gemm.c","src/cob_hk15.inc","src/cob_fused_strassen.inc","src/cob_hk_pack_b.inc",str(SCHEDULE)]:
            print("candidate_sha256",hashlib.sha256((candidate/rel).read_bytes()).hexdigest(),rel,flush=True)
        flags = ["-std=c11","-O3","-fPIC","-DCOB_ENABLE_PORTABLE_STRASSEN=1","-DCOB_ENABLE_RECURSIVE_STRASSEN=1",
                 "-DCOB_ENABLE_HK15=1","-DCOB_ENABLE_HK15_PARENT_960=1"]
        flags += ["-D"+line for line in (baseline/"tools/world-rank/challenger-flags.txt").read_text().splitlines() if line]
        if platform.machine() in ("aarch64","arm64"):
            flags += ["-mcpu=native","-DCOB_USE_NEON=1","-DCOB_ENABLE_N2_SMALL=1"]
        else:
            flags += ["-march=native","-DCOB_ENABLE_X86_MEDIUM_ROUTE=1"]
        if platform.system() == "Darwin":
            flags += ["-DCOB_DISABLE_APPLE_AMX=1","-DCOB_DISABLE_APPLE_SME=1"]
        if args.stage == "tiny":
            for tree in (baseline,candidate):
                tiny_thresholds(tree)
            flags += ["-DCOB_FUSED_STRASSEN_MIN_HARMONIC_DIM=48"]
        print("stage",args.stage,"baseline",REF,"flags",*flags,flush=True)
        subprocess.run([compiler,"-std=c11","-O2",HERE/"check.c","-o",work/"check","-ldl","-lm"],check=True)

        def compile_entry(tree, name, extra=(), checker=False):
            source = tree/"src/gemm.c"
            link = ["-dynamiclib"] if platform.system()=="Darwin" else ["-shared"]
            if checker:
                wrapper = work/"pack-test.c"
                wrapper.write_text('#include "gemm.c"\n#include "pack_test.c"\n')
                source,link = wrapper,[]
            output = work/name
            subprocess.run([compiler,*flags,*extra,*link,"-I",tree/"include","-I",tree/"src","-I",HERE,
                            source,"-o",output,"-lm"],check=True)
            return output

        flag = "-DCOB_ENABLE_HK_PACK_B_FORMS=1"
        trace = "-DCOB_HK_PACK_B_TRACE=1"
        old = compile_entry(baseline,"old.so")
        plain = compile_entry(candidate,"plain.so",[flag])
        traced = compile_entry(candidate,"trace.so",[flag,trace])

        def check(m,n,k,pad,mode,counts,a=plain,b=traced,basis="full",seed=712297):
            subprocess.run([work/"check",old,a,b,*map(str,(m,n,k,pad,seed,mode,*counts,basis))],check=True)

        if args.stage == "panel":
            with (HERE/"panel.csv").open(newline="") as stream:
                layouts = list(csv.DictReader(stream))
            for row in layouts:
                shape = [int(row[k]) for k in ("m","n","k","pad")]
                counts = (0,0,0) if row["family"] == "off-route-transpose" else (4,28,0)
                check(*shape,"random",counts,basis="repeat")
            return
        if args.stage == "full":
            check(1926,3375,2823,5,"random",(4,28,0),basis="repeat")
            return
        subprocess.run([compile_entry(candidate,"pack-test",[flag],checker=True)],check=True)
        for m,n,k,pad in ((66,105,99,5),(72,108,108,0),(64,96,96,3),(260,396,405,3)):
            for mode in ("random","cancel","zeros"):
                check(m,n,k,pad,mode,(4,28,0))
        # Small children at the ordinary cutoff retain the dispatcher's
        # earlier native small route. The same inputs still require bit parity.
        for mode in ("random","cancel","zeros"):
            check(96,144,144,3,mode,(0,0,4))
        check(30,45,45,1,"random",(0,0,4))
        check(67,105,99,5,"random",(0,0,0))
        # Fail exactly one nominated prepacking allocation, or every one. Original scratch remains available.
        target = candidate/"src/cob_fused_strassen.inc"
        change(target,"float* work = cob_aligned_alloc(64, slots*sizeof(float));",
               "static _Thread_local int fixture_requests;\n"
               "    int reject = form && (++fixture_requests == FIXTURE_FAIL_REQUEST || FIXTURE_FAIL_REQUEST < 0);\n"
               "    float* work = reject ? NULL : cob_aligned_alloc(64, slots*sizeof(float));")
        for fail,counts in ((1,(3,21,1)),(-1,(0,0,4))):
            hook = f"-DFIXTURE_FAIL_REQUEST={fail}"
            a = compile_entry(candidate,f"fail{fail}.so",[flag,hook])
            b = compile_entry(candidate,f"fail{fail}-trace.so",[flag,trace,hook])
            check(66,105,99,5,"cancel",counts,a,b)


if __name__ == "__main__":
    main()
