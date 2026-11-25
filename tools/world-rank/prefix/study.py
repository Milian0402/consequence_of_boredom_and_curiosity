"""Use the existing public-view worker with the registered prefix study roster."""
from pathlib import Path
import runpy

directory = Path(__file__).resolve().parent
runpy.run_path(str(directory.parent/"grouped/study.py"), run_name="__main__", init_globals={
    "method_override": ("openblas","mkl","mkl-jit","aocl","blasfeo","onednn-oneshot","onednn-cached-plan",
                        "libxsmm-dispatch","libxsmm-retained","stream-incumbent","host-forward","host-split8","host-prefix8"),
    "subject_override": ("stream-incumbent",),
    "assisted_override": ("host-forward","host-split8","host-prefix8"),
    "seed_override": (1047891527,1936510481),
    "receipt_files": ("panel.csv","protocol.md","novelty.json","methods.csv","prefix-source.sha256","grouped-sources.json")
})
