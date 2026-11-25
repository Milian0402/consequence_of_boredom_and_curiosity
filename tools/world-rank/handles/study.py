"""Run the pinned complete-lifecycle/retained-field comparison on public views."""
from pathlib import Path
import runpy

directory=Path(__file__).resolve().parent
runpy.run_path(str(directory.parent/"grouped/study.py"),run_name="__main__",init_globals={
    "method_override":("openblas","mkl","mkl-jit","aocl","blasfeo","onednn-oneshot","onednn-cached-plan",
                       "libxsmm-dispatch","libxsmm-retained","stream-incumbent","mkl-handle-oneshot","mkl-handle-prepared"),
    "subject_override":("stream-incumbent",),
    "retained_override":("onednn-cached-plan","libxsmm-retained","mkl-handle-prepared"),
    "handle_methods":("mkl-handle-oneshot","mkl-handle-prepared"),
    "seed_override":(1587984119,384761239),
    "receipt_files":("panel.csv","protocol.md","methods.csv","handle-source.sha256","grouped-sources.json")
})
