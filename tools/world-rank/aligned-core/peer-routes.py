"""Verify the native incumbent and external FastMMW on the frozen aligned panel."""
import csv
import importlib.util
from pathlib import Path
import sys

directory = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("previous", directory / "leaf-wave-routes.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
panel, plugins, vendors, output = map(Path, sys.argv[1:])
with panel.open() as f:
    shapes = [tuple(int(r[k]) for k in previous.DIMS) for r in csv.DictReader(f)]
for group, root, filename, columns in [
    ([previous.HK], plugins, "routes.csv", ["route", "observed", "expected"]),
    (list(previous.FAST), vendors, "fastmmw-routes.csv", ["leaf","leaf_calls","padded","leaf_evidence"]),
]:
    with (output / filename).open("w",newline="") as f:
        table=csv.DictWriter(f,["implementation",*previous.DIMS,*columns,*previous.EXACT,"passed","error"])
        table.writeheader()
        for shape in shapes:
            m,n,k,pad=shape
            matrices=[previous.parent.fixture.Matrix(r,c,pad) for r,c in ((m,k),(k,n),(m,n))]
            for name in group:
                row=dict(zip(previous.DIMS,shape),implementation=name)
                try:
                    row.update(previous.check(name,shape,root,matrices))
                except (ValueError,OSError,RuntimeError,AttributeError) as error:
                    row.update(passed=0,error=str(error))
                table.writerow(row)
                f.flush()
