"""Record the independent FastMMW adapters' exact-output, padding and native leaf routes."""
import csv
import importlib.util
from pathlib import Path
import sys


def run(panel, vendors, output):
    spec = importlib.util.spec_from_file_location("leaf_routes", Path(__file__).resolve().parents[1] / "leaf-wave-routes.py")
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    with panel.open(newline="") as stream:
        shapes = [tuple(int(r[d]) for d in leaf.DIMS) for r in csv.DictReader(stream)]
    available = [name for name in leaf.FAST
                 if (vendors / (name + ".so")).is_file() and (vendors / (name + ".trace.so")).is_file()]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["implementation", *leaf.DIMS, "leaf", "leaf_calls", "padded",
                                          "leaf_evidence", *leaf.EXACT, "passed", "error"])
        writer.writeheader()
        for shape in shapes:
            if not available:
                break
            m, n, k, pad = shape
            matrices = [leaf.parent.fixture.Matrix(r, c, pad) for r,c in ((m,k),(k,n),(m,n))]
            for name in available:
                row = dict(zip(leaf.DIMS, shape), implementation=name)
                try:
                    row.update(leaf.check(name, shape, vendors, matrices))
                except (ValueError, OSError, RuntimeError, AttributeError) as error:
                    row.update(passed=0, error=str(error))
                writer.writerow(row)
                stream.flush()


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
