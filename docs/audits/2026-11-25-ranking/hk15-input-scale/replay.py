"""Validate per-fixture counters and summarize every output recorded on the M5."""
import csv
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [("outer", shift) for shift in (0, 4, 5, 6, 7, 8, 12, 16, 24)]
cases += [("inside", shift) for shift in (8, 16, 24)]
cases += [("scatter", 8), ("scatter", 16), ("spike", 12), ("zero-axes", 12),
          ("cancel", 10), ("global", 127), ("subnormal", 0), ("complement", 8),
          ("complement", 16), ("complement-tile", 16)]

def summarize(path, size, exact_required=False):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["pattern"], int(r["shift"])) for r in rows] == cases, path
    for row in rows:
        m, n, k = map(int, (row["m"], row["n"], row["k"]))
        assert (m, n, k) == size
        count = int(row["cells"])
        assert count == m*n
        wrong = int(row["wrong"])
        assert 0 <= wrong <= count
        for column in ("over_0_002", "gt_1pct", "nonfinite", "zeros_lost", "negative_from_positive"):
            assert 0 <= int(row[column]) <= wrong
        maximum = float(row["max_abs"])
        assert not math.isnan(maximum) and maximum >= 0
        assert (maximum > .002) == bool(int(row["over_0_002"]))
        if exact_required:
            assert wrong == 0 and maximum == 0, (path, row["pattern"], row["shift"])
    return dict(layout=size, fixtures=len(rows), cells=sum(int(r["cells"]) for r in rows),
                incorrect=sum(int(r["wrong"]) for r in rows),
                nonfinite=sum(int(r["nonfinite"]) for r in rows),
                failed_absolute=sum(int(r["over_0_002"]) > 0 for r in rows))

if __name__ == "__main__":
    suites = {}
    for shape in ((1931,2903,2881),(2117,3071,3013)):
        for name in ("issued","openblas","guarded"):
            path = root/f"{name}-{shape[0]}.csv"
            if name == "guarded" and not path.exists():
                continue
            suites[path.name] = summarize(path, shape, exact_required=name=="openblas")
    print(json.dumps(suites, indent=2))
