"""Require exact structure and scalars, allowing four binary64 units for platform math."""
import csv,json,math
from pathlib import Path

def consistent(a,b):
    if type(a) != type(b):raise ValueError("Replayed scalar type differs")
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError("Replayed keys differ")
        for k in a:consistent(a[k],b[k])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError("Replayed length differs")
        for x,y in zip(a,b):consistent(x,y)
    elif isinstance(a,float):
        if not(math.isfinite(a) and math.isfinite(b)) or abs(a-b)>4*max(math.ulp(a),math.ulp(b)):
            raise ValueError(f"Replayed floating statistic differs: {a}, {b}")
    elif a!=b:raise ValueError(f"Replayed discrete value differs: {a}, {b}")

def file_pair(a,b):
    if a.suffix==".json":
        consistent(json.loads(a.read_text()),json.loads(b.read_text()));return
    def table(p):
        with p.open(newline="") as h:
            rows=list(csv.reader(h))
        return [[(float(v) if any(c in v for c in (".","e","E")) and _number(v) else v) for v in row] for row in rows]
    consistent(table(a),table(b))

def _number(v):
    try:float(v);return True
    except ValueError:return False
