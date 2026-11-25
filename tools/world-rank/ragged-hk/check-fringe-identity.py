"""Prove the fringe partition on every residue and verify the pinned core schedule."""
from collections import Counter
from itertools import product
from pathlib import Path
import argparse
import subprocess
import sys

ap=argparse.ArgumentParser()
ap.add_argument("release",type=Path)
args=ap.parse_args()
subprocess.run([sys.executable,"-B",str(args.release/"third_party/fast-matmul/verify.py")],check=True)
residues=terms=0
for rm,rn,rk in product(range(2),range(3),range(3)):
    mc,nc,kc=4,9,6
    m,n,k=mc+rm,nc+rn,kc+rk
    tally=Counter()
    # Each triple denotes one algebraically distinct term A[i,p] * B[p,j] in C[i,j].
    regions=[
        (range(mc),range(nc),range(kc)),
        (range(mc),range(nc),range(kc,k)),
        (range(mc),range(nc,n),range(k)),
        (range(mc,m),range(n),range(k)),
    ]
    for region in regions: tally.update(product(*region))
    assert len(tally)==m*n*k
    assert all(tally[i,j,p]==1 for i,j,p in product(range(m),range(n),range(k)))
    terms+=len(tally)
    residues+=1
print(f"{residues} remainder triples, {terms} distinct scalar reduction terms across all cases, no overlaps or gaps.")
