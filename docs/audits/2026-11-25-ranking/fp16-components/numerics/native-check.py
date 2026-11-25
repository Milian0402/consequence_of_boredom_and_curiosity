"""Repeat a component-path measurement on an applicable Arm Mac with a rebuilt DSO."""
import argparse
from locate import reference
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo')
p.add_argument('--library',required=True)
p.add_argument('--control-library')
p.add_argument('--label',required=True)
p.add_argument('--out',required=True)
args=p.parse_args()
reference(args.repo)
import evaluate
evaluate.MODES=(1,3,4,6,9)
args.modes=None
evaluate.run(args)
