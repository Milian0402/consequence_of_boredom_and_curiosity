"""Build a separate fault-injected trace library; the shipping binary remains pristine."""
import argparse
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__)
for n in ('trace_leaf_c','driver_o','adapter_c','include_dir','out_dylib'):p.add_argument(n,type=Path)
a=p.parse_args()
h=Path(__file__).resolve().parent
o=a.out_dylib.with_suffix('.o')
subprocess.run(['cc','-O3','-std=c11','-DCOB_ENABLE_APPLE_MEDIUM_WIDE=1',
                '-mcpu=apple-m4','-march=armv8.7-a+sme+sme2','-include',h/'fault-prefix.h',
                '-c',a.trace_leaf_c,'-o',o],check=True)
subprocess.run(['cc','-O3','-dynamiclib','-I'+str(a.include_dir),a.adapter_c,a.driver_o,o,'-o',a.out_dylib],check=True)
