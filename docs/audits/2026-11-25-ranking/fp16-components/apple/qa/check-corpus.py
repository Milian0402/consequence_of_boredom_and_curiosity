"""Match actual one-shot results to the registered common SME component run."""
import argparse
import csv
import ctypes as ct
import json
from pathlib import Path
import struct
import sys

parser=argparse.ArgumentParser()
parser.add_argument('library',type=Path)
parser.add_argument('fixtures',type=Path)
parser.add_argument('scores',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.fixtures.resolve()))
from fixtures import build
source=json.loads(args.scores.read_text())
lib=ct.CDLL(str(args.library.resolve()))
ptr=ct.POINTER(ct.c_float)
for name in ('apple_fp16_raw','apple_fp16_three'):
    getattr(lib,name).argtypes=[ct.c_int]*3+[ptr,ct.c_int,ptr,ct.c_int,ptr,ct.c_int]
rows=[]
for row in source['cases']:
    if row['mode'] not in ('1','3'): continue
    m,n,k,a,b=build(row['case'],row['seed'])
    def floats(bits):
        return (ct.c_float*len(bits)).from_buffer_copy(struct.pack(f'<{len(bits)}I',*bits))
    av,bv=floats(a),floats(b)
    c=(ct.c_float*(m*n))(*[711.]*(m*n)); before=bytes(c)
    f=lib.apple_fp16_raw if row['mode']=='1' else lib.apple_fp16_three
    ok=f(m,n,k,av,k,bv,n,c,n)
    route=lib.apple_fp16_last_route()
    assert ok or (bytes(c)==before and route==0)
    assert not ok or k==0 or route==int(row['mode'])
    match=(bytes(c)==bytes.fromhex(row['output_words_le_hex'])) if ok else None
    if row['category'] in ('normal','cancellation'): assert ok and match, (row['case'],row['mode'])
    rows.append({'case':row['case'],'category':row['category'],'seed':row['seed'],'mode':row['mode'],
                 'accepted':ok,'native_products':route,'status':lib.apple_fp16_last_status(),
                 'match_common_output_bits':int(match) if match is not None else ''})
with args.output.open('w',newline='') as f:
    csvw=csv.DictWriter(f,fieldnames=tuple(rows[0]));csvw.writeheader();csvw.writerows(rows)
from collections import Counter
print(Counter((r['category'],r['accepted'],r['match_common_output_bits']) for r in rows))
