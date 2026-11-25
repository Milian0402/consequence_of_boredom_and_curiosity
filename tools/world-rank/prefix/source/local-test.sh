#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?evidence directory required}" && cd "$1" && pwd)
work=$(mktemp -d)
mkdir "$work/vendor" "$work/build"
cc=${CC:-cc}
"$cc" -O3 -march=x86-64 -std=c11 -Wall -Wextra -Werror -fPIC -shared "$root/mock-provider.c" -o "$work/vendor/mkl.so"
bash "$root/build.sh" "$work/vendor" "$work/build"
cat > "$work/panel.csv" <<'CSV'
family,layout,m,n,k,lda,ldb,ldc,ar,ac,br,bc,cr,cc
rows-0,compact,24,1039,97,97,1039,1039,0,0,0,0,0,0
rows-1,cropped,25,1041,109,132,1065,1078,2,11,1,7,1,15
rows-2,cropped,26,1043,111,131,1067,1092,1,7,2,10,3,13
rows-3,cropped,27,1045,113,135,1063,1075,2,13,1,9,2,7
unaligned,cropped,31,1047,99,128,1061,1085,3,17,1,5,3,1
below-width,cropped,53,1009,103,125,1040,1083,0,7,1,0,1,3
CSV
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
taskset -c "$cpu" python3 -B "$root/routes.py" "$work/panel.csv" "$work/vendor" "$work/build" "$out" --mock
python3 - "$out" <<'PY'
import csv,sys
from pathlib import Path
out=Path(sys.argv[1]);rows=list(csv.DictReader((out/'prefix-routes.csv').open()))
assert len(rows)==18
capable=int(rows[0]['intel_avx2'])
if capable:
    assert sum(r['route']=='vector-prefix' for r in rows)==4
    assert sum(r['route']=='provider-split' for r in rows)==4
print('Intel AVX2+FMA',capable,'records',len(rows),'ordinary+trace cells',2*sum(int(r['checked']) for r in rows))
PY
