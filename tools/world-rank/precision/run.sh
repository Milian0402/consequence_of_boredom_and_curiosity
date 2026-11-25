#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$1" && pwd); vendor=$(cd "$2" && pwd); out=$(cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd); rank=$(dirname "$here"); work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/precision-study.XXXXXXXX")
cpu=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT PYTHONDONTWRITEBYTECODE=1
unset ONEDNN_JIT_DUMP ONEDNN_VERBOSE DNNL_VERBOSE
mkdir -p "$work/goldens" "$out/vendor-builds" "$out/build"
cp "$here"/{panel.csv,protocol.md,methods.md} "$out/"
(cd "$out" && sha256sum -c <<'HASH'
5f5735f5d4b464c0f665dd0595fb695384893ba593406c9b897131a1afe069e7  panel.csv
485dbc8b05d57f264af682fb1d444449eaca1d3ef4efc220c528cc9631572c30  protocol.md
HASH
)
lscpu > "$out/lscpu.txt"; getconf PAGE_SIZE > "$out/page-bytes.txt"; printf '%s\n' "$cpu" > "$out/cpu.txt"
printf '%s\n' "${GITHUB_RUN_ID:-local}" > "$out/run-id.txt"
git -C "$here" rev-parse HEAD > "$out/benchmark-sha.txt"; git -C "$base" rev-parse HEAD > "$out/release-sha.txt"
(cd "$rank" && git ls-files -z precision views/worker.c ragged-hk/golden.h worker.c vendor.h | xargs -0 sha256sum) > "$out/study-sources.sha256"
for f in "$vendor"/*-build.txt "$vendor"/*-configuration.txt "$vendor"/*-identity.sha256 "$vendor"/tblis-status.txt;do if [[ -f $f ]];then cp "$f" "$out/vendor-builds/";fi;done
bash "$here/core/build-plugins.sh" "$base" "$vendor" "$work" "$out" "$cpu"
# The two older Winograd levels provide public, vendor-assisted competition on large rectangles.
export PRECISION_VENDOR_WORKER_SOURCE=$here/worker.c
status=0;bash "$rank/leaf-confirm/vendor-release.sh" "$base" "$vendor" "$work" "$out" "$cpu" > "$out/build/released-profiles.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/build/released-profiles-status.txt"
if [[ -s $work/vendor-release.csv ]];then
    cp "$work/vendor-release.csv" "$out/build/original-profiles.csv"
    python3 - "$work/vendor-release.csv" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1]);rs=list(csv.DictReader(p.open()));fields=['method','worker','trace_worker','plugin','role'];good=[]
for r in rs:
    b=Path(r['worker']).with_suffix('.precision.worker')
    if b.is_file(): r['worker']=str(b);good.append(r)
with p.open('w') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(good)
PY
fi
unset PRECISION_VENDOR_WORKER_SOURCE
taskset -c "$cpu" python3 -B "$here/core/admit.py" "$work" "$out" "$cpu"
taskset -c "$cpu" python3 -B "$here/study.py" "$work" "$vendor" "$out" "$cpu"
python3 -B "$here/score.py" "$out"
