#!/usr/bin/env bash
set -euo pipefail
source=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$dir")
repo=$(git -C "$shared" rev-parse --show-toplevel)
release=a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/frontier.XXXXXXXX")
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
[[ $(git -C "$source" rev-parse HEAD) == "$release" ]]
git -C "$source" diff --quiet HEAD --
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT RANK_VERIFY_INPUTS=1 RANK_MEDIUM_BATCH=1
export RANK_HOST_BLAS_DIR="$vendor"
cp "$dir"/{panel.csv,protocol.md,peer-extension.md,novelty.json,candidate.sha256} "$out/"
for name in openblas-build.txt aocl-configuration.txt blasfeo-build.txt mkl-build.txt mkl-jit-build.txt libxsmm-build.txt; do
    if [[ -f $vendor/$name ]]; then cp "$vendor/$name" "$out/"; fi
done
for name in qualified-oneshot.log qualified-cached-plan.log; do
    cp "$vendor/onednn/$name" "$out/onednn-$name"
done
{ uname -a; lscpu; cc --version; cmake --version; git -C "$repo" rev-parse HEAD; } > "$out/host.txt"
[[ -s $vendor/onednn/qualified || -f $vendor/onednn/qualified ]]
bash "$dir/build.sh" "$source" "$build" "$out" "$cpu" > "$out/frontier-build.log" 2>&1 || { tail -60 "$out/frontier-build.log"; exit 1; }
taskset -c "$cpu" python3 -B "$dir/routes.py" "$out/panel.csv" "$build" "$out"
printf 'implementation,reason\n' > "$out/omissions.csv"
if taskset -c "$cpu" bash "$shared/host-pad/build.sh" "$vendor" "$build/host" "$out/panel.csv" > "$out/host-pad-build.log" 2>&1; then
    cp "$build/host"/{host-pad-routes.csv,host-pad-witness.csv,host-pad-provenance.txt} "$out/"
    has_host=1
else
    has_host=0
    printf '%s\n' 'host-direct,provider_unavailable_or_route_rejected' 'host-pad8,provider_unavailable_or_route_rejected' >> "$out/omissions.csv"
    printf 'implementation,family,m,n,k,pad,passed\n' > "$out/host-pad-routes.csv"
    tail -30 "$out/host-pad-build.log"
fi
if [[ -f $vendor/libxsmm-dispatch.so && -f $vendor/libxsmm-retained.so ]]; then
    taskset -c "$cpu" python3 -B "$dir/xsmm-routes.py" "$out/panel.csv" "$vendor" "$out/libxsmm-routes.csv"
    has_xsmm=1
else
    has_xsmm=0
    printf '%s\n' 'libxsmm-dispatch,native_build_unavailable' 'libxsmm-retained,native_build_unavailable' >> "$out/omissions.csv"
    printf 'implementation,family,m,n,k,pad,code_bytes,admitted,reason\n' > "$out/libxsmm-routes.csv"
fi
printf 'implementation,role,policy,sha256\n' > "$out/methods.csv"
declare -a entries=()
method(){
    local name=$1 role=$2 policy=$3 path=$4
    [[ -f $path ]]
    printf '%s,%s,%s,%s\n' "$name" "$role" "$policy" "$(sha256sum "$path" | cut -d' ' -f1)" >> "$out/methods.csv"
    if [[ $name == openblas ]]; then return; fi
    case $role in
      independent) entries+=(--vendor "$name=$path") ;;
      standalone) entries+=(--candidate "$name=linked-public:$path") ;;
      vendor-assisted) entries+=(--candidate "$name=direct:$path") ;;
      *) return 2 ;;
    esac
}
method openblas independent one-shot "$vendor/openblas.so"
for name in mkl mkl-jit aocl blasfeo; do method "$name" independent one-shot "$vendor/$name.so"; done
method onednn-oneshot independent one-shot "$vendor/onednn/onednn-f32-oneshot.so"
method onednn-cached-plan independent retained "$vendor/onednn/onednn-f32-cached-plan.so"
if ((has_xsmm)); then
    method libxsmm-dispatch independent one-shot "$vendor/libxsmm-dispatch.so"
    method libxsmm-retained independent retained "$vendor/libxsmm-retained.so"
fi
for name in frontier-incumbent stream-packed-a; do method "$name" standalone one-shot "$build/$name.worker"; done
if ((has_host)); then
    for name in host-direct host-pad8; do method "$name" vendor-assisted one-shot "$build/host/$name.so"; done
fi
python3 -B - "$out" "$repo" "$release" "$cpu" <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]); sha=lambda f:hashlib.sha256((out/f).read_bytes()).hexdigest()
receipt=dict(production=sys.argv[3],benchmark=subprocess.check_output(['git','-C',sys.argv[2],'rev-parse','HEAD'],text=True).strip(),
        cpu=int(sys.argv[4]),run=int(os.environ['GITHUB_RUN_ID']),attempt=int(os.environ['GITHUB_RUN_ATTEMPT']),replica=int(os.environ['FRONTIER_REPLICA']),
        source_files={f:sha(f) for f in ('panel.csv','protocol.md','peer-extension.md','novelty.json','methods.csv','frontier-sources.json','candidate.sha256')})
assert receipt['source_files']['panel.csv']=='b6d451644a0a979e1b09796ff22c6b4ff24b537a7a701e68afad72ada14d5190'
assert receipt['source_files']['protocol.md']=='91df1250132c20dc7a32de34985956fbe6ebaffd4cbf0536b617df044041b83a'
(out/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
study_status=0
taskset -c "$cpu" python3 -B "$shared/study.py" "$build/frontier-incumbent.worker" "$out/panel.csv" "$out" "$vendor/openblas.so" \
    --phase frontier-x86 --rounds 3 --stress-all "${entries[@]}" || study_status=$?
printf '%s\n' "$study_status" > "$out/study-status.txt"
python3 -B "$dir/score.py" "$out"
