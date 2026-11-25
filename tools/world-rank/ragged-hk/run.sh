#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$1" && pwd);vendor=$(cd "$2" && pwd);out=$(mkdir -p "$3" && cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd);work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/cob-ragged-hk.XXXXXXXX")
cpu=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE GOTO_NUM_THREADS=1
cp "$here"/{panel.csv,protocol.md,geometry.csv,novelty.json} "$out/"
python3 -B "$here/check-geometry.py" | cmp - "$out/geometry.csv"
printf '%s  %s\n' 51ac3e957ad14d06eadcbea25ba17a60c75d5a237e2c2c64cf8d951ceacc5460 "$out/protocol.md" | sha256sum -c
python3 -B "$here/check-fringe-identity.py" "$base" > "$out/algebra.txt"
git rev-parse HEAD > "$out/benchmark-sha.txt";git -C "$base" rev-parse HEAD HEAD^{tree} > "$out/production-sha.txt"
lscpu > "$out/lscpu.txt";getconf PAGE_SIZE > "$out/page-bytes.txt";gcc --version > "$out/compiler.txt";printf '%s\n' "$cpu" > "$out/pinned-cpu.txt"
(cd "$here" && sha256sum candidate.patch *.c *.h *.py *.sh *.md panel.csv geometry.csv novelty.json ../worker.c ../views/worker.c ../vendor-public/trace.c ../signed/trace.c ../openblas/build.sh ../vendors/build.sh ../fastmmw/native-ci.sh ../release-replay/restore-peers.sh) > "$out/study-source.sha256"
mkdir -p "$out/vendor-builds" "$work/goldens"
for f in "$vendor"/*-build.txt "$vendor"/tblis-status.txt "$vendor"/onednn/qualified-*.log;do if [[ -f $f ]];then cp "$f" "$out/vendor-builds/";fi;done
bash "$here/build.sh" "$base" "$work" "$out" "$cpu"
bash "$here/vendor-release.sh" "$base" "$vendor" "$work" "$out" "$cpu" || echo "See the per-provider release build receipts" >&2
taskset -c "$cpu" python3 -B "$here/study.py" "$vendor" "$work" "$out" "$cpu"
python3 -B "$here/score.py" "$out"
