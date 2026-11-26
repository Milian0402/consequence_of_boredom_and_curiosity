#!/usr/bin/env bash
set -euo pipefail
release=$(cd "$1" && pwd); vendor=$(cd "$2" && pwd); out=$(cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd); work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/exact-study.XXXXXXXX");cpu=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT
export PYTHONDONTWRITEBYTECODE=1
ulimit -c 0
lscpu > "$out/lscpu.txt";printf '%s\n' "$cpu" > "$out/cpu.txt"
git -C "$here" rev-parse HEAD > "$out/benchmark-sha.txt";git -C "$release" rev-parse HEAD > "$out/release-sha.txt"
panel_dir=$here
if [[ ${EXACT_CHECKED_S8:-0} == 1 ]]; then panel_dir=$here/s8; fi
cp "$panel_dir"/{panel.csv,protocol.md} "$out/"
(cd "$here/fixtures/data" && sha256sum --check SHA256SUMS)
bash "$here/build.sh" "$release" "$vendor" "$work" "$out" "$cpu"
if [[ $(uname -m) == x86_64 ]];then
 native=mkl; export EXACT_LARGE_FP32="$work/adapters/release-mkl-aligned-two.so" EXACT_OTHER_FP32="$vendor/mkl-jit.so"
else
 native=armpl; export EXACT_LARGE_FP32="$vendor/fastmmw-1536.so" EXACT_OTHER_FP32="$vendor/armpl.so"
fi
{
 echo 'method,role,plugin'
 if [[ ${EXACT_CHECKED_S8:-0} == 1 ]]; then
  echo "s8-dispatch,candidate,$work/adapters/s8-dispatch.so"
  echo "exact-dispatch,checked-bf16-control,$work/adapters/exact-dispatch.so"
 else
  echo "exact-dispatch,candidate,$work/adapters/exact-dispatch.so"
 fi
 echo "raw-bf16,unchecked-integer-reference,$work/adapters/raw-bf16.so"
 echo "fp32-policy,fp32-backing,$work/adapters/fp32-policy.so"
 echo "cob-standalone,standalone,$work/adapters/cob-standalone.so"
 for p in openblas "$native" blasfeo fastmmw-1536 fastmmw-768;do echo "$p,independent-fp32,$vendor/$p.so";done
 if [[ $native == mkl ]];then
  for p in mkl-jit aocl;do echo "$p,independent-fp32,$vendor/$p.so";done
  echo "release-mkl-aligned-two,release-vendor-assisted,$work/adapters/release-mkl-aligned-two.so"
 fi
 echo "onednn-oneshot,independent-fp32,$vendor/onednn/onednn-f32-oneshot.so"
 echo "onednn-cached-plan,independent-retained,$vendor/onednn/onednn-f32-cached-plan.so"
 echo "onednn-int8,integer-specialist,$work/adapters/onednn-int8.so"
 echo "bitset-popcount,binary-specialist,$work/adapters/bitset-popcount.so"
} > "$work/methods.csv"
mkdir -p "$out/vendor-builds"
for f in "$vendor"/*-build.txt "$vendor"/*-configuration.txt;do if [[ -f $f ]];then cp "$f" "$out/vendor-builds/";fi;done
if [[ ${EXACT_CHECKED_S8:-0} == 1 ]]; then
 mkdir -p "$out/checked-entry"
 taskset -c "$cpu" bash "$here/s8/test.sh" "$vendor/onednn/prefix" "$work/checked-test"
 taskset -c "$cpu" "$work/checked-test/check" "$work/adapters/s8-dispatch.so" auto > "$out/checked-entry/timed-binary.txt"
 if grep -q "mode=positive" "$out/checked-entry/timed-binary.txt"; then
  taskset -c "$cpu" "$work/checked-test/check" "$work/checked-test/diagnostic.so" faults > "$out/checked-entry/atomic-faults.txt"
 else
  echo "Small kernel unavailable on this host; native fault injections skipped" > "$out/checked-entry/atomic-faults.txt"
 fi
 EXACT_INT8_DNNL_PATH=/absent/missing-library.so taskset -c "$cpu" "$work/checked-test/check" "$work/adapters/s8-dispatch.so" missing > "$out/checked-entry/missing-library.txt"
 cp "$work/checked-test/host-transfer.txt" "$out/checked-entry/"
 sha256sum "$work/checked-test/check" "$work/checked-test/ordinary.so" "$work/checked-test/diagnostic.so" "$work/adapters/s8-dispatch.so" > "$out/checked-entry/artifacts.sha256"
fi
taskset -c "$cpu" python3 -B "$here/prove.py" "$work" "$vendor" "$out"
taskset -c "$cpu" python3 -B "$here/driver/study.py" "$work" "$out" "$cpu" --panel "$panel_dir/panel.csv" --corpus "$here/fixtures/data"
