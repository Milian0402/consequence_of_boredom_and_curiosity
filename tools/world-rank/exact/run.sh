#!/usr/bin/env bash
set -euo pipefail
release=$(cd "$1" && pwd); vendor=$(cd "$2" && pwd); out=$(cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd); work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/exact-study.XXXXXXXX");cpu=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT
export PYTHONDONTWRITEBYTECODE=1
ulimit -c 0
lscpu > "$out/lscpu.txt";printf '%s\n' "$cpu" > "$out/cpu.txt"
git -C "$here" rev-parse HEAD > "$out/benchmark-sha.txt";git -C "$release" rev-parse HEAD > "$out/release-sha.txt"
cp "$here"/{panel.csv,protocol.md} "$out/"
(cd "$here/fixtures/data" && sha256sum --check SHA256SUMS)
bash "$here/build.sh" "$release" "$vendor" "$work" "$out" "$cpu"
if [[ $(uname -m) == x86_64 ]];then
 native=mkl; export EXACT_LARGE_FP32="$work/adapters/release-mkl-aligned-two.so" EXACT_OTHER_FP32="$vendor/mkl-jit.so"
else
 native=armpl; export EXACT_LARGE_FP32="$vendor/fastmmw-1536.so" EXACT_OTHER_FP32="$vendor/armpl.so"
fi
{
 echo 'method,role,plugin'
 echo "exact-dispatch,candidate,$work/adapters/exact-dispatch.so"
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
taskset -c "$cpu" python3 -B "$here/prove.py" "$work" "$vendor" "$out"
taskset -c "$cpu" python3 -B "$here/driver/study.py" "$work" "$out" "$cpu" --panel "$here/panel.csv" --corpus "$here/fixtures/data"
