#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?artifact directory}
include=${2:-/usr/include}
mkdir -p "$out"
"${CXX:-c++}" -std=c++17 -O2 -fno-fast-math -I"$include" "$here/check.cpp" -ldl -o "$out/check"
cpu=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/);print a[1]}' /proc/self/status)
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT DNNL_DEFAULT_FPMATH_MODE=STRICT
uname -a > "$out/host.txt"
lscpu >> "$out/host.txt"
for policy in oneshot cached-plan; do
    taskset -c "$cpu" "$out/check" "$out/onednn-f32-$policy.so" | tee "$out/qualified-$policy.log"
done
