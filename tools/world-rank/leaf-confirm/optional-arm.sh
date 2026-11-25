#!/usr/bin/env bash
set -euo pipefail
vendor=$(cd "$1" && pwd)
work=$(cd "$2" && pwd)
out=$(cd "$3" && pwd)
cpu=$4
here=$(cd "$(dirname "$0")" && pwd)
worker=$work/released-hk15/ordinary.worker
[[ -x $worker ]]
printf 'method,worker,plugin,trace_worker,trace_plugin\n' > "$work/optional-arm.csv"
mkdir -p "$out/arm-peer"
printf 'method,status,build_rc\n' > "$out/arm-peer/admission.csv"
for method in blis acl; do
    build="$work/arm-$method"
    mkdir -p "$build"
    if taskset -c "$cpu" bash "$here/$method/build.sh" "$vendor" "$build" >"$out/arm-peer/$method-install.log" 2>&1; then
        printf '%s,%s,%s,%s,%s\n' "$method" "$worker" "$vendor/$method.so" "$worker" "$vendor/$method-trace.so" >> "$work/optional-arm.csv"
        printf '%s,build_and_correctness_passed,0\n' "$method" >> "$out/arm-peer/admission.csv"
    else
        rc=$?
        printf '%s,build_or_smoke_failed,%s\n' "$method" "$rc" >> "$out/arm-peer/admission.csv"
        tail -15 "$out/arm-peer/$method-install.log" >&2
    fi
    for item in "$vendor/$method-build.txt" "$vendor/$method-ordinary-dynamic.txt" "$vendor/$method-ordinary-exports.txt" "$vendor/$method-trace-exports.txt" "$vendor/$method-binaries.sha256"; do
        [[ ! -f $item ]] || cp "$item" "$out/arm-peer/"
    done
done
