#!/usr/bin/env bash
# Build direct, zero-BLAS-dependency plugin binaries from the pinned public COB.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "${1:?usage: build.sh REPO OUTPUT}" && pwd)
mkdir -p "${2:?missing output directory}"
out=$(cd "$2" && pwd)
[[ $# == 2 ]] || exit 2
rev=0cff6ec089a43e65eeb3bfefd12a1fc68f205be4
root="$out/winograd-cob-build"
mkdir -p "$root/base"
base="$root/base"
git -C "$repo" archive "$rev" src include tools/world-rank/finite/hk233.inc \
    tools/world-rank/vendor.h tools/world-rank/challenger-flags.txt tools/world-rank/hybrid_next.c | tar -x -C "$base"
# Only the private consumer changes: thread the selected leaf function through
# the original schedule and fringe calls. Vendor baselines use the original file.
patch -s -d "$base" -p1 < "$here/schedule.patch"
cc=${CC:-cc}
common=(-std=c11 -O3 -fPIC -fvisibility=hidden -I"$base/include" -I"$base/src" -I"$base/tools/world-rank")
native=()
while IFS= read -r define; do [[ -z $define ]] || native+=("-D$define"); done < "$base/tools/world-rank/challenger-flags.txt"
std_leaf=(-DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1)
case $(uname -m) in
    x86_64) common+=(-march=native); native+=(-DCOB_ENABLE_X86_MEDIUM_ROUTE=1) ;;
    aarch64|arm64) common+=(-mcpu=native); native+=(-DCOB_USE_NEON=1 -DCOB_ENABLE_N2_SMALL=1) ;;
    *) echo "Expected a native x86-64 or arm64 host" >&2; exit 2 ;;
esac
case $(uname -s) in
    Linux) link=(-shared '-Wl,-z,defs') ;;
    Darwin) link=(-dynamiclib); common+=(-D_DARWIN_C_SOURCE); native+=(-DCOB_DISABLE_APPLE_AMX=1 -DCOB_DISABLE_APPLE_SME=1) ;;
    *) exit 2 ;;
esac
log="$out/winograd-cob-build.txt"
printf 'source=%s\nschedule=tools/world-rank/hybrid_next.c\ncommon=%s\nnative=%s\npublic-leaf=%s\n' \
    "$rev" "${common[*]}" "${native[*]}" "${std_leaf[*]}" > "$log"
"$cc" --version >> "$log"
# Give every symbol in each independent COB object a private name. The dynamic
# ABI exports only the standalone plugin itself (plus counters in trace builds).
public_helpers=(cob_sgemm_free_packed_a cob_sgemm_free_packed_b cob_sgemm_pack_a
    cob_sgemm_pack_b cob_sgemm_ref_rowmajor cob_sgemm_rowmajor_packed_ab cob_sgemm_rowmajor_packed_b)
for kind in plain trace; do
    objects=()
    adapter_flags=()
    if [[ $kind == trace ]]; then adapter_flags+=(-DCOB_WINOGRAD_TRACE=1); fi
    for budget in full one zero; do
        stage="$root/$budget-$kind.o"
        opts=(); tracer=()
        renames=("-Drank_hk15_route_count=cob_winograd_${budget}_route")
        for sym in "${public_helpers[@]}"; do renames+=("-D$sym=${sym}_${budget}"); done
        source="$base/src/gemm.c"
        case $budget in
            full) opts=("${std_leaf[@]}"); entry=cob_winograd_full_public ;;
            one) opts=(-DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=0
                    -DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_PARENT_960=1)
                source="$here/one_leaf.c"; entry=cob_winograd_unused_hk_entry ;;
            zero) opts=(-DCOB_ENABLE_PORTABLE_STRASSEN=0 -DCOB_ENABLE_RECURSIVE_STRASSEN=0 -DCOB_ENABLE_HK15=0)
                entry=cob_winograd_zero_public ;;
        esac
        if [[ $kind == trace && $budget != zero ]]; then
            tracer+=(-DCOB_HK15_CHILD_TRACE=1 -DCOB_WINOGRAD_TRACE=1)
        fi
        "$cc" "${common[@]}" "${native[@]}" "${opts[@]}" "${renames[@]}" "${tracer[@]}" \
            -Dcob_sgemm_rowmajor="$entry" -c "$source" -o "$stage"
        objects+=("$stage")
    done
    for depth in 1 2; do
        stem="winograd-cob-$depth"
        if [[ $kind == trace ]]; then stem="$stem.trace"; fi
        "$cc" "${common[@]}" "${adapter_flags[@]}" -DCOB_WINOGRAD_OUTER_DEPTH="$depth" \
            "${link[@]}" "$here/adapter.c" "${objects[@]}" -lm -o "$out/$stem.so"
        printf 'plugin=%s.so outer_cap=%s total_cap=2 counters=%s\n' "$stem" "$depth" "$kind" >> "$log"
    done
done
