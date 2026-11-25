#!/usr/bin/env bash
set -euo pipefail
rank_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "${2:-.}" && pwd)
build=$(mkdir -p "${1:?usage: build.sh OUTPUT [REPO]}" && cd "$1" && pwd)
base=828109bf7c1c2c430f5f292d099eb669e46a96fa
cc=${CC:-cc}
flags=(-O3 -std=c11 -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1
       "-I$repo/include" "-I$repo/src" "-I$rank_dir")
case "$(uname -m)" in
    x86_64) flags+=(-march=native) ;;
    aarch64) flags+=(-mcpu=native -DCOB_USE_NEON=1) ;;
    arm64) flags+=(-mcpu=native -DCOB_USE_NEON=1 -DCOB_DISABLE_APPLE_AMX -DCOB_DISABLE_APPLE_SME -D_DARWIN_C_SOURCE) ;;
    *) echo "Unsupported CPU architecture" >&2; exit 1 ;;
esac
git -C "$repo" show "$base:src/gemm.c" > "$build/base.c"
python3 "$rank_dir/hybrid.py" "$build/base.c" > "$build/hybrid.c"
renames=()
for fn in sgemm_rowmajor sgemm_ref_rowmajor sgemm_pack_a sgemm_free_packed_a sgemm_pack_b sgemm_free_packed_b sgemm_rowmajor_packed_b sgemm_rowmajor_packed_ab; do
    renames+=("-Dcob_$fn=hybrid_$fn")
done
"$cc" "${flags[@]}" -c "$build/base.c" -o "$build/native.o"
"$cc" "${flags[@]}" "${renames[@]}" -c "$build/hybrid.c" -o "$build/hybrid.o"
"$cc" "${flags[@]}" -Wall -Wextra -Werror "$rank_dir/worker.c" "$build/native.o" "$build/hybrid.o" -lm -ldl -o "$build/worker"

extra=()
while IFS= read -r definition; do
    case "$definition" in
        ''|'#'*) ;;
        COB_ENABLE_BLOCKED_LEAF=[01]|COB_ENABLE_DIRECT_SMALL_GEMM=[01]|COB_ENABLE_SHORT_AXES=[01]|COB_ENABLE_FUSED_STRASSEN=[01]|COB_ENABLE_SMALL_X86=[01]|COB_ENABLE_FEW_ROW_PANEL=[01]) extra+=("-D$definition") ;;
        *) echo "Invalid challenger flag $definition" >&2; exit 1 ;;
    esac
done < "$rank_dir/challenger-flags.txt"
if [[ $(cat "$rank_dir/phase.txt") == discovery ]]; then
    "$cc" "${flags[@]}" "${extra[@]}" -UCOB_ENABLE_FUSED_STRASSEN -DCOB_ENABLE_FUSED_STRASSEN=0 -fPIC -shared "$repo/src/gemm.c" "$rank_dir/cob_adapter.c" -o "$build/unfused.so"
fi
"$cc" "${flags[@]}" "${extra[@]}" -fPIC -shared "$repo/src/gemm.c" "$rank_dir/cob_adapter.c" -o "$build/challenger.so"
if [[ $(cat "$rank_dir/phase.txt") =~ ^(replicate|short)$ ]]; then
    previous=3cd1efc303b0de0318b75f6b04487951ff06a739
    mkdir -p "$build/previous"
    git -C "$repo" archive "$previous" src include | tar -x -C "$build/previous"
    prev_flags=(-DCOB_ENABLE_BLOCKED_LEAF=1 -DCOB_ENABLE_SHORT_AXES=1 -DCOB_ENABLE_FUSED_STRASSEN=1)
    "$cc" "${flags[@]}" "${prev_flags[@]}" -fPIC -shared "$build/previous/src/gemm.c" \
          "$rank_dir/cob_adapter.c" -o "$build/previous.so"
fi
printf 'baseline_revision=%s\nchallenger_revision=%s\nflags=%s\nchallenger_flags=%s\n' "$base" "$(git -C "$repo" rev-parse HEAD)" "${flags[*]}" "${extra[*]}" > "$build/build.txt"
