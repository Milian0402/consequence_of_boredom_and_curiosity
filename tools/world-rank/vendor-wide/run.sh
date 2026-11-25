#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:?result directory}" && cd "$1" && pwd)
repo=$(git -C "$dir" rev-parse --show-toplevel)
primary="${RANK_STUDY_ROOT:?}/stream-plugins"
build="$RANK_STUDY_ROOT/vendor-wide-secondary"
worker=${RANK_STUDY_WORKER:?}
cpu=${RANK_STUDY_CPU:?}
vendor=${RANK_STUDY_VENDOR_DIR:?}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
cp "$dir/"{panel.csv,protocol.md,registration.sha256} "$out/"
(cd "$dir" && sha256sum -c registration.sha256) > "$out/freeze-check.txt"
# Recover the verified release independently when an earlier primary stage failed.
if [[ ! -s $primary/release/src/gemm.c ]]; then
    mkdir -p "$build/restore"
    primary_source="$build/restore/release"
    python3 -B "$repo/tools/world-rank/release-replay/prepare-sources.py" "$repo" "$primary_source"
    python3 -B "$repo/tools/world-rank/release-public/apply-test-backport.py" "$primary_source"
else primary_source=$primary/release; fi
mkdir -p "$build"
bash "$dir/build.sh" "$primary_source" "$vendor" "$build" "$out" > "$out/provider-build.log" 2>&1 || {
    tail -60 "$out/provider-build.log"; exit 1; }
status=0
taskset -c "$cpu" python3 -B "$dir/routes.py" "$build" "$out" "$dir/panel.csv" || status=1
printf 'implementation,role,provider,abi,artifact_sha256\n' > "$out/methods.csv"
entries=()
add(){
    local name=$1 role=$2 provider=$3 abi=$4 file=$5
    if [[ ! -s $file ]]; then echo "Missing required participant $name" >&2; status=1; return; fi
    printf '%s,%s,%s,%s,%s\n' "$name" "$role" "$provider" "$abi" "$(sha256sum "$file" | cut -d' ' -f1)" >> "$out/methods.csv"
    case $role in
        oracle) ;;
        direct) entries+=(--vendor "$name=$file");;
        standalone|assisted) entries+=(--candidate "$name=$abi:$file");;
        *) return 2;;
    esac
}
add openblas oracle openblas direct "$vendor/openblas.so"
for name in mkl mkl-jit aocl blasfeo; do add "$name" direct "$name" direct "$vendor/$name.so"; done
for mode in oneshot cached-plan; do
    name=onednn-$mode
    if [[ -f $vendor/onednn/qualified ]]; then add "$name" direct onednn direct "$vendor/onednn/onednn-f32-$mode.so"
    else status=1; fi
done
for name in stream-incumbent stream-native; do add "$name" standalone cob linked-public "$primary/$name.worker"; done
for provider in openblas mkl; do
    add matched-$provider direct "$provider" direct "$build/matched-$provider.so"
    for mode in full split; do
        name=vendor-wide-$provider-$mode
        add "$name" assisted "$provider" direct "$build/$name.so"
    done
done
export RANK_MEDIUM_BATCH=1 RANK_VERIFY_INPUTS=1
taskset -c "$cpu" python3 -B "$dir/../study.py" "$worker" "$dir/panel.csv" "$out" "$vendor/openblas.so" \
    --phase stream-b --rounds 3 --stress-all "${entries[@]}" || status=1
python3 -B "$dir/score.py" "$out" "$dir/panel.csv" || status=1
exit "$status"
