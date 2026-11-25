#!/usr/bin/env bash
set -euo pipefail
source=$(cd "$1" && pwd)
vendor=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$dir")
repo=$(git -C "$shared" rev-parse --show-toplevel)
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/stream-release.XXXXXXXX")
cc=${CC:-gcc}
frozen=b2cdc6a2985726cb1658dc7cc730b5a1c4b99680
release=$(cat "$dir/release-revision.txt")
base=b218beff1eaf8fd606d0934405cdc1e28ecdfe15
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
[[ $release =~ ^[a-f0-9]{40}$ && $(git -C "$source" rev-parse HEAD) == "$release" ]]
[[ $(git -C "$source" rev-parse HEAD^) == "$base" ]]
git -C "$source" diff --quiet HEAD --
git -C "$repo" diff --quiet "$frozen" HEAD -- tools/world-rank/stream tools/world-rank/worker.c \
    tools/world-rank/release-replay tools/world-rank/release-public tools/world-rank/medium/source.json
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
export RANK_VERIFY_INPUTS=1 RANK_MEDIUM_BATCH=1
mkdir -p "$build/research" "$build/frozen-results" "$out/research"
cp "$shared/stream/panel.csv" "$dir/protocol.md" "$dir/release-revision.txt" "$out/"
lscpu | sed -n -e '/^Architecture:/p' -e '/^Model name:/p' -e '/^CPU family:/p' -e '/^Model:/p' -e '/^Stepping:/p' -e '/^Flags:/p' > "$out/host.txt"
{ "$cc" --version; cmake --version; make --version; pkg-config --modversion openblas; } > "$out/toolchain.txt"
(
    cd "$source"
    git ls-files -z CMakeLists.txt Makefile include src tests | xargs -0 sha256sum
) > "$out/production-files.sha256"
python3 -B - "$out" "$release" "$base" "$frozen" "$(git -C "$repo" rev-parse HEAD)" "$(git -C "$source" rev-parse HEAD^{tree})" "$cpu" <<'PY'
import json,os,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1])
def digest(name): return hashlib.sha256((root/name).read_bytes()).hexdigest()
assert digest("panel.csv")=="466c2b51ee6df5fd737f3f4b1977c5b4ad75cdb7ae61bbaec7bc5919530f18fd"
data=dict(release=sys.argv[2],base=sys.argv[3],research=sys.argv[4],benchmark=sys.argv[5],release_tree=sys.argv[6],
          cpu=int(sys.argv[7]),panel_sha256=digest("panel.csv"),protocol_sha256=digest("protocol.md"),
          source_manifest_sha256=digest("production-files.sha256"),github_run=os.environ["GITHUB_RUN_ID"],
          github_attempt=os.environ["GITHUB_RUN_ATTEMPT"])
(root/"sources.json").write_text(json.dumps(data,indent=2)+"\n")
PY
bash "$shared/stream/build.sh" "$repo" "$build/research" "$build/frozen-results" > "$out/research/build.log" 2>&1 ||
    { tail -50 "$out/research/build.log"; exit 1; }
cp "$build/frozen-results"/{dispatch-flags.txt,native-features.txt} "$out/"
cp "$build/frozen-results"/{builds.csv,sources.json,*-fixtures.txt} "$out/research/"
taskset -c "$cpu" python3 -B "$shared/stream/routes.py" "$out/panel.csv" "$build/research" "$build/frozen-results"
cp "$build/frozen-results/routes.csv" "$out/research-routes.csv"
bash "$dir/build.sh" "$source" "$shared" "$build" "$out" "$cpu"
taskset -c "$cpu" python3 -B "$dir/routes.py" "$build" "$out"
"$cc" -O3 -std=c11 -fPIC -shared $(pkg-config --cflags openblas) "$shared/openblas.c" $(pkg-config --libs openblas) -o "$vendor/openblas.so"
entries=(--vendor "mkl=$vendor/mkl.so" --vendor "mkl-jit=$vendor/mkl-jit.so" --vendor "aocl=$vendor/aocl.so")
printf 'implementation,sha256\n' > "$out/timed-binaries.csv"
receipt(){ printf '%s,%s\n' "$1" "$(sha256sum "$2" | cut -d' ' -f1)" >> "$out/timed-binaries.csv"; }
for provider in openblas mkl mkl-jit aocl; do receipt "$provider" "$vendor/$provider.so"; done
for label in cmake-default cmake-att cmake-intel make-default make-att make-intel cmake-native-default cmake-native-on; do
    entries+=(--candidate "$label=linked-public:$build/$label.worker"); receipt "$label" "$build/$label.worker"
done
for pair in research-incumbent:stream-incumbent research-candidate:stream-native; do
    label=${pair%:*}; original=${pair#*:}
    entries+=(--candidate "$label=linked-public:$build/research/$original.worker"); receipt "$label" "$build/research/$original.worker"
done
cp "$vendor"/mkl{,-jit}-build.txt "$vendor/aocl-configuration.txt" "$out/"
taskset -c "$cpu" python3 -B "$shared/study.py" "$build/cmake-default.worker" "$out/panel.csv" "$out" "$vendor/openblas.so" \
    --phase stream-release --rounds 3 --stress-all "${entries[@]}"
python3 -B "$dir/verify.py" "$out" --release "$release" --benchmark "$(git -C "$repo" rev-parse HEAD)" > "$out/acceptance.json"
python3 -B - "$out/acceptance.json" <<'PYGATE'
import json,sys
from pathlib import Path
gates=json.loads(Path(sys.argv[1]).read_text())["gates"]
for name,passed in gates.items():
    print(f"{name} {passed}")
if not all(gates.values()):
    sys.exit("Registered timing gates evaluated. See acceptance.json for per-layout outcomes.")
PYGATE
