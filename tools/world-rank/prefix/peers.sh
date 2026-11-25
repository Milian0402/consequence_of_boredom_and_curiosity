#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
vendor=$(mkdir -p "$1" && cd "$1" && pwd)
out=$(mkdir -p "$2" && cd "$2" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export RANK_FRONTIER_LIBXSMM=1
printf 'family,status\n' > "$out/peer-build-status.csv"
for name in vendors openblas aocl blasfeo libxsmm; do
    if bash "$here/$name/build.sh" "$vendor" > "$out/$name-build.log" 2>&1; then
        printf '%s,ready\n' "$name" >> "$out/peer-build-status.csv"
    else
        printf '%s,failed\n' "$name" >> "$out/peer-build-status.csv"
        tail -12 "$out/$name-build.log"
        if [[ $name == vendors ]]; then exit 1; fi
    fi
done
archive_id=15049984729
digest=df9673efd4ab4cf3b382e9392a0c17bdb324a018ef13aaba14df242ff8a0ae8d
dest=$(mktemp -d)
get_runtime() (
    set -euo pipefail
    gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$archive_id" > "$out/onednn-upstream-receipt.json" || exit 1
    jq -e --arg digest "sha256:$digest" '.digest==$digest and .workflow_run.id==45672997916 and .workflow_run.head_sha=="29f5817ed45a728bc012f332fa378dc95c6e7ef5"' "$out/onednn-upstream-receipt.json" >/dev/null || exit 1
    gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$archive_id/zip" > "$dest/artifact.zip" || exit 1
    printf '%s  %s\n' "$digest" "$dest/artifact.zip" | sha256sum --check || exit 1
    unzip -q "$dest/artifact.zip" -d "$dest" || exit 1
    cd "$dest" || exit 1
    sha256sum --check runtime.tgz.sha256 || exit 1
    mkdir -p "$vendor/onednn" || exit 1
    tar xzf runtime.tgz -C "$vendor/onednn" || exit 1
    bash "$here/onednn/qualify.sh" "$vendor/onednn" "$vendor/onednn/prefix/include"
)
if get_runtime > "$out/onednn-restore.log" 2>&1; then
    touch "$vendor/onednn/qualified"
    printf 'onednn,restored_and_qualified\n' >> "$out/peer-build-status.csv"
elif bash "$here/onednn/build.sh" "$vendor/onednn" > "$out/onednn-rebuild.log" 2>&1; then
    touch "$vendor/onednn/qualified"
    printf 'onednn,rebuilt_and_qualified\n' >> "$out/peer-build-status.csv"
else
    printf 'onednn,failed\n' >> "$out/peer-build-status.csv"
fi
