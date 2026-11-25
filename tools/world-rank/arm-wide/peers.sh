#!/usr/bin/env bash
set -euo pipefail
shared=$(cd "$(dirname "$0")/.." && pwd)
vendors=$(mkdir -p "$1" && cd "$1" && pwd)
out=$(mkdir -p "$2" && cd "$2" && pwd)
export RANK_FRONTIER_LIBXSMM=1 LIBXSMM_BUILD_JOBS=3 BLASFEO_BUILD_JOBS=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash "$shared/vendors/build.sh" "$vendors"
bash "$shared/blasfeo/build.sh" "$vendors"
bash "$shared/libxsmm/build.sh" "$vendors"
bash "$shared/openblas/build.sh" "$vendors"
cp "$vendors"/{blasfeo-build.txt,armpl-build.txt,libxsmm-build.txt,openblas-build.txt} "$out/"
# This artifact was produced by this repo's pinned sequential strict-f32 build.
id=15050168540
digest=320291dc07053789ccc0ef0c91a5dd2df6d5558021aac83ebb8df2aaccf7f227
tmp=$(mktemp -d)
restore(){
    gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$id/zip" > "$tmp/hosted.zip" || return 1
    printf '%s  %s\n' "$digest" "$tmp/hosted.zip" | sha256sum --check --status || return 1
    unzip -qq "$tmp/hosted.zip" -d "$tmp/original" || return 1
    (cd "$tmp/original" && sha256sum --check runtime.tgz.sha256) || return 1
    mkdir -p "$vendors/onednn" || return 1
    tar xzf "$tmp/original/runtime.tgz" -C "$vendors/onednn" || return 1
    bash "$shared/onednn/qualify.sh" "$vendors/onednn" "$vendors/onednn/prefix/include" || return 1
}
if restore; then
    printf '%s  upstream-actions-archive-%s.zip\n' "$digest" "$id" > "$out/onednn-source.txt"
else
    # Keep the failed extraction separate from a clean build.
    if [[ -d "$vendors/onednn" ]]; then mv "$vendors/onednn" "$tmp/incomplete"; fi
    bash "$shared/onednn/build.sh" "$vendors/onednn"
    cp "$vendors/onednn/build-fingerprint.txt" "$out/onednn-source.txt"
fi
cp "$vendors/onednn"/qualified-*.log "$out/"
sha256sum "$vendors"/{openblas,armpl,blasfeo,libxsmm-dispatch,libxsmm-retained}.so "$vendors"/onednn/onednn-f32-*.so > "$out/peers.sha256"
