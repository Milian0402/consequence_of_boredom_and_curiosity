#!/usr/bin/env bash
# Reuse the previously authenticated binaries, then qualify on the actual runner.
set -euo pipefail
vendor=$1; out=$2; repo=$(git rev-parse --show-toplevel)
case $(uname -m) in
 aarch64) id=15050168540; digest=320291dc07053789ccc0ef0c91a5dd2df6d5558021aac83ebb8df2aaccf7f227;;
 x86_64) id=15049984729; digest=df9673efd4ab4cf3b382e9392a0c17bdb324a018ef13aaba14df242ff8a0ae8d;;
 *) exit 1;;
esac
s=$(mktemp -d "$RUNNER_TEMP/ragged-dnn.XXXXXX")
mkdir -p "$s/stage" "$vendor/onednn" "$out/onednn"
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$id" > "$out/onednn/github.json"
jq -e --arg digest "sha256:$digest" --arg name "onednn-$(uname -m)-45672997916-1" \
 '.name==$name and .digest==$digest and .workflow_run.id==45672997916 and .workflow_run.head_sha=="29f5817ed45a728bc012f332fa378dc95c6e7ef5"' "$out/onednn/github.json"
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$id/zip" > "$s/hosted.zip"
printf '%s  %s\n' "$digest" "$s/hosted.zip" | sha256sum --check
unzip -q "$s/hosted.zip" -d "$s/stage"
(cd "$s/stage" && sha256sum --check runtime.tgz.sha256 && tar xzf runtime.tgz -C "$vendor/onednn")
bash "$repo/tools/world-rank/onednn/qualify.sh" "$vendor/onednn" "$vendor/onednn/prefix/include"
cp "$vendor/onednn"/{qualified-*.log,host.txt} "$out/onednn/"
cp "$s/stage"/{runtime.tgz.sha256,build-fingerprint.txt} "$out/onednn/"
touch "$vendor/onednn/qualified"
