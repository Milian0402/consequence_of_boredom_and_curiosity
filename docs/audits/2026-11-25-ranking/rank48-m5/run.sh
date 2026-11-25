#!/bin/bash
set -euo pipefail
package=$(cd "$(dirname "$0")" && pwd -P)
repository=$(cd "${BENCH_REPO:?Set BENCH_REPO to the public benchmark checkout.}" && pwd -P)
destination=${1:?Pass a new directory for the raw results.}
[[ ! -e $destination ]] || { echo "Choose a fresh output directory." >&2; exit 2; }
temporary=$(mktemp -d "${TMPDIR:-/tmp}/rank48-m5.XXXXXXXX")
trap 'rm -rf -- "$temporary"' EXIT
cp "$package"/{accelerate.c,apple_thread_count.c,prepare.py,profile.py,profile-run.py,audit.py,measure.sh} "$temporary/"
BENCH_REPO="$repository" RANK_APPLE_RESULT=measurements bash "$temporary/measure.sh"
mkdir -p "$(dirname "$destination")"
cp -R "$temporary/measurements" "$destination"
