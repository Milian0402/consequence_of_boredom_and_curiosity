#!/bin/sh
set -eu
case "$(uname -s):$(uname -m)" in
    Linux:aarch64) neon=-DCOB_USE_NEON=1 ;;
    Linux:x86_64) neon= ;;
    *) echo "Use a native Linux Arm64 or x86-64 build with serial OpenBLAS." >&2; exit 2 ;;
esac

checkout=${1:-.}
base=b218beff1eaf8fd606d0934405cdc1e28ecdfe15
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
compiler=${CC:-cc}
stage=$(mktemp -d "${TMPDIR:-/tmp}/cob-signed.XXXXXXXX")
trap 'rm -rf "$stage"' EXIT
trap 'exit 1' HUP INT TERM
mkdir "$stage/source"
git -C "$checkout" archive -o "$stage/base.tar" "$base" src include tests
tar -xf "$stage/base.tar" -C "$stage/source"
(cd "$stage/source" && patch -p1 < "$here/signed-leaves.patch")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set -- -O2 -std=c11 -march=native -DCOB_ENABLE_PORTABLE_STRASSEN=1 \
    -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_ALIGNED_WINOGRAD=1 \
    -DCOB_ENABLE_OPENBLAS_LEAF=1 -I"$stage" -I"$stage/source/include"
if [ -n "$neon" ]; then set -- "$@" "$neon"; fi
for sign in 0 1; do
    knob="-DCOB_ENABLE_SIGNED_WINOGRAD_LEAVES=$sign"
    "$compiler" "$@" "$knob" "$here/validate.c" -lopenblas -lm -o "$stage/check"
    "$stage/check"
    "$compiler" "$@" "$knob" "$stage/source/tests/test_openblas_winograd.c" -lopenblas -lm -o "$stage/edges"
    "$stage/edges"
    "$compiler" "$@" "$knob" -c "$stage/source/src/gemm.c" -o "$stage/gemm.o"
    "$compiler" "$@" "$knob" "$here/public.c" "$stage/gemm.o" -lopenblas -lm -o "$stage/public"
    "$stage/public"
done
