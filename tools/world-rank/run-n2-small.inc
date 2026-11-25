# shellcheck shell=bash disable=SC2154
# Sourced by run-kernels.sh, which supplies dir, out, build, phase, vendor
# and cc. The capture helper also includes the previous
# COB build, pinned OpenBLAS and the machine's native vendor BLAS.
run_n2_small() {
    [[ $(uname -m) == aarch64 ]] || return 0
    local src="$dir/n2-small"
    local dst="$out/n2-small"
    local flags=(-std=c11 -O3 -mcpu=native -Wall -Wextra -Werror)
    mkdir -p "$dst"
    "$cc" "${flags[@]}" -DN2_NO_PLUGIN "$src/n2.c" "$src/check.c" -lm -o "$build/test-n2-small"
    "$build/test-n2-small" | tee "$dst/native-validation.txt"
    local candidate="$build/n2-small.so"
    if [[ "$phase" == heldout ]]; then
        cob_build n2-integrated -DCOB_ENABLE_N2_SMALL=1
        candidate="$build/n2-integrated.so"
        "$cc" "${flags[@]}" -DN2_TEST_PUBLIC=1 -Iinclude "$src/check.c" "$candidate" -lm -o "$build/test-n2-public"
        "$build/test-n2-public" | tee "$dst/public-validation.txt"
    else
        "$cc" "${flags[@]}" -fPIC -shared "$src/n2.c" -o "$candidate"
    fi
    {
        if [[ "$phase" == heldout ]]; then
            printf 'entry=cob_sgemm_rowmajor\nflags=%s\n' "${common[*]} -DCOB_ENABLE_N2_SMALL=1"
        else
            printf 'entry=standalone\nflags=%s\n' "${flags[*]} -fPIC -shared"
        fi
        (cd "$src" && sha256sum n2.c check.c)
        sha256sum src/cob_n2_small.inc "$dir/n2-small-$phase.csv" "$candidate" "$build/worker"
    } > "$dst/n2-small-build.txt"
    capture n2-small --vendor "libxsmm-wide=$vendor/libxsmm-wide.so" \
        --vendor "libxsmm-cached=$vendor/libxsmm-cached.so" \
        --candidate "cob-n2-small=direct:$candidate"
}
