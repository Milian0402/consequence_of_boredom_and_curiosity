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
    "$cc" "${flags[@]}" -fPIC -shared "$src/n2.c" -o "$build/n2-small.so"
    {
        printf 'flags=%s\n' "${flags[*]} -fPIC -shared"
        (cd "$src" && sha256sum n2.c check.c)
        sha256sum "$dir/n2-small-$phase.csv" "$build/n2-small.so" "$build/worker"
    } > "$dst/n2-small-build.txt"
    capture n2-small --vendor "libxsmm-wide=$vendor/libxsmm-wide.so" \
        --vendor "libxsmm-cached=$vendor/libxsmm-cached.so" \
        --candidate "cob-n2-small=direct:$build/n2-small.so"
}
