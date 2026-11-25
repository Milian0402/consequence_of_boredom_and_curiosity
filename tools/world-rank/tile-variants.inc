register_tile_variants() {
    local arch=$1 phase=$2
    case "$arch/$phase" in
        x86_64/tile-screen)
            "$cc" -std=c11 -O3 "${target[@]}" -DCOB_ENABLE_BLOCKED_LEAF=1 -DCOB_ENABLE_X86_EXACT_EDGE=1 \
                -I"$repo/include" -I"$repo/src" "$dir/edge/check_edge.c" -lm -o "$build/check-edge"
            "$build/check-edge" | tee "$out/edge-validation.txt"
            tile_candidate exact-edge -DCOB_ENABLE_X86_EXACT_EDGE=1
            tile_candidate tile14 -DCOB_ENABLE_X86_TILE14=1
            tile_candidate direct-b96 -DCOB_ENABLE_DIRECT_B=1 -DCOB_DIRECT_B_KC=96
            tile_candidate direct-b256 -DCOB_ENABLE_DIRECT_B=1 -DCOB_DIRECT_B_KC=256
            ;;
        aarch64/tile-screen)
            tile_candidate n2-tail6 -DCOB_N2_TAIL_ROWS=6
            tile_candidate n2-tail8 -DCOB_N2_TAIL_ROWS=8
            tile_candidate n2-tail24 -DCOB_N2_TAIL_ROWS=4
            for rows in 6 8 4; do
                "$cc" -O3 -std=c11 -mcpu=native -DN2_NO_PLUGIN -DCOB_N2_TAIL_ROWS="$rows" \
                    "$dir/n2-small/n2.c" "$dir/n2-small/check.c" -lm -o "$build/check-n2-$rows"
                "$build/check-n2-$rows" | tee "$out/n2-$rows-validation.txt"
            done
            ;;
        x86_64/tile-confirm)
            tile_candidate exact-edge -DCOB_ENABLE_X86_EXACT_EDGE=1
            ;;
        aarch64/tile-confirm)
            tile_candidate n2-tail6 -DCOB_N2_TAIL_ROWS=6
            ;;
        *) echo "No configuration frozen for $arch/$phase" >&2; return 2 ;;
    esac
}
