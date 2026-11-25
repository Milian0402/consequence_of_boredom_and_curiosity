#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
vendor=$(cd "${1:?vendor directory required}" && pwd)
out=$(mkdir -p "${2:?output directory required}" && cd "$2" && pwd)
[[ $(uname -s) == Linux && $(uname -m) == x86_64 && -s $vendor/mkl.so ]]
cc=${CC:-cc}
common=(-O3 -march=x86-64 -std=c11 -Wall -Wextra -Werror -fPIC -shared -Wl,-z,defs)
names=(host-forward host-split8 host-prefix8)
"$cc" "${common[@]}" "$here/observer.c" -ldl -o "$out/prefix-observer.so"
printf 'method,ordinary_sha256,trace_sha256,provider_sha256\n' > "$out/prefix-builds.csv"
printf 'method,ordinary_sha256,alloc_imports,copy_imports,passed\n' > "$out/prefix-zero-copy.csv"
hash(){ sha256sum "$1" | cut -d' ' -f1; }
for n in 0 1 2; do
    name=${names[n]}
    "$cc" "${common[@]}" -DPREFIX_VARIANT="$n" "$here/adapter.c" -ldl -o "$out/$name.so"
    "$cc" "${common[@]}" -DPREFIX_VARIANT="$n" -DPREFIX_TRACE=1 "$here/adapter.c" -ldl -o "$out/$name.trace.so"
    nm -D --undefined-only "$out/$name.so" > "$out/$name.imports"
    if awk '{print $NF}' "$out/$name.imports" | cut -d@ -f1 | grep -Eq '^(malloc|calloc|realloc|reallocarray|free|aligned_alloc|memalign|posix_memalign|valloc|pvalloc|mmap|munmap|memcpy|memmove|memset|bcopy|__memcpy_chk|__memmove_chk|__memset_chk)$'; then
        echo "Unexpected explicit copy or allocation import in $name" >&2
        exit 1
    fi
    printf '%s,%s,0,0,1\n' "$name" "$(hash "$out/$name.so")" >> "$out/prefix-zero-copy.csv"
    nm --defined-only "$out/$name.so" > "$out/$name.nm"
    if grep -q 'rank_prefix_snapshot' "$out/$name.nm"; then exit 1; fi
    nm -D --defined-only "$out/$name.trace.so" | grep ' rank_prefix_snapshot$' > /dev/null
    printf '%s,%s,%s,%s\n' "$name" "$(hash "$out/$name.so")" "$(hash "$out/$name.trace.so")" "$(hash "$vendor/mkl.so")" >> "$out/prefix-builds.csv"
done
"$cc" -march=x86-64 -O2 -std=c11 -Wall -Wextra -Werror "$here/check.c" -ldl -lm -o "$out/prefix-check"
