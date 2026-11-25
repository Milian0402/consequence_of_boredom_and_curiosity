#!/bin/sh
# Run from anywhere after all native timed work has stopped.
set -eu
if [ "$#" -ne 6 ]; then
    echo "usage: sh test.sh BASE CANDIDATE TRACE TRACE_LEAF_C SHIPPED_DRIVER_O INCLUDE_DIR" >&2
    exit 2
fi
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cc -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off "$here/compact-check.c" -o "$here/compact-check"
cc -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off "$here/fallback-check.c" -o "$here/fallback-check"
python3 "$here/make-fault.py" "$4" "$5" "$here/abi.c" "$6" "$here/fault-trace.dylib"
"$here/compact-check" "$1" "$2" "$3" compact_observed > "$here/checks.csv"
"$here/fallback-check" "$1" "$here/fault-trace.dylib" > "$here/fallbacks.csv"
python3 "$here/summarize.py" "$here/checks.csv" "$here/fallbacks.csv" "$here/summary.json"
