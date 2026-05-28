---
title: COB SGEMM Benchmark Dashboard
sdk: static
license: mit
pinned: false
---

# COB SGEMM Benchmark Dashboard

Static explorer for the Apple Silicon single-thread SGEMM benchmark data from
`consequence_of_boredom_and_curiosity`.

The app now has a compact best-result card, bar charts for headline throughput
and fresh current-vs-head wins, stronger filtering on the large tables, and a
short methodology boundary so the claim is readable without digging through raw
CSV first.

The interesting bits are the fresh paired A/B runs, the packed-AB rows near
1.9 TFLOP/s median, and the searchable optimization timeline. It also keeps the
open Accelerate gaps and rejected probes visible instead of only showing wins.

The dashboard is intentionally read-only and runs entirely in the browser. It
does not run AMX or SME kernels on Hugging Face hardware; it filters and
summarizes recorded local measurements.

Fresh May 27 additions include a headline throughput sweep, parsed paired A/B
runs, correctness validation, and an accepted/mixed/rejected research-attempt
table.

The Space also exposes the structured optimization timeline so the full route
search history can be filtered by accepted, rejected, mixed, measurement,
tooling, comparison, or blocked entries.
