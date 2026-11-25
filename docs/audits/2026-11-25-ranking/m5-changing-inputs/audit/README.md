# Independent M5 rotation audit

Both source-bound executions covered nine compact FP32 layouts, 11 labels, two fresh operand seeds and two memory regimes. A and B designate the same parent function; D is the shipped opt-in. Each execution contributed 4,356 batches, 318,956 complete products, and 773,295,534 output entries independently checked against FP64. The maximum absolute error was 0.000031291 on either execution.

| Eight prior layouts, wall time | Prefaulted | Page-locked |
| --- | ---: | ---: |
| D vs fastest independent, hot | 1.0133×, 4/8 | 1.0256×, 5/8 |
| D vs fastest independent, rotating | 0.9358×, 2/8 | 0.9501×, 3/8 |
| D vs same-configured parent, rotating | 1.0445× | 1.0510× |
| Rotating seed 1, strongest independent selected within seed | 0.9322× | 0.9446× |
| Rotating seed 2, strongest independent selected within seed | 0.9309× | 0.9315× |
| Rotating parent alias range | 0.9461–1.0586× | 0.9337–1.0521× |
| Rotating alias ratios beyond ±3% | 5/8 | 3/8 |

Ratios above 1 favor D. Both executions fell short of the rotation rule requiring a field advantage on each seed and eight same-function controls within 3%. The separate 24 × 1031 × 383 boundary case reached 1.0395× in the first rotating run and 0.9300× in the page-locked replay. For the eight scored layouts the rotating caller-CPU clock gave corresponding field speedups of 0.9312× and 0.9459×. All batches observed a single task thread at both endpoints. Scheduling remained under macOS control.

The prefaulted run had two arena pages missing at one intermediate observation (8,246 pages in that pool). All 18 page-locked arenas were fully resident before, after and at the recorded minimum per pool. Across all 36 locked epochs the process recorded zero major and 9,337 minor faults. This accounts for residency and process-wide faults; cache hits, DRAM accesses and physical CPU placement were unmeasured.

## Reproduction

Run `python3 -B check-design.py` for the synthetic schedule test. Run `python3 -B score.py RAW --output OUTPUT` on each raw directory. The scorer requires the recorded CSV sequence itself to match both the alternating order of memory modes and all within-epoch method positions. Its summary includes seed-specific field leaders, both pooled-leader and per-seed-leader geometric means, per-layout nulls and the registered rule.

Run `python3 -B bind.py FROZEN RAW primary` or `python3 -B bind.py FROZEN RAW wired` with the original freeze manifest, recipe and source files. When using the original measured local workspace, add `--native-binary-root FROZEN` to hash both worker executables, the linked LIBXSMM archive and each ordinary-call library at their original locations. The included source-binding JSON records this check for each execution. The frozen worker digest supports comparison with a retained original executable even when the package only redistributes its source.

The independent output is stored in `primary/` and `wired/`. Each directory contains per-shape, per-peer and rotation-effect CSVs, geometric-mean tables and the source and raw-data bindings.
