# Single-threaded SGEMM on nine medium layouts

**0 of 2 processors met the predeclared speed criterion** for the standalone COB transposed-tile implementation. The first layout repeats a previously measured shape, five are fresh wide layouts and three exercise the candidate's fallbacks. Each processor supplied one independently timed Linux job. The measurements establish the rankings for those two hosts and these nine FP32 layouts. [Benchmark run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45686443679) · [Frozen panel](panel.csv)

| Processor | Incumbent time / candidate time, fresh geometric mean | Fresh wins vs incumbent | Fresh fastest complete one-shot | Meets criterion |
|---|---:|---:|---:|---|
| INTEL(R) XEON(R) PLATINUM 8573C | 0.7746× | 0/5 | 0/5 | Failed |
| Neoverse-N2 | 0.9803× | 0/5 | 0/5 | Failed |

The geometric mean gives each of the five fresh layouts equal weight. Enabling the route required at least 1.05× overall, four fresh paired-median wins, a gain in each seed block, at most 1.10× candidate/incumbent elapsed time on every fresh layout, qualification on all nine layouts, and bitwise parity on the controls. [Decision details](decision.csv)

## Fresh layouts against the fastest independent one-shot method

| Processor | M × N × K / padding | COB candidate, ms | Fastest independent one-shot method · ms | Rival time / candidate time, paired median | Pairs won by COB |
|---|---|---:|---|---:|---:|
| INTEL(R) XEON(R) PLATINUM 8573C | 73 × 1291 × 281 / +7 | 0.5368 | mkl · 0.3261 | 0.6091× | 0/6 |
| INTEL(R) XEON(R) PLATINUM 8573C | 149 × 1201 × 619 / +3 | 1.7990 | mkl-jit · 1.3187 | 0.7290× | 0/6 |
| INTEL(R) XEON(R) PLATINUM 8573C | 160 × 2304 × 384 / +0 | 2.0535 | mkl-jit · 1.5632 | 0.7603× | 0/6 |
| INTEL(R) XEON(R) PLATINUM 8573C | 49 × 2861 × 157 / +11 | 0.5511 | mkl · 0.2636 | 0.4792× | 0/6 |
| INTEL(R) XEON(R) PLATINUM 8573C | 241 × 1033 × 401 / +5 | 1.6614 | mkl · 1.1937 | 0.7195× | 0/6 |
| Neoverse-N2 | 73 × 1291 × 281 / +7 | 1.2188 | openblas · 1.0924 | 0.8959× | 0/6 |
| Neoverse-N2 | 149 × 1201 × 619 / +3 | 4.6048 | blasfeo · 4.4775 | 0.9696× | 0/6 |
| Neoverse-N2 | 160 × 2304 × 384 / +0 | 5.7851 | blasfeo · 5.5719 | 0.9636× | 0/6 |
| Neoverse-N2 | 49 × 2861 × 157 / +11 | 1.0662 | blasfeo · 1.0034 | 0.9409× | 0/6 |
| Neoverse-N2 | 241 × 1033 × 401 / +5 | 4.1963 | blasfeo · 3.9487 | 0.9397× | 0/6 |

The winner is the qualified implementation with the lowest median of six warm per-call times. The ratio takes the median of six rival/candidate ratios taken within the same seed and round. Thus a ratio above 1 favors the candidate, and the displayed absolute medians can give a slightly different quotient. A pair earns one win only when the candidate's time is strictly lower. The tables also identify the fastest qualified method when previously cached oneDNN plans are allowed. [All nine layouts and winners](summary.csv) · [Every method, paired ratios and first calls](method-summary.csv)

## Timing and eligibility

All implementations receive the same row-major FP32 operands with leading dimensions K + padding for A and N + padding for B and C. `medium-incumbent` is the released standalone COB implementation on its native routes. `medium-transposed` adds a guarded standalone native tile. External methods use direct library calls with their recorded adapters. `onednn-oneshot` pays for creating its operation on each call, and `onednn-cached-plan` reuses a primitive across calls of the same layout. The `independent_oneshot` and `whole_field_oneshot` rankings omit that cached variant. Their `*_all` or `whole_field` counterparts include it. [Method roles](methods.csv) · [Coverage](coverage.csv)

Each worker ran on one pinned logical CPU. Both random seeds, 68539041 and 93461257, supplied three interleaved warm rounds. The same batch size applied to every method on a layout, using the ceiling of 80 million divided by 2MNK, bounded to 32–512 complete calls. Each raw `seconds` value is the batch time per call. The 864 warm observations contain 27,648 complete multiplications. The two first-call times per method and layout came from the first unbatched multiplication after the library was loaded in a fresh worker for each random seed. [Raw warm observations, order and batch sizes](timings.csv) · [First calls in milliseconds](method-summary.csv)

For the first call on each fresh shape and seed, the candidate beat the incumbent in 0/10 comparisons on Xeon Platinum 8573C (0.7109×) and 10/10 on Neoverse N2 (1.0949×). These ratios are the geometric means of the ten individual incumbent/candidate first-call times per CPU. The decision used warm timings. [First-call measurements](accuracy.csv)

INTEL(R) XEON(R) PLATINUM 8573C qualified 81 of 81 measured method/layout pairs. Neoverse-N2 qualified 63 of 63 measured method/layout pairs. Qualification used two random inputs, a cancellation fixture and a scaled fixture, checking every output against FP64 with maximum absolute error 0.002. The worker also checked input preservation, finite values, output guards and thread count. The 576 accuracy records preserve all startup durations and signed error ranges. Independent route probes recorded the actual tile geometry and tested the compiled timed and traced builds, including seeded parity against the incumbent on the three controls. [Accuracy, guards and startup times](accuracy.csv) · [Standalone route probes](standalone-routes.csv)

| Processor | ISA | Pinned CPU | Candidate tile, rows × columns | Output values checked across all fixtures | Largest absolute error, field | Largest absolute error, candidate | Failed accuracy records |
|---|---|---:|---:|---:|---:|---:|---:|
| INTEL(R) XEON(R) PLATINUM 8573C | x86_64 | 0 | 32 × 12 | 57,974,328 | 3.6254193e-05 | 3.6254193e-05 | 0 |
| Neoverse-N2 | aarch64 | 0 | 8 × 12 | 45,091,144 | 3.6254193e-05 | 3.6254193e-05 | 0 |

To reproduce the medians, paired results and decision from these public records, run `python3 verify.py` in this directory. The verifier also checks the digests of the published files. GitHub artifact IDs, source digests, job links, native target flags and executable digests identify the original measurements. [Provenance](provenance.json) · [Published file digests](SHA256SUMS) · [Benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b0114de73cb91342cd5119c8b0892edaeb2e999f/tools/world-rank)
