# Streaming B panels for single-threaded FP32 matrix multiplication on AMD EPYC 9V45 96-Core Processor

The ordinary COB entry with the streamed-B kernel passed its registered five-layout test against the matched COB incumbent. Its geometric mean paired speedup was **1.120×**, with faster medians on **5/5** fresh layouts. Against the fastest qualified independent one-shot interface on each layout, its geometric mean was **0.958×** and it won **0/5** fresh medians (3/30 paired timings). Including retained oneDNN plans, the corresponding result was 0.949× and 0/5.

The table lists the two previously measured layouts, the five fresh layouts in registration order, and the outside-width control. Times are median warm-call microseconds. Each speedup is the median of six matching peer-to-candidate time ratios. Values above 1 favor the candidate. The final column allows a retained oneDNN plan.

| M × N × K, padding | Incumbent µs | Candidate µs | vs incumbent (wins) | Fastest independent one-shot, µs | vs one-shot (wins) | Fastest including cached, µs |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 49 × 2861 × 157, +11 | 253.9 | 204.8 | 1.248× (6/6) | aocl, 186.8 | 0.911× (0/6) | aocl, 186.8 |
| 112 × 1744 × 357, +0 | 616.1 | 596.6 | 1.034× (6/6) | aocl, 544.2 | 0.913× (0/6) | aocl, 544.2 |
| 37 × 2387 × 119, +5 | 134.0 | 102.3 | 1.314× (6/6) | aocl, 96.6 | 0.939× (1/6) | aocl, 96.6 |
| 61 × 3451 × 191, +13 | 436.3 | 365.6 | 1.192× (6/6) | aocl, 344.9 | 0.944× (1/6) | aocl, 344.9 |
| 89 × 1933 × 257, +0 | 401.8 | 386.3 | 1.039× (6/6) | aocl, 378.1 | 0.980× (1/6) | onednn-cached-plan, 361.0 |
| 108 × 1728 × 320, +4 | 512.0 | 495.5 | 1.035× (6/6) | aocl, 470.8 | 0.951× (0/6) | aocl, 470.8 |
| 137 × 2179 × 347, +7 | 920.7 | 876.5 | 1.047× (6/6) | aocl, 857.6 | 0.976× (0/6) | aocl, 857.6 |
| 39 × 959 × 193, +2 | 89.2 | 90.5 | 0.992× (2/6) | aocl, 74.1 | 0.810× (0/6) | onednn-cached-plan, 68.1 |

## Qualification and applicability

The primary binary used AVX512 on one pinned hardware thread. The two seed-block geometric means were 1.114× and 1.123×, and its slowest relative fresh-layout result was 1.035×. The preregistered gate required a five-layout geometric mean of at least 1.03×, gains in both seed blocks, at least four positive layout medians, and a floor of 0.98× on every fresh layout. Independent native reproduction was also required before promotion.

Each admitted method had two random inputs, a cancellation fixture and a scaled fixture, with every output compared to FP64, a maximum allowed absolute error of 0.002, and checks for nonfinite values, input mutation, padding and process thread count. Across the field, 416 fixtures checked 69,104,464 output elements. The maximum absolute error was 2.39592641051e-05 for the candidate and 2.39592641051e-05 for the incumbent. The archive records 416 first calls separately from 624 warm time batches, with 32 complete multiplications per batch. Each matched comparison uses the same seed, round and batch size.

The uninstrumented public entry and its trace companion each checked all cells on the integer route fixture. The candidate selected streaming B on the seven in-scope layouts, with one packing pass per column panel. It bypassed on the eighth. The traces distinguish the incumbent medium panel from its blocked fallback. The two forced-ISA pairs are sensitivity checks, and the direct comparators include OpenBLAS, AOCL-BLAS, BLASFEO, ordinary oneMKL, its sequential direct-call/JIT-capable interface, and oneDNN. All eight shapes exceed the oneMKL direct-call header's 262,144-element M×N×K JIT selection limit, so this run measures that interface's larger-shape dispatch. oneDNN's recorded compute implementation(s) were brg_matmul:avx512_core.

| Matched build | Fresh geometric mean | Fresh wins | Minimum | Gate |
| --- | ---: | ---: | ---: | --- |
| stream-native | 1.120× | 5/5 | 1.035× | Passed |
| stream-avx2 | 1.087× | 5/5 | 1.032× | Passed |
| stream-avx512 | 1.121× | 5/5 | 1.036× | Passed |

Every tested method met the per-layout qualification threshold on all eight layouts.

## Reproducing the evidence

Run `python3 verify.py raw` from this directory for an independent reconstruction of the route, numerical and paired measurements. The raw directory includes the input panel, the original qualification and performance CSVs, source and executable digests, per-method admission, and the matching first-call results. The source manifest covers the 30 release files and the two touched paths in the experimental build. The GitHub artifact digest in provenance.json identifies the unedited native upload; its public subset replaces absolute build paths with file names.

The baseline is `b218beff1eaf8fd606d0934405cdc1e28ecdfe15`. The benchmark is `b2cdc6a2985726cb1658dc7cc730b5a1c4b99680`. [Native run and full log](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697) · [Frozen source, patch and protocol](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b2cdc6a2985726cb1658dc7cc730b5a1c4b99680/tools/world-rank/stream)
