# Streaming B panels for single-threaded FP32 matrix multiplication on Intel(R) Xeon(R) 6975P-C

The ordinary COB entry with the streamed-B kernel passed its registered five-layout test against the matched COB incumbent. Its geometric mean paired speedup was **1.120×**, with faster medians on **5/5** fresh layouts. Against the fastest qualified independent one-shot interface on each layout, its geometric mean was **0.935×** and it won **2/5** fresh medians (12/30 paired timings). Including retained oneDNN plans, the corresponding result was 0.932× and 2/5.

The table lists the two previously measured layouts, the five fresh layouts in registration order, and the outside-width control. Times are median warm-call microseconds. Each speedup is the median of six matching peer-to-candidate time ratios. Values above 1 favor the candidate. The final column allows a retained oneDNN plan.

| M × N × K, padding | Incumbent µs | Candidate µs | vs incumbent (wins) | Fastest independent one-shot, µs | vs one-shot (wins) | Fastest including cached, µs |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 49 × 2861 × 157, +11 | 318.4 | 265.1 | 1.197× (6/6) | mkl-jit, 231.3 | 0.874× (0/6) | mkl-jit, 231.3 |
| 112 × 1744 × 357, +0 | 775.6 | 761.9 | 1.019× (6/6) | mkl-jit, 673.1 | 0.884× (0/6) | mkl-jit, 673.1 |
| 37 × 2387 × 119, +5 | 158.5 | 119.1 | 1.338× (6/6) | mkl, 106.7 | 0.899× (0/6) | mkl, 106.7 |
| 61 × 3451 × 191, +13 | 548.2 | 472.0 | 1.165× (6/6) | mkl-jit, 396.6 | 0.844× (0/6) | mkl-jit, 396.6 |
| 89 × 1933 × 257, +0 | 514.0 | 497.8 | 1.029× (6/6) | mkl-jit, 439.0 | 0.882× (0/6) | mkl-jit, 439.0 |
| 108 × 1728 × 320, +4 | 669.7 | 648.0 | 1.041× (6/6) | mkl-jit, 662.5 | 1.027× (6/6) | onednn-cached-plan, 654.2 |
| 137 × 2179 × 347, +7 | 1174.9 | 1115.5 | 1.055× (6/6) | openblas, 1156.3 | 1.037× (6/6) | openblas, 1156.3 |
| 39 × 959 × 193, +2 | 104.9 | 104.8 | 1.005× (4/6) | mkl-jit, 70.4 | 0.673× (0/6) | mkl-jit, 70.4 |

## Qualification and applicability

The primary binary used AVX512 on one pinned hardware thread. The two seed-block geometric means were 1.120× and 1.119×, and its slowest relative fresh-layout result was 1.029×. The preregistered gate required a five-layout geometric mean of at least 1.03×, gains in both seed blocks, at least four positive layout medians, and a floor of 0.98× on every fresh layout. Independent native reproduction was also required before promotion.

Each admitted method had two random inputs, a cancellation fixture and a scaled fixture, with every output compared to FP64, a maximum allowed absolute error of 0.002, and checks for nonfinite values, input mutation, padding and process thread count. Across the field, 416 fixtures checked 69,104,464 output elements. The maximum absolute error was 2.39592641051e-05 for the candidate and 2.39592641051e-05 for the incumbent. The archive records 416 first calls separately from 624 warm time batches, with 32 complete multiplications per batch. Each matched comparison uses the same seed, round and batch size.

The uninstrumented public entry and its trace companion each checked all cells on the integer route fixture. The candidate selected streaming B on the seven in-scope layouts, with one packing pass per column panel. It bypassed on the eighth. The traces distinguish the incumbent medium panel from its blocked fallback. The two forced-ISA pairs are sensitivity checks, and the direct comparators include OpenBLAS, AOCL-BLAS, BLASFEO, ordinary oneMKL, its sequential direct-call/JIT-capable interface, and oneDNN. All eight shapes exceed the oneMKL direct-call header's 262,144-element M×N×K JIT selection limit, so this run measures that interface's larger-shape dispatch. oneDNN's recorded compute implementation(s) were brg_matmul:avx512_core.

| Matched build | Fresh geometric mean | Fresh wins | Minimum | Gate |
| --- | ---: | ---: | ---: | --- |
| stream-native | 1.120× | 5/5 | 1.029× | Passed |
| stream-avx2 | 1.131× | 5/5 | 1.063× | Passed |
| stream-avx512 | 1.115× | 5/5 | 1.025× | Passed |

Every tested method met the per-layout qualification threshold on all eight layouts.

## Reproducing the evidence

Run `python3 verify.py raw` from this directory for an independent reconstruction of the route, numerical and paired measurements. The raw directory includes the input panel, the original qualification and performance CSVs, source and executable digests, per-method admission, and the matching first-call results. The source manifest covers the 30 release files and the two touched paths in the experimental build. The GitHub artifact digest in provenance.json identifies the unedited native upload; its public subset replaces absolute build paths with file names.

The baseline is `b218beff1eaf8fd606d0934405cdc1e28ecdfe15`. The benchmark is `b2cdc6a2985726cb1658dc7cc730b5a1c4b99680`. [Native run and full log](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697) · [Frozen source, patch and protocol](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b2cdc6a2985726cb1658dc7cc730b5a1c4b99680/tools/world-rank/stream)
