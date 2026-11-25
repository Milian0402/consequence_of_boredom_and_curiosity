# Supplemental oneMKL eight-column prefix measurement

The vendor-assisted prefix wrapper cleared the registered promotion gate on **0 of 1 Intel jobs**.

After both original assignments used AMD processors, one additional host was requested on the exact original commit, input seeds, panel and timing protocol. The choice was recorded before this last VM was assigned. The supplemental result is reported separately and retained regardless of manufacturer or outcome. Both original jobs verified ordinary forwarding on all eleven layouts. [Original replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139521811697) · [Original replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139521811716).

The candidate uses an AVX2/FMA kernel for the first eight columns and sends the remaining N−8 columns to sequential oneMKL on nine eligible Intel layouts. The matched forwarder calls oneMKL on all N columns. The split control calls oneMKL on eight columns, then on N−8. On the two cropped bypass layouts each wrapper forwards the full product. Every timing covers the complete single-threaded row-major FP32 overwrite operation.

A ratio above 1 favors the prefix wrapper. Each warm layout ratio is the median of six paired peer-to-wrapper times, three rounds per seed with 32 completed calls per round. Each of the six fresh layouts has equal weight in their geometric mean. Three historical anchors and two bypass controls are reported individually. Independent leaders have the lowest qualified warm median in their tier on that job and layout. The all-method ranking also admits standalone COB and the three assisted wrappers.

| Native job | Intel gate | vs direct oneMKL | vs forwarder | vs split | vs fastest qualified one-shot | vs fastest qualified incl. retained |
| --- | --- | --- | --- | --- | --- | --- |
| [Intel(R) Xeon(R) 6973P-C, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139527431524) | Failed | 0.9505× | 0.9496× | 0.9961× | 0.9449× | 0.9435× |

Promotion requires complete qualification of the assisted wrappers and direct oneMKL on all eleven layouts, including route and bitwise bypass checks. Both direct and matched forwarder comparisons must reach 1.03× in the six-shape geometric mean, reach 1.03× on the same five or more fresh shapes and stay at or above 0.97× on all nine active shapes. The split control measures the effect of replacing the first vendor call with the prefix kernel.

## Intel(R) Xeon(R) 6973P-C · replica 1

**Intel promotion failed.** 0/6 fresh layouts reached 1.03× against both direct oneMKL and the matched forwarder. The lowest ratios across all nine active shapes were 0.8559× against direct oneMKL and 0.8554× against the forwarder.

| Warm comparison | Six-layout geomean | Case wins | Round wins | Seed 1 | Seed 2 | Leave-one-layout-out range |
| --- | --- | --- | --- | --- | --- | --- |
| Direct oneMKL | 0.9505× | 0/6 | 3/36 | 0.9495× | 0.9521× | 0.9419×–0.9706× |
| Matched forwarder | 0.9496× | 0/6 | 1/36 | 0.9479× | 0.9501× | 0.9409×–0.9697× |
| Two-call split | 0.9961× | 0/6 | 8/36 | 0.9941× | 0.9993× | 0.9956×–0.9964× |
| Fastest qualified independent one-shot | 0.9449× | 0/6 | 1/36 | 0.9403× | 0.9466× | 0.9359×–0.9645× |
| Fastest qualified independent incl. retained | 0.9435× | 0/6 | 1/36 | 0.9427× | 0.9443× | 0.9342×–0.9628× |
| Fastest across all qualified methods | 0.9428× | 0/6 | 0/36 | 0.9423× | 0.9426× | 0.9334×–0.9619× |

On the fresh shapes the registered independent one-shot field was complete and the retained-inclusive field was complete. The prefix wrapper had the lowest warm median on 0/6 rankable fresh shapes.

### Warm whole-call measurements

Each control cell gives its median time in microseconds and its median paired ratio against the prefix wrapper. A “forward” candidate route covers one full-width oneMKL call.

| Layout ID | M × N × K | Candidate route | Prefix µs | Direct oneMKL | Matched forwarder | Two-call split |
| --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | prefix | 654.57 | 642.24 · 0.9773× | 657.16 · 0.9810× | 650.62 · 0.9929× |
| repeat-ragged | 137 × 2179 × 347 | prefix | 1171.39 | 1159.91 · 0.9910× | 1160.63 · 0.9914× | 1172.35 · 1.0012× |
| repeat-short | 43 × 2291 × 137 | prefix | 149.49 | 147.30 · 0.9860× | 147.58 · 0.9852× | 148.69 · 0.9938× |
| fresh-thin | 29 × 1229 × 103 | prefix | 38.70 | 34.89 · 0.9007× | 34.95 · 0.9007× | 38.40 · 0.9949× |
| fresh-odd | 44 × 1897 × 151 | prefix | 130.57 | 127.89 · 0.9806× | 127.60 · 0.9764× | 129.42 · 0.9967× |
| fresh-exact | 68 × 2176 × 244 | prefix | 415.55 | 354.67 · 0.8559× | 355.30 · 0.8554× | 413.72 · 0.9950× |
| fresh-long | 83 × 3067 × 271 | prefix | 657.84 | 650.82 · 0.9876× | 649.76 · 0.9863× | 655.57 · 0.9944× |
| fresh-high-depth | 112 × 3581 × 337 | prefix | 1325.89 | 1314.62 · 0.9929× | 1315.04 · 0.9947× | 1321.04 · 0.9967× |
| fresh-upper | 151 × 2529 × 311 | prefix | 1160.42 | 1154.35 · 0.9948× | 1152.12 · 0.9936× | 1158.79 · 0.9987× |
| control-cropped-deeper | 108 × 1728 × 320 | forward | 592.85 | 591.98 · 1.0002× | 592.38 · 1.0015× | 593.07 · 1.0006× |
| control-cropped-short | 43 × 2291 × 137 | forward | 138.11 | 138.06 · 1.0019× | 137.78 · 0.9948× | 137.92 · 0.9991× |

Against direct oneMKL the strongest fresh result was 151 × 2529 × 311 at 0.9948× and the weakest was 68 × 2176 × 244 at 0.8559×.

### Fastest qualified methods by layout

Each entry gives the winning method, its warm median and its paired time ratio against the prefix wrapper. A ratio of 1.0000× against the same wrapper compares it with itself.

| Layout ID | Independent one-shot | Independent incl. retained | All methods |
| --- | --- | --- | --- |
| repeat-deeper | oneMKL direct-call 638.24 µs · 0.9738× | oneDNN cached 633.96 µs · 0.9672× | COB standalone 633.60 µs · 0.9626× |
| repeat-ragged | OpenBLAS 1127.57 µs · 0.9631× | OpenBLAS 1127.57 µs · 0.9631× | COB standalone 1093.60 µs · 0.9339× |
| repeat-short | oneMKL 147.30 µs · 0.9860× | oneMKL 147.30 µs · 0.9860× | oneMKL 147.30 µs · 0.9860× |
| fresh-thin | LIBXSMM dispatch 33.82 µs · 0.8817× | LIBXSMM retained 33.79 µs · 0.8737× | LIBXSMM retained 33.79 µs · 0.8737× |
| fresh-odd | oneMKL 127.89 µs · 0.9806× | oneMKL 127.89 µs · 0.9806× | Matched forwarder 127.60 µs · 0.9764× |
| fresh-exact | oneMKL direct-call 353.47 µs · 0.8529× | oneMKL direct-call 353.47 µs · 0.8529× | oneMKL direct-call 353.47 µs · 0.8529× |
| fresh-long | oneMKL direct-call 648.21 µs · 0.9842× | oneMKL direct-call 648.21 µs · 0.9842× | oneMKL direct-call 648.21 µs · 0.9842× |
| fresh-high-depth | oneMKL direct-call 1311.01 µs · 0.9896× | oneMKL direct-call 1311.01 µs · 0.9896× | oneMKL direct-call 1311.01 µs · 0.9896× |
| fresh-upper | oneMKL direct-call 1150.71 µs · 0.9913× | oneMKL direct-call 1150.71 µs · 0.9913× | oneMKL direct-call 1150.71 µs · 0.9913× |
| control-cropped-deeper | oneMKL direct-call 590.61 µs · 0.9970× | oneMKL direct-call 590.61 µs · 0.9970× | oneMKL direct-call 590.61 µs · 0.9970× |
| control-cropped-short | oneMKL 138.06 µs · 1.0019× | oneMKL 138.06 µs · 1.0019× | Matched forwarder 137.78 µs · 0.9948× |

### First calls

Each worker timed one complete multiplication after process and library initialization. The peer for each ranked tier is the same warm-selected peer used above. The range covers the six fresh shapes’ individual paired medians.

| First-call comparison | Six-layout geomean | Range of layout medians |
| --- | --- | --- |
| Direct oneMKL | 0.9919× | 0.9459×–1.0241× |
| Matched forwarder | 0.9894× | 0.9485×–1.0120× |
| Two-call split | 1.0028× | 0.9885×–1.0201× |
| Fastest qualified independent one-shot | 0.7254× | 0.1566×–1.0091× |
| Fastest qualified independent incl. retained | 0.7244× | 0.1554×–1.0091× |
| Fastest across all qualified methods | 0.7251× | 0.1554×–1.0091× |

The 68 × 2176 × 244 compact case had the following median first-call times across six workers. Prefix wrapper 1206.82 µs, oneMKL 1141.77 µs, Matched forwarder 1144.28 µs, Two-call split 1207.73 µs.

### Excluded measurements

| Method | Tier and plan policy | Layout IDs | Recorded reasons |
| --- | --- | --- | --- |
| BLASFEO | independent, one-shot | repeat-ragged | worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 · missing, duplicate or invalid FP64 fixtures · missing, duplicate or invalid matched timings |

### Qualification and binary identity

142/143 method/layout combinations qualified. There were 279,554,950 FP64-reference output checks, including 111,821,980 in four dedicated accuracy fixtures per complete combination. The maximum recorded absolute error was 2.97693568e-05. The checker inspected 19,586,170 guard or padding locations. The timing workers executed 27,264 full warm multiplications in 852 batches on logical CPU 0.

The native diagnostics verified 33 wrapper/layout routes and 24 bit-identical full wrapper outputs across 2 forwarding layouts and 4 fixtures per layout. Each active prefix route issued one eight-column vector computation and one N−8 provider call. Each active split route issued provider calls of widths eight and N−8. [Route evidence](replica-1/raw/prefix-routes.csv) · [Bypass parity](replica-1/raw/prefix-bypass-parity.csv)

The common sequential oneMKL adapter has SHA-256 `c1b5bb0eadf3b661d579031fc6d0b65218a25f3c9dc96d16ff22783eba29705a`. The linked oneMKL distribution was 2026.2.1, runtime build 20260915. The build and route records give ordinary and diagnostic wrapper identities, the oneMKL provider description, and the dimensions, strides and offsets of each observed call. [Wrapper builds](replica-1/raw/prefix-builds.csv) · [Library identity](replica-1/raw/mkl-build.txt) · [Provider observations](replica-1/raw/prefix-provider.txt) · [Call arguments](replica-1/raw/prefix-provider-calls.csv) · [Ordinary executables and adapters](replica-1/raw/methods.csv)

| Timed wrapper | Ordinary binary SHA-256 |
| --- | --- |
| Matched forwarder | `01398d24966f07a7c090986e2c0e1e0ef22c85830705b35f7ffe9870ae05aa7b` |
| Two-call split | `4241947cdd65c27193b5e9867fcd3aedb8771c325a44eda9f75757dae546393c` |
| Prefix wrapper | `082a01e758c6eae81db55c5cde0f056a28b433677ddf2b61984e9e5694951fa5` |

The timed oneMKL direct-call method uses the JIT-capable direct-call macro. A separate unranked probe invoked the explicit oneMKL JIT API for each exact geometry.
Across the eleven registered layouts the API reported 9 MKL_JIT_SUCCESS, 2 MKL_NO_JIT. The separate 16 × 16 × 16 control returned MKL_JIT_SUCCESS. [Explicit JIT API outcomes](replica-1/raw/explicit-jit.csv)

The strict FP32 oneDNN processes reported brg_matmul:avx512_core (220 workers). [Measured implementations and plan counts](replica-1/raw/onednn-routes.csv)

On the nine active Intel layouts the matched forwarder passed complete FP64 checks and passed through the original problem to the observed oneMKL binary. Direct cross-binary output hashes were recorded on the two nonaligned Intel bypasses for all four registered input fixtures.

## Reproduction

Run `python3 -B packet.py` in the extracted evidence directory to replay the validated scores. Each per-job raw directory contains the observed leading dimensions and pointer residues, every warm observation, first-call times, numerical checks, eligibility and exclusions. The [comparison CSV](comparison.csv) includes the six paired warm and six paired first-call ratios for each qualified method against the candidate. oneDNN one-shot creates a plan for each call, while its cached variant reuses a plan after creation. LIBXSMM per-call dispatch performs a kernel lookup each time and its retained variant reuses the kernel pointer.
