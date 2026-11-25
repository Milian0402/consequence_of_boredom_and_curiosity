# An eight-column prefix before sequential oneMKL

All validated assignments used AMD processors and exercised the forwarding control. Intel prefix performance remains unmeasured.

The candidate uses an AVX2/FMA kernel for the first eight columns and sends the remaining N−8 columns to sequential oneMKL on nine eligible Intel layouts. The matched forwarder calls oneMKL on all N columns. The split control calls oneMKL on eight columns, then on N−8. On the two cropped bypass layouts each wrapper forwards the full product. Every timing covers the complete single-threaded row-major FP32 overwrite operation.

A ratio above 1 favors the prefix wrapper. Each warm layout ratio is the median of six paired peer-to-wrapper times, three rounds per seed with 32 completed calls per round. Each of the six fresh layouts has equal weight in their geometric mean. Three historical anchors and two bypass controls are reported individually. Independent leaders have the lowest qualified warm median in their tier on that job and layout. The all-method ranking also admits standalone COB and the three assisted wrappers.

| Native job | Intel gate | vs direct oneMKL | vs forwarder | vs split | vs fastest qualified one-shot | vs fastest qualified incl. retained |
| --- | --- | --- | --- | --- | --- | --- |
| [AMD EPYC 7763 64-Core Processor, replica 1](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139521811697) | AMD forwarding control | 1.0005× | 1.0006× | 1.0010× | 0.9043× | 0.9024× |
| [AMD EPYC 9V45 96-Core Processor, replica 2](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45704182519/job/139521811716) | AMD forwarding control | 0.9985× | 1.0003× | 0.9993× | 0.4173× | 0.4125× |

Promotion requires complete qualification of the assisted wrappers and direct oneMKL on all eleven layouts, including route and bitwise bypass checks. Both direct and matched forwarder comparisons must reach 1.03× in the six-shape geometric mean, reach 1.03× on the same five or more fresh shapes and stay at or above 0.97× on all nine active shapes. The split control measures the effect of replacing the first vendor call with the prefix kernel.

## AMD EPYC 7763 64-Core Processor · replica 1

**AMD forwarding control.** All eleven geometries use whole-matrix oneMKL forwarding in all three wrappers. Their measured differences describe that path.

| Warm comparison | Six-layout geomean | Case wins | Round wins | Seed 1 | Seed 2 | Leave-one-layout-out range |
| --- | --- | --- | --- | --- | --- | --- |
| Direct oneMKL | 1.0005× | 3/6 | 22/36 | 0.9981× | 1.0015× | 0.9990×–1.0026× |
| Matched forwarder | 1.0006× | 3/6 | 19/36 | 1.0007× | 1.0013× | 0.9993×–1.0013× |
| Two-call split | 1.0010× | 4/6 | 22/36 | 0.9994× | 1.0017× | 0.9995×–1.0017× |
| Fastest qualified independent one-shot | 0.9043× | 0/6 | 0/36 | 0.9028× | 0.9048× | 0.8932×–0.9392× |
| Fastest qualified independent incl. retained | 0.9024× | 0/6 | 0/36 | 0.9036× | 0.9017× | 0.8927×–0.9368× |
| Fastest across all qualified methods | 0.8805× | 0/6 | 1/36 | 0.8800× | 0.8797× | 0.8707×–0.9096× |

On the fresh shapes the registered independent one-shot field was complete and the retained-inclusive field was complete. The prefix wrapper had the lowest warm median on 0/6 rankable fresh shapes.

### Warm whole-call measurements

Each control cell gives its median time in microseconds and its median paired ratio against the prefix wrapper. A “forward” candidate route covers one full-width oneMKL call.

| Layout ID | M × N × K | Candidate route | Prefix µs | Direct oneMKL | Matched forwarder | Two-call split |
| --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | forward | 1335.87 | 1331.91 · 0.9971× | 1332.16 · 0.9985× | 1329.73 · 0.9969× |
| repeat-ragged | 137 × 2179 × 347 | forward | 2398.49 | 2401.93 · 0.9987× | 2406.17 · 1.0000× | 2407.65 · 1.0019× |
| repeat-short | 43 × 2291 × 137 | forward | 374.81 | 374.17 · 1.0046× | 373.19 · 1.0029× | 372.90 · 0.9973× |
| fresh-thin | 29 × 1229 × 103 | forward | 110.61 | 110.69 · 1.0083× | 111.25 · 1.0068× | 110.97 · 1.0088× |
| fresh-odd | 44 × 1897 × 151 | forward | 325.00 | 321.89 · 0.9900× | 322.16 · 0.9967× | 324.19 · 0.9981× |
| fresh-exact | 68 × 2176 × 244 | forward | 854.42 | 853.46 · 0.9979× | 852.25 · 0.9980× | 853.05 · 0.9979× |
| fresh-long | 83 × 3067 × 271 | forward | 1743.96 | 1753.14 · 1.0041× | 1748.14 · 1.0016× | 1743.67 · 1.0004× |
| fresh-high-depth | 112 × 3581 × 337 | forward | 3138.37 | 3144.80 · 0.9993× | 3142.98 · 0.9999× | 3142.04 · 1.0008× |
| fresh-upper | 151 × 2529 × 311 | forward | 2784.80 | 2794.92 · 1.0036× | 2787.83 · 1.0005× | 2792.45 · 1.0002× |
| control-cropped-deeper | 108 × 1728 × 320 | forward | 1377.22 | 1376.08 · 0.9973× | 1376.05 · 0.9976× | 1374.62 · 0.9972× |
| control-cropped-short | 43 × 2291 × 137 | forward | 379.59 | 381.31 · 1.0008× | 379.53 · 0.9993× | 376.41 · 0.9939× |

### Fastest qualified methods by layout

Each entry gives the winning method, its warm median and its paired time ratio against the prefix wrapper. A ratio of 1.0000× against the same wrapper compares it with itself.

| Layout ID | Independent one-shot | Independent incl. retained | All methods |
| --- | --- | --- | --- |
| repeat-deeper | BLASFEO 1325.94 µs · 0.9940× | BLASFEO 1325.94 µs · 0.9940× | COB standalone 1265.16 µs · 0.9474× |
| repeat-ragged | oneMKL 2401.93 µs · 0.9987× | oneDNN cached 2363.30 µs · 0.9828× | COB standalone 2223.72 µs · 0.9280× |
| repeat-short | AOCL 335.95 µs · 0.8940× | oneDNN cached 330.07 µs · 0.8827× | COB standalone 310.44 µs · 0.8283× |
| fresh-thin | AOCL 82.90 µs · 0.7486× | AOCL 82.90 µs · 0.7486× | AOCL 82.90 µs · 0.7486× |
| fresh-odd | AOCL 312.42 µs · 0.9624× | oneDNN cached 308.74 µs · 0.9529× | COB standalone 292.85 µs · 0.9049× |
| fresh-exact | AOCL 794.70 µs · 0.9318× | AOCL 794.70 µs · 0.9318× | COB standalone 793.70 µs · 0.9301× |
| fresh-long | BLASFEO 1597.47 µs · 0.9146× | oneDNN cached 1592.61 µs · 0.9120× | COB standalone 1509.67 µs · 0.8660× |
| fresh-high-depth | BLASFEO 2991.08 µs · 0.9516× | BLASFEO 2991.08 µs · 0.9516× | COB standalone 2922.22 µs · 0.9313× |
| fresh-upper | BLASFEO 2606.89 µs · 0.9363× | BLASFEO 2606.89 µs · 0.9363× | COB standalone 2552.30 µs · 0.9173× |
| control-cropped-deeper | BLASFEO 1324.93 µs · 0.9608× | BLASFEO 1324.93 µs · 0.9608× | COB standalone 1276.15 µs · 0.9257× |
| control-cropped-short | BLASFEO 363.85 µs · 0.9569× | oneDNN cached 332.53 µs · 0.8730× | COB standalone 312.28 µs · 0.8231× |

### First calls

Each worker timed one complete multiplication after process and library initialization. The peer for each ranked tier is the same warm-selected peer used above. The range covers the six fresh shapes’ individual paired medians.

| First-call comparison | Six-layout geomean | Range of layout medians |
| --- | --- | --- |
| Direct oneMKL | 1.0156× | 0.9827×–1.0419× |
| Matched forwarder | 1.0155× | 0.9876×–1.0425× |
| Two-call split | 0.9884× | 0.9751×–1.0064× |
| Fastest qualified independent one-shot | 0.3570× | 0.0921×–0.7273× |
| Fastest qualified independent incl. retained | 0.5344× | 0.0921×–1.3165× |
| Fastest across all qualified methods | 0.2930× | 0.0921×–0.5521× |

The 68 × 2176 × 244 compact case had the following median first-call times across six workers. Prefix wrapper 2411.86 µs, oneMKL 2444.70 µs, Matched forwarder 2477.87 µs, Two-call split 2378.79 µs.

### Excluded measurements

| Method | Tier and plan policy | Layout IDs | Recorded reasons |
| --- | --- | --- | --- |
| BLASFEO | independent, one-shot | repeat-ragged | worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 · missing, duplicate or invalid FP64 fixtures · missing, duplicate or invalid matched timings |

### Qualification and binary identity

142/143 method/layout combinations qualified. There were 279,554,950 FP64-reference output checks, including 111,821,980 in four dedicated accuracy fixtures per complete combination. The maximum recorded absolute error was 2.97693568e-05. The checker inspected 19,586,170 guard or padding locations. The timing workers executed 27,264 full warm multiplications in 852 batches on logical CPU 0.

The native diagnostics verified 33 wrapper/layout routes and 132 bit-identical full wrapper outputs across 11 forwarding layouts and 4 fixtures per layout. The diagnostics observed one full-width provider call per wrapper on every layout. [Route evidence](replica-1/raw/prefix-routes.csv) · [Bypass parity](replica-1/raw/prefix-bypass-parity.csv)

The common sequential oneMKL adapter has SHA-256 `20b7233375a687c24cd07383cf4eade9c3dbff28f5733f0cd108cd67e56cce82`. The linked oneMKL distribution was 2026.2.1, runtime build 20260915. The build and route records give ordinary and diagnostic wrapper identities, the oneMKL provider description, and the dimensions, strides and offsets of each observed call. [Wrapper builds](replica-1/raw/prefix-builds.csv) · [Library identity](replica-1/raw/mkl-build.txt) · [Provider observations](replica-1/raw/prefix-provider.txt) · [Call arguments](replica-1/raw/prefix-provider-calls.csv) · [Ordinary executables and adapters](replica-1/raw/methods.csv)

| Timed wrapper | Ordinary binary SHA-256 |
| --- | --- |
| Matched forwarder | `01398d24966f07a7c090986e2c0e1e0ef22c85830705b35f7ffe9870ae05aa7b` |
| Two-call split | `4241947cdd65c27193b5e9867fcd3aedb8771c325a44eda9f75757dae546393c` |
| Prefix wrapper | `082a01e758c6eae81db55c5cde0f056a28b433677ddf2b61984e9e5694951fa5` |

The timed oneMKL direct-call method uses the JIT-capable direct-call macro. A separate unranked probe invoked the explicit oneMKL JIT API for each exact geometry.
Across the eleven registered layouts the API reported 11 MKL_NO_JIT. The separate 16 × 16 × 16 control returned MKL_NO_JIT. [Explicit JIT API outcomes](replica-1/raw/explicit-jit.csv)

The strict FP32 oneDNN processes reported brg_matmul:avx2 (220 workers). [Measured implementations and plan counts](replica-1/raw/onednn-routes.csv)

## AMD EPYC 9V45 96-Core Processor · replica 2

**AMD forwarding control.** All eleven geometries use whole-matrix oneMKL forwarding in all three wrappers. Their measured differences describe that path.

| Warm comparison | Six-layout geomean | Case wins | Round wins | Seed 1 | Seed 2 | Leave-one-layout-out range |
| --- | --- | --- | --- | --- | --- | --- |
| Direct oneMKL | 0.9985× | 2/6 | 19/36 | 0.9985× | 1.0001× | 0.9968×–0.9996× |
| Matched forwarder | 1.0003× | 4/6 | 19/36 | 0.9990× | 1.0048× | 0.9985×–1.0035× |
| Two-call split | 0.9993× | 3/6 | 20/36 | 0.9990× | 0.9995× | 0.9981×–1.0006× |
| Fastest qualified independent one-shot | 0.4173× | 0/6 | 0/36 | 0.4160× | 0.4166× | 0.4107×–0.4309× |
| Fastest qualified independent incl. retained | 0.4125× | 0/6 | 0/36 | 0.4110× | 0.4119× | 0.4066×–0.4252× |
| Fastest across all qualified methods | 0.4125× | 0/6 | 0/36 | 0.4110× | 0.4119× | 0.4066×–0.4252× |

On the fresh shapes the registered independent one-shot field was complete and the retained-inclusive field was complete. The prefix wrapper had the lowest warm median on 0/6 rankable fresh shapes.

### Warm whole-call measurements

Each control cell gives its median time in microseconds and its median paired ratio against the prefix wrapper. A “forward” candidate route covers one full-width oneMKL call.

| Layout ID | M × N × K | Candidate route | Prefix µs | Direct oneMKL | Matched forwarder | Two-call split |
| --- | --- | --- | --- | --- | --- | --- |
| repeat-deeper | 108 × 1728 × 320 | forward | 1087.28 | 1085.70 · 0.9996× | 1095.87 · 1.0045× | 1091.05 · 1.0005× |
| repeat-ragged | 137 × 2179 × 347 | forward | 1958.75 | 1942.21 · 1.0013× | 1944.69 · 0.9931× | 1948.99 · 1.0011× |
| repeat-short | 43 × 2291 × 137 | forward | 284.33 | 285.03 · 1.0036× | 288.46 · 1.0185× | 290.32 · 1.0138× |
| fresh-thin | 29 × 1229 × 103 | forward | 79.94 | 79.97 · 0.9963× | 80.16 · 1.0073× | 80.67 · 1.0050× |
| fresh-odd | 44 × 1897 × 151 | forward | 252.46 | 253.25 · 1.0068× | 251.51 · 1.0059× | 251.64 · 1.0027× |
| fresh-exact | 68 × 2176 × 244 | forward | 688.72 | 691.59 · 0.9925× | 694.03 · 1.0086× | 691.02 · 0.9984× |
| fresh-long | 83 × 3067 × 271 | forward | 1360.81 | 1340.46 · 0.9933× | 1363.97 · 0.9847× | 1357.42 · 0.9925× |
| fresh-high-depth | 112 × 3581 × 337 | forward | 2526.02 | 2523.88 · 0.9963× | 2492.87 · 0.9861× | 2534.85 · 0.9938× |
| fresh-upper | 151 × 2529 × 311 | forward | 2207.35 | 2200.54 · 1.0055× | 2242.26 · 1.0097× | 2218.54 · 1.0032× |
| control-cropped-deeper | 108 × 1728 × 320 | forward | 1109.25 | 1090.09 · 0.9929× | 1088.27 · 0.9802× | 1105.87 · 0.9924× |
| control-cropped-short | 43 × 2291 × 137 | forward | 292.62 | 289.02 · 0.9787× | 290.87 · 1.0031× | 288.46 · 1.0051× |

### Fastest qualified methods by layout

Each entry gives the winning method, its warm median and its paired time ratio against the prefix wrapper. A ratio of 1.0000× against the same wrapper compares it with itself.

| Layout ID | Independent one-shot | Independent incl. retained | All methods |
| --- | --- | --- | --- |
| repeat-deeper | AOCL 462.27 µs · 0.4252× | AOCL 462.27 µs · 0.4252× | AOCL 462.27 µs · 0.4252× |
| repeat-ragged | AOCL 839.48 µs · 0.4293× | oneDNN cached 812.67 µs · 0.4153× | oneDNN cached 812.67 µs · 0.4153× |
| repeat-short | AOCL 119.83 µs · 0.4208× | AOCL 119.83 µs · 0.4208× | AOCL 119.83 µs · 0.4208× |
| fresh-thin | LIBXSMM dispatch 28.40 µs · 0.3554× | LIBXSMM retained 28.31 µs · 0.3546× | LIBXSMM retained 28.31 µs · 0.3546× |
| fresh-odd | AOCL 111.78 µs · 0.4521× | oneDNN cached 111.63 µs · 0.4436× | oneDNN cached 111.63 µs · 0.4436× |
| fresh-exact | AOCL 289.94 µs · 0.4211× | AOCL 289.94 µs · 0.4211× | AOCL 289.94 µs · 0.4211× |
| fresh-long | AOCL 583.58 µs · 0.4295× | oneDNN cached 561.39 µs · 0.4094× | oneDNN cached 561.39 µs · 0.4094× |
| fresh-high-depth | AOCL 1070.94 µs · 0.4230× | AOCL 1070.94 µs · 0.4230× | AOCL 1070.94 µs · 0.4230× |
| fresh-upper | AOCL 951.80 µs · 0.4296× | AOCL 951.80 µs · 0.4296× | AOCL 951.80 µs · 0.4296× |
| control-cropped-deeper | AOCL 463.45 µs · 0.4236× | AOCL 463.45 µs · 0.4236× | AOCL 463.45 µs · 0.4236× |
| control-cropped-short | AOCL 120.96 µs · 0.4128× | AOCL 120.96 µs · 0.4128× | AOCL 120.96 µs · 0.4128× |

### First calls

Each worker timed one complete multiplication after process and library initialization. The peer for each ranked tier is the same warm-selected peer used above. The range covers the six fresh shapes’ individual paired medians.

| First-call comparison | Six-layout geomean | Range of layout medians |
| --- | --- | --- |
| Direct oneMKL | 1.0088× | 0.9872×–1.0331× |
| Matched forwarder | 1.0183× | 0.9850×–1.0606× |
| Two-call split | 0.9952× | 0.9844×–1.0238× |
| Fastest qualified independent one-shot | 0.1948× | 0.1160×–0.2699× |
| Fastest qualified independent incl. retained | 0.3501× | 0.1725×–1.1410× |
| Fastest across all qualified methods | 0.3501× | 0.1725×–1.1410× |

The 68 × 2176 × 244 compact case had the following median first-call times across six workers. Prefix wrapper 2031.41 µs, oneMKL 2034.75 µs, Matched forwarder 2017.88 µs, Two-call split 1990.22 µs.

### Excluded measurements

| Method | Tier and plan policy | Layout IDs | Recorded reasons |
| --- | --- | --- | --- |
| BLASFEO | independent, one-shot | repeat-ragged | worker exit -11: affinity_cpu=0 BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 · missing, duplicate or invalid FP64 fixtures · missing, duplicate or invalid matched timings |

### Qualification and binary identity

142/143 method/layout combinations qualified. There were 279,554,950 FP64-reference output checks, including 111,821,980 in four dedicated accuracy fixtures per complete combination. The maximum recorded absolute error was 2.97693568e-05. The checker inspected 19,586,170 guard or padding locations. The timing workers executed 27,264 full warm multiplications in 852 batches on logical CPU 0.

The native diagnostics verified 33 wrapper/layout routes and 132 bit-identical full wrapper outputs across 11 forwarding layouts and 4 fixtures per layout. The diagnostics observed one full-width provider call per wrapper on every layout. [Route evidence](replica-2/raw/prefix-routes.csv) · [Bypass parity](replica-2/raw/prefix-bypass-parity.csv)

The common sequential oneMKL adapter has SHA-256 `1f98294cff0fb23c5ffbded58c4d4c2dd593dbb8a85c6a92f05ec51df6ea0338`. The linked oneMKL distribution was 2026.2.1, runtime build 20260915. The build and route records give ordinary and diagnostic wrapper identities, the oneMKL provider description, and the dimensions, strides and offsets of each observed call. [Wrapper builds](replica-2/raw/prefix-builds.csv) · [Library identity](replica-2/raw/mkl-build.txt) · [Provider observations](replica-2/raw/prefix-provider.txt) · [Call arguments](replica-2/raw/prefix-provider-calls.csv) · [Ordinary executables and adapters](replica-2/raw/methods.csv)

| Timed wrapper | Ordinary binary SHA-256 |
| --- | --- |
| Matched forwarder | `01398d24966f07a7c090986e2c0e1e0ef22c85830705b35f7ffe9870ae05aa7b` |
| Two-call split | `4241947cdd65c27193b5e9867fcd3aedb8771c325a44eda9f75757dae546393c` |
| Prefix wrapper | `082a01e758c6eae81db55c5cde0f056a28b433677ddf2b61984e9e5694951fa5` |

The timed oneMKL direct-call method uses the JIT-capable direct-call macro. A separate unranked probe invoked the explicit oneMKL JIT API for each exact geometry.
Across the eleven registered layouts the API reported 11 MKL_NO_JIT. The separate 16 × 16 × 16 control returned MKL_NO_JIT. [Explicit JIT API outcomes](replica-2/raw/explicit-jit.csv)

The strict FP32 oneDNN processes reported brg_matmul:avx512_core (220 workers). [Measured implementations and plan counts](replica-2/raw/onednn-routes.csv)

## Reproduction

Run `python3 -B packet.py` in the extracted evidence directory to replay the validated scores. Each per-job raw directory contains the observed leading dimensions and pointer residues, every warm observation, first-call times, numerical checks, eligibility and exclusions. The [comparison CSV](comparison.csv) includes the six paired warm and six paired first-call ratios for each qualified method against the candidate. oneDNN one-shot creates a plan for each call, while its cached variant reuses a plan after creation. LIBXSMM per-call dispatch performs a kernel lookup each time and its retained variant reuses the kernel pointer.
