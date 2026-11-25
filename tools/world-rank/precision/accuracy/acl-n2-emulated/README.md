# ACL BF16 instruction-path correctness under Neoverse N2 emulation

The published Arm Compute Library 53.3.2 BFMMLA assembly executed all registered numerical cases under QEMU 8.2.2 with the Neoverse N2 CPU model. The ordinary composite and a separately linked diagnostic composite returned bit-identical outputs on all **228 matrix/method combinations**. The diagnostic intercepted **414 calls** to `a64_interleaved_bf16fp32_mmla_8x12`. Each call matched the expected matrix dimensions, padded reduction depth, packed operand bytes, scratch allocation and number of invocations. This run measured correctness only. Physical-host arithmetic and speed are separate observations.

| Method | Ordinary outputs above γₖ, of 1,378 | Above 2 ULP, of 553 well-conditioned outputs | Correct-rounding mismatches, of 1,378 | Matrices passing companion max/RMS, of 18 |
|---|---:|---:|---:|---:|
| Direct COB FP32 | 0 | 39 | 776 | 18 |
| Raw BF16 | 713 | 536 | 968 | 6 |
| Three products | 42 | 525 | 966 | 6 |
| Six products | 0 | 17 | 474 | 18 |
| Nine products | 0 | 17 | 231 | 18 |
| Six products in three calls | 0 | 17 | 473 | 18 |

Six, nine and three-call six passed every registered gate. Their worst matrix maximum error reached 93.23% of the companion limit, and the worst RMS error reached 64.69%. The rescue produced the correctly rounded value or correct nonfinite result class on all 1,086 range and exceptional outputs in each guarded method. In the constructed small-residual family, six and three-call six lost 86 representable nonzero outputs while nine recovered the full 126. The BFMMLA path differed from the scalar-FMA test double on 35/108 component-path matrix hashes after normalizing NaN payloads and signed zero. The registered envelope allowed both sets of observations.

The build linked the actual COB FP32 release at f322a9ab58454762563b97323fdc5f29016cc39f, configured for N2 with vendor leaves disabled, and compiled the revised mixed core with GCC 13.3.0. Both adapters resolved the same Arm Compute Library DSO, SHA-256 321fd6d02efe5d0cd1fa126f581d2addb5b36bdf7ab3c288c7b4d9c55bb5e8f0. Numerical input identities and thresholds were frozen before this run.

The full per-case results include the FP32 controls, zero-depth overwrite checks, exceptional classes, native plan depths, component counts, direct-control comparisons, and correctly rounded output witnesses. Recheck the call-by-call evidence from the parent directory with `python3 -B acl-n2-audit.py acl-n2-emulated`. [Ordinary numerical results](ordinary/cases.csv) · [Per-method verdicts](ordinary/verdicts.json) · [Diagnostic trace](trace.log) · [Verified call census](audit.json) · [Binary and compiler identities](identities.txt)
