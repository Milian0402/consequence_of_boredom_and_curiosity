# BF16 decomposition accuracy on exact binary32 dot products

The guarded six- and nine-product candidates met the ordinary FP32 error envelope on all **1,378 outputs** in the serial-FMA laboratory backend. Grouping the six terms into three calls also passed. Three products exceeded the envelope 42 times. Among the guarded candidates, nine products was the only one to recover every output on the fixture constructed to isolate the omitted terms. Hardware uses its own accumulation order. These runs establish local arithmetic behavior and contain no timing measurements.

The public C entry point ran on protected cropped inputs beside the actual local COB FP32 control. Its BF16 primitive was an explicitly scalar C `fmaf` test double. The independent integer model reproduced all 108 observed component-path matrices from the guarded build, allowing only NaN payloads and signed zeros to differ. The initial build used ordinary FP32 range rescue. A subsequent build switched that route to binary64 accumulation followed by a single binary32 cast. Both builds produced byte-identical ordinary-case matrices in all six methods.

## Ordinary finite inputs

The 18 matrices comprise nine families with two frozen inputs each. The two-ULP column uses the 553 outputs whose exact dot magnitude was at least a quarter of the sum of absolute products. The companion column counts matrices satisfying both the max-error and RMS-error bounds against direct COB. Each threshold is twice the larger of the direct error and the corresponding half-ULP rounding budget. All 18 matrices were complete and all 1,378 output classes matched in every method.

| Local implementation | Above γₖ | Above 2 ULP when well conditioned | Rounded-result mismatches | Companion max/RMS pass |
|---|---:|---:|---:|---:|
| Direct COB FP32 | 0 | 39 | 776 | 18/18 |
| Raw BF16, one product | 713 | 536 | 968 | 6/18 |
| Three products | 42 | 526 | 966 | 6/18 |
| Six products | 0 | 3 | 431 | 18/18 |
| Nine products | 0 | 3 | 208 | 18/18 |
| Six products in three calls | 0 | 3 | 432 | 18/18 |

The normalized error is |observed − exact| / Σ|aₗbₗ|, compared with γₖ = k/(2²⁴ − k). It measures error relative to the scale of a single dot's terms. A matrix with many nearly cancelling products can pass while its small output has large relative error. For the six-product, nine-product and three-call candidates, the largest ratios to γₖ were 0.048. All three reached at most 0.706 of the companion's matrix maximum-error limit and 0.541 of its RMS limit. The two seeds are deterministic examples; universal accuracy and hardware reproducibility require other evidence.

| Frozen witness | What changed |
|---|---|
| BF16 midpoint ties | Three products exceeded γₖ on 41/154 outputs. Signed full-mantissa operands accounted for the remaining violation. |
| Omitted component products | All 126 exact dots are representable and nonzero. Direct FP32, six products and three-call six returned zero on 86. Raw and three products returned zero on all 126. Nine products recovered all 126 exactly. |
| Adjacent FP32 values with a shared BF16 high part | On 126 small positive residuals, worst relative errors were 6.92% for direct, 100% for raw, 284% for three, 0.517% for six, 0.517% for three-call six, and 0.00000639% for nine. |
| Scaling a positive correlated input by 2¹² | On 169/182 outputs even the nearest FP32 value misses an absolute tolerance of 0.002. Six, nine and three-call six hit that exact lower bound of 169 failures. |
| Signed exact dyadics, near-zero dots, and complementary scales with zero axes | Every tested method returned the correctly rounded result on all 410 outputs from the three HK families. |

The largest **absolute differences from direct COB** over ordinary matrices were 0.25 for six, nine and three-call six. The largest case-level RMS differences were 0.08416, 0.08390 and 0.08416, respectively. The largest RMS differences divided by RMS direct output were 2.89%, 2.86% and 2.89%. The exact references and direct-control differences answer different questions when COB itself loses a small result.

## Range, exceptional values and zero-depth

The guard admits zero and finite nonzero input magnitudes from 2⁻³⁰ through 2³⁰. Each guarded 3/6/9/three-call method delegated all 18 range and exceptional matrices to the selected rescue. The binary64 revision returned the correctly rounded output or expected IEEE class on **all 1,086 cells** in those matrices. For 90 cancellation dots whose individual products exceed binary32 range, the original FP32 rescue and ordinary COB returned nonfinite results. The revision recovered the exact finite answers. This fixture exercises a specific overflow pattern; binary64 accumulation can also lose contributions in sufficiently ill-conditioned sums. These observations belong to each rescue, separately from the BF16 path.

Raw BF16 used its own component route throughout. On the finite conversion-overflow matrices it lost all 126 finite results to infinity, and the intermediate-overflow matrices contributed 90 additional nonfinite results. Correct class alone can conceal underflow. All 110 nonzero target outputs from the FP32-subnormal-scaled family rounded to zero through raw BF16. The guarded range rescue recovered them. All five candidate modes overwrote the 30 zero-depth outputs with zero. The standalone COB control left the initial NaNs untouched there.

## Replay and evidence

Run from this directory with Python 3.10 or later. The models use only the standard library. The actual-library runner uses POSIX memory protection. A matching DSO is required for each host architecture.

```sh
python3 -B registration.py
python3 -B test_accuracy.py
python3 -B evaluate.py --model --label 'unrestricted arithmetic model' --out replay-model
python3 -B evaluate.py --library ../local/shape-math-range64.dylib --label 'scalar test double, binary64 rescue' --out replay-revised
```

For an individual native candidate, supply `--modes 6` and, if its DSO does not export an ordinary FP32 control, `--control-library /path/to/libcob.so`. The native CI must independently verify the hardware and primitive route before making performance claims. Native library failures and missing rows are recorded in `verdicts.json`. In each `cases.csv`, absolute, relative and ULP counts apply to finite observed/reference pairs. Class failures, unavoidable overflow and the number of comparable values have separate columns. `scores.json` records output hashes, exact integer witnesses, implementation names, primitive dimensions and binary identities.

- [Frozen protocol](protocol.md), [operand hashes](registration.json), [companion direct-control policy](direct-control-policy.md).
- [Initial ordinary-rescue checksums](local-scalar/SHA256SUMS) and [original measurements](local-scalar/cases.csv).
- [Revised range-rescue measurements](range64-scalar/cases.csv) and [revised verdicts](range64-scalar/verdicts.json).
- [Unguarded arithmetic models](unrestricted/cases.csv). On extreme finite and nonfinite inputs, decomposing an infinite high part can produce NaN residuals.
- [Henry, Tang and Heinecke's original component-product method](https://arxiv.org/abs/1904.06376) and [Cui and Liu's AMX implementation study](https://arxiv.org/abs/2609.04663).
