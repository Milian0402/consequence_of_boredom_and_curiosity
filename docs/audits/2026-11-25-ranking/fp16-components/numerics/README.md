# Numerical evidence for FP16 component products on Apple M5 Max

Three FP16 products passed the original FP32 dot-error envelope on all **1,378 ordinary outputs**, but failed the direct-FP32 maximum and RMS comparison on one of 18 ordinary matrices with native Apple SME. The same cancellation matrix failed when the three component products were summed exactly before one final FP32 rounding. Four products still failed that matrix. Six and nine products passed every registered gate with SME. The **six- and nine-product whole-call measurements** in the companion field report used `sme-0`, the general frontend with A-row and B-column scaling and depth balancing disabled.

| Component products | Original numerical gate | Serial scalar FMA, companion matrices passed | Arm FMLAL, serial | Apple SME FMOPA | Pairwise FMLAL | Exact component sum |
|---|---|---:|---:|---:|---:|---:|
| 1 | Fail | 9/18 | 9/18 | 10/18 | 10/18 | 10/18 |
| 3 | Pass | 16/18 | 16/18 | 17/18 | 17/18 | 17/18 |
| 4 | Pass | 16/18 | 16/18 | 17/18 | 17/18 | 17/18 |
| 6 | Pass | 17/18 | 17/18 | 18/18 | 18/18 | 18/18 |
| 9 | Pass | 17/18 | 17/18 | 18/18 | 18/18 | 18/18 |

The original gate covers the ordinary componentwise γₖ envelope, exact dyadics, zero-depth overwrite, and range and nonfinite result classes. The companion gate compares each matrix's maximum and RMS errors to twice the larger of the direct COB error and the corresponding half-ULP budget. The exact-sum column covers the 18 ordinary matrices only. Both scaling policies produced the same ordinary pass counts. The tested scalar and serial FMLAL paths also exceeded the maximum bound on a BF16-midpoint family. The SME and pairwise reductions passed that family.

On the failing adjacent-ULP cancellation draw (seed 0x51D9A47B), SME three-product maximum and RMS errors were 3.139×10⁻⁶ and 1.591×10⁻⁶. Their limits were 2.785×10⁻⁶ and 1.311×10⁻⁶. On the same matrix, six products gave 6.54×10⁻¹⁰ and 2.62×10⁻¹⁰. The direct comparison used the same physical host, input bits and public COB FP32 code in every case. Passing a finite corpus does not establish correctly rounded arithmetic on arbitrary inputs.

## One scalar multiplication explains the limitation

For K=1 and A=B=1+2⁻¹¹, the rounded high half is 1 and the low contribution is exactly 2⁻¹¹. Three products return 1+2⁻¹⁰. The exact binary32 answer also contains 2⁻²² from the low-by-low product, a difference of **two FP32 ULPs**. Four products recover it. A second case squares binary32 word `3f800801`. The two-plane input split loses information, and both the three- and four-product SME results differ from the reference by three ULPs. All four hardware/software accumulation variants reproduced these examples. [Recorded input and output words](data/scalar-witnesses.csv)

The C splitter was checked on every positive normal binary32 word with unbiased exponent from −14 through 12 after scaling. Two planes failed exact reconstruction on 62,914,560 of 226,492,416 values. Three reconstructed every one exactly, including 4,194,304 values whose third plane was a subnormal half. The host FP control register was zero. Separate products and partial-sum rounding can still change the output after exact input reconstruction. [Exhaustive split result](data/split-audit.txt)

## Scaling, exceptional inputs and reduction order

For each normalized matrix element x, the splitter rounds x to the nearest-even half and multiplies its residual by 2¹¹ before rounding the second half. Three-plane modes repeat that residual step. The frontend rescales each FP32 component dot by its exact power-of-two weight and merges low weights first. Each of the 18 ordinary matrices took the component path in every configuration. The row/column policy completed 24 of all 38 inputs on that path, sent 12 through binary64 rescue and zeroed two zero-depth outputs. Reciprocal per-depth scaling admitted both broad-exponent draws and reduced rescues to 10. Nonfinite inputs, excessive per-axis exponent spread, and overflow after final FP32 rescaling have distinct recorded reasons. The latter can occur after all requested component products have executed. [Frozen numerical contract](contract.md)

The pairwise hardware path and the exact integer component-sum diagnostic were specified after the serial result. The independent exact calculation uses the stored half-component expansion, sums all retained products in integer units and rounds once to binary32. All nine terms then gave the correctly rounded result on all 1,378 ordinary outputs. Exact six terms missed 126 rounded outputs yet passed every matrix-level direct comparison. [Follow-up design](reduction-followup.md) · [Exact component results](data/exact-component-sums.csv)

## Reproduce the numerical scores

From this directory in the source repository, use Python 3.10 or later. A separate staging checkout can be supplied with `--repo path/to/checkout`. Both commands use the existing pinned evaluator under `tools/world-rank/precision/accuracy`.

```sh
python3 -B replay.py
python3 -B exact-components.py
```

The offline replay checks 38 operand pairs, all 1,824 method executions, 174 unique input/result summaries, 297 exact error witnesses, and 48 method verdicts. Raw binary32 output and control arrays are stored once per SHA-256 digest as little-endian hexadecimal. The 171 arrays contain all observed words including NaNs and signed zeros. `executions.csv` joins each case, mode and scaling policy to `scores.csv`. The latter identifies the observed and direct-control arrays in `words.csv`. [Per-method verdicts](data/verdicts.csv) · [Executed routes and term counts](data/executions.csv) · [Scores](data/scores.csv) · [Raw output and control words](data/words.csv) · [Physical host](data/platform.txt) · [Original binary and record hashes](data/identities.csv)

The exact measured common frontend is in `source/fp16.c`, and its SME adapter is in `source/sme-bridge.c`. The sibling `apple/plane.c` and `apple/tiles.c` supply the genuine SME half-plane product. The existing `tools/world-rank/precision/core/fp32.c` supplies the direct COB call and binary64 rescue. On a compatible Apple Arm host, the optional build script links a separately built PIC `libcob_matmul.a` from release f322a9ab58454762563b97323fdc5f29016cc39f. The original dynamic and rebuilt static control linkage can produce different library hashes. Preserve the observed-word comparison when validating a rebuild.

```sh
bash source/build.sh path/to/libcob_matmul.a scratch-build
python3 -B native-check.py --library scratch-build/sme-0.dylib --label SME-row-column --out scratch-results
```
