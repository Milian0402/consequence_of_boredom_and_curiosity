# FP16 component products on Apple M5 Max

**None of the four FP16 methods won a paired call** against the fastest accuracy-qualified FP32 implementation on any of six layouts. Raw FP16 failed the field's accuracy checks on all six. Three, six and nine component products each passed on five layouts. On those five, their geometric-mean throughput was **18.0%, 3.95% and 3.11%** of the fastest qualifying FP32 control, respectively. Each call included conversion, packing, multiplication and result restoration. [Recorded timing and accuracy gates](timing/protocol.md) · [Six- and nine-product protocol](timing/higher-precision.md)

## Complete-call results

A comparison restricted to independent libraries produced the same win counts. MaxMulSK led those libraries on every square layout and Accelerate led on both cropped layouts in both runs.

Each cell gives the median of twelve paired FP32-control-time / candidate-time ratios. A ratio above 1 favors the candidate. A dagger marks a candidate that missed an accuracy gate. The controls were timed afresh for the higher-precision panel.

| M × N × K | Fastest qualified FP32, first panel | Raw FP16 | Three terms | Fastest qualified FP32, second panel | Six terms | Nine terms |
|---|---|---:|---:|---|---:|---:|
| 512³ | MaxMulSK | 0.210×† | 0.142× | COB | 0.027× | 0.022× |
| 1024³ | COB | 0.337×† | 0.188× | COB | 0.037× | 0.030× |
| 2048³ | MaxMulSK | 0.506×† | 0.245× | MaxMulSK | 0.050× | 0.041× |
| 4096³ | MaxMulSK | 0.667×† | 0.284×† | MaxMulSK | 0.071×† | 0.055×† |
| 1471 × 1973 × 1559, cropped | Accelerate | 0.472×† | 0.232× | Accelerate | 0.066× | 0.049× |
| 187 × 2711 × 433, cropped | Accelerate | 0.178×† | 0.125× | COB | 0.029× | 0.022× |

At 4096³, COB had the lowest median time in both panels. On the positive fixture, its maximum absolute error was 0.003935 against the 0.002 limit. Accelerate and Arm KleidiAI also exceeded the limit. MaxMulSK passed with 0.001179. The three-, six- and nine-term candidates all reached 0.003522. The raw tables retain every method's timings and accuracy results, including exclusions. [First panel](timing/results/summary.json) · [Second panel](timing/higher/summary.json)

## Numerical behavior and hardware

The separate 19-family, two-seed numerical study used exact dot products and direct-FP32 maximum/RMS error comparisons. With the measured widening SME primitive, six and nine products passed both gates. Three and four products each missed the direct gate on one same-sign input near adjacent floating-point values. The larger positive 4096³ fixture above exposed a further accuracy limit for six and nine products. [Numerical study and saved output words](numerics/README.md)

The M5's widening FP16 outer-product instruction accumulated two depth positions into binary32 in each instruction. In a register-only test with the same four independent ZA tiles and instruction count, it delivered 2.135 TFLOP/s against FP32's 2.146 TFLOP/s. The median paired throughput ratio was **0.9982×**. The whole-call measurements additionally charged splitting, packing, temporary buffers and combining the components. The higher-precision implementation repacked each binary16 plane for each separate SME product. [Instruction-level measurements](apple/ceiling/README.md) · [Measured one-shot kernel and checks](apple/README.md)

## Measurement scope

The two panels each contain 432 warm timings and 144 full-output accuracy records. Centered inputs used both seeds. Positive and paired-cancellation fixtures used the first seed. Accuracy admission required finite outputs, absolute error at most 0.002 and maximum/RMS error within the recorded direct-COB bounds. A single-thread OpenBLAS binary64 GEMM supplied all reference outputs, with 19 scalar dot products checked for each matrix. During timing, the driver checked output finiteness and padding after each call and input hashes after each block. It recorded calling-thread CPU and wall time. macOS core affinity and per-call process thread counts were unavailable in the saved evidence.

The FP32 controls were Apple Accelerate in explicit single-thread mode, standalone COB, MaxMulSK and Arm KleidiAI's elastic FP32 SME2 implementation. MaxMulSK used compact native input on squares and an in-call conversion on the cropped layouts. The controls' binary hashes appear in each panel's identity record. OpenBLAS served as the FP64 reference; its FP32 SGEMM was outside the timed field. [Accelerate adapter](../hk15-stability/accelerate.c) · [MaxMulSK cropped-view adapter](../m5-compact-adapters/compact-adapter.c) · [KleidiAI adapter and provider](../m5-kleidiai-field/README.md) · [Binary identities](timing/results/identity.json)

## Reproduce offline

From this folder in a local COB checkout, run the following command. It regenerates the results from the recorded measurements and numerical outputs without executing a multiply or accessing the network.

```sh
python3 -B verify.py
```

The timing scorer also accepts either result directory independently.

```sh
python3 -B timing/score.py timing/results
python3 -B timing/score.py timing/higher
```

On compatible Apple hardware the native kernel can be rebuilt and its protected-input QA rerun. The historical full-field driver accepts the seven matching local libraries in the order printed by `python3 timing/study.py --help`. The higher-precision panel additionally used the row/column-normalized SME frontend via `--qualified-dylib`. [Apple build instructions](apple/README.md) · [Higher-precision source and numerical replay](numerics/README.md)
