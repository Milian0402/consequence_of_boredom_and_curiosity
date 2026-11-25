# Widening FP16 and FP32 FMOPA on one Apple M5 Max

Widening FP16 delivered **0.998× the arithmetic throughput of FP32** by the median paired ratio. Across all six alternated pairs, the ratio ranged from 0.987× to 1.007×. Both loops kept their operands in vector registers and cycled through four independent FP32 accumulator tiles.

| Source operands | Multiply-adds per instruction | Median elapsed | Median arithmetic rate |
|---|---:|---:|---:|
| FP32 | 256 | 44.417 ms | 2.146 TFLOP/s |
| FP16, widening to FP32 | 512 | 89.317 ms | 2.135 TFLOP/s |

Both forms executed 186,179,584 FMOPA instructions per observation. With a 64-byte streaming vector the four 16 × 16 accumulator tiles hold 1,024 FP32 results. Each chip zeroed the tiles, dispatched 256 iterations of the same 32-instruction schedule, and used exact signed input constants. Each call checked every element of the final chip and the output guards. Scalar wall-clock timing included the streaming-mode transitions, tile resets, loop control and final store. The test ran on one native macOS thread with user-initiated QoS. macOS selected the core.

On this schedule, the FP16 instruction took about twice as long and performed twice the arithmetic. A compensated multiplication requiring three full FP16 products would need additional gains elsewhere to beat one FP32 product. These register-level measurements do not time packing or a complete SGEMM.

## Reproduction

On a compatible Apple Silicon machine with Xcode command-line tools:

```sh
cc -O3 -march=armv9-a+sme rings.S trial.c -o trial
xcrun llvm-objdump -d --mattr=+sme trial > disassembly.txt
./trial measure > trials.csv 2> platform.txt
python3 -B summarize.py
```

The runner exits with code 77 if the instruction is unavailable. The CSV preserves all six rounds in execution order. The summary verifies the disassembly schedule, recomputes each paired ratio and records the input digests.
