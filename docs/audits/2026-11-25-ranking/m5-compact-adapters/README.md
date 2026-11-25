# Compact-only matrix multipliers on padded M5 Max inputs

The installed COB option beat both adapted competitors on **all six** independently selected padded thin layouts. Its geometric-mean whole-call speedup was **3.08× over MaxMulSK** and **2.77× over MpGEMM**. Both external entry points require compact row-major A, B and C. The adapters allocate one aligned workspace per call, copy logical A and B rows into compact form, invoke the actual native kernel, copy compact C into the caller's padded output and free the workspace. Every step is timed. The inputs are ordinary FP32 matrices and the adapters retain no state between multiplies beyond the library entry point.

The earlier direct-shape comparison included these two libraries on compact layouts. This supplement adds their adapted performance on the six padded fresh layouts. Their native arithmetic plus conversion is slower than COB on all six, while serial Accelerate or prepared LIBXSMM remains faster on four of six. Against the fastest external of this five-method supplemental field, COB averages 0.956×.

| M × N × K | Pad | vs adapted MaxMulSK | vs adapted MpGEMM | vs serial Accelerate | vs prepared LIBXSMM | vs fastest external |
|---|---:|---:|---:|---:|---:|---:|
| 28 × 1057 × 107 | 9 | 3.258× | 2.775× | 1.255× | 0.879× | 0.879× |
| 31 × 4033 × 199 | 7 | 2.638× | 2.488× | 1.289× | 1.134× | 1.134× |
| 34 × 1799 × 241 | 3 | 3.460× | 2.798× | 0.965× | 0.900× | 0.900× |
| 46 × 2983 × 353 | 13 | 3.691× | 2.865× | 1.077× | 1.577× | 1.077× |
| 49 × 2059 × 127 | 5 | 3.541× | 3.208× | 0.902× | 1.121× | 0.902× |
| 52 × 3403 × 283 | 7 | 2.194× | 2.541× | 0.876× | 1.370× | 0.876× |

Each ratio is the median of six matched warm trials across two seeds. Each method also ran once in its own freshly started process for every layout and seed. The geometric-mean process-first ratio was 4.37× against adapted MaxMulSK and 3.05× against adapted MpGEMM. Fresh processes load each library before the clock starts. Explicit initialization and JIT setup are charged for direct LIBXSMM in the process-first comparison. The warm result for LIBXSMM calls its already generated kernel.

All 180 warm observations and 60 separate process-first calls passed, including 31,542,500 full-output FP64 comparisons after first calls and measured batches. The largest absolute error was 0.00002476 on the timing inputs. Each adapted library and COB also passed 24 protected-memory cases covering 2,642,118 result cells per method, using read-only A/B, guard pages immediately after each row, output-prefix canaries, and normalized, cancellation and mixed-scale inputs. Mach reported one thread at the checked times. The original, separately frozen acceptance rules and two independent installed results are in the thin-layout report.

Run `python3 analyze.py` from the unpacked dataset to recalculate its scores. The optional native command is `python3 measure.py COB.dylib Accelerate.dylib MaxMulSK.dylib MpGEMM.dylib LIBXSMM_CHECKOUT NEW_OUTPUT_DIR`. Each dylib implements the saved row-major benchmark ABI, including an explicit serial setting for Accelerate. The real native peer and compiled adapter hashes accompany the records.
