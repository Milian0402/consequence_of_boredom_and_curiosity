# M5 Max compact FP32 field including Arm KleidiAI

On eight independently frozen compact FP32 layouts, COB beat **both** Arm KleidiAI v1.33.0 SME2 implementations on every warm layout. COB averaged **1.749× the fixed kernel** and **1.784× the elastic kernel**, with all per-call packing and workspace included. Against the **fastest external method for each layout** in the expanded field, COB achieved **1.053×** geometric-mean speedup on this M5 Max and won by at least 1% on six of the eight. The corresponding ratio in the separate fresh-process comparison was **0.922×**.

The exact production source was `69b239d6ab9b9a86425ef6874634e3a1952a7f73` with `COB_ENABLE_APPLE_MEDIUM_WIDE=ON`. It was compared against its parent `602a9dddce7e940a282b0716a7b2ca40bd2ad35e` with the same option. The warm workload supplies ordinary compact row-major matrix pointers to the public interfaces. Consecutive warm trials reuse the same buffers and contents across all methods, while both COB and both KleidiAI adapters repack on every call. Prepared LIBXSMM reuses generated code for the shape. Each adapter for KleidiAI allocates scratch, creates zero bias, packs A and B with the compatible upstream routines, invokes the actual native FP32 reduction and frees the scratch. The fixed and elastic kernels are separate implementations from the same library.

| M × N × K | COB µs | vs KleidiAI fixed | vs KleidiAI elastic | Fastest external | vs fastest |
|---|---:|---:|---:|---|---:|
| 24 × 1088 × 280 | 14.50 | 1.445× | 1.584× | prepared-LIBXSMM | 0.974× |
| 27 × 2240 × 96 | 8.09 | 2.018× | 1.622× | MpGEMM | 1.120× |
| 29 × 2624 × 216 | 22.42 | 1.800× | 1.826× | MpGEMM | 1.048× |
| 32 × 3904 × 368 | 100.15 | 2.027× | 2.290× | MpGEMM | 1.049× |
| 25 × 2287 × 115 | 10.94 | 1.836× | 1.627× | prepared-LIBXSMM | 1.096× |
| 26 × 3487 × 307 | 57.69 | 1.711× | 1.980× | MpGEMM | 1.066× |
| 30 × 1043 × 275 | 13.83 | 1.434× | 1.562× | Accelerate | 0.971× |
| 31 × 3135 × 187 | 27.67 | 1.827× | 1.895× | MpGEMM | 1.109× |

The other independent contenders were single-thread Apple Accelerate, OpenBLAS, ArmPL, MaxMulSK, MpGEMM and direct prepared LIBXSMM. Neither KleidiAI variant displaced the external leader of any layout in this run. At 24 × 1088 × 280, prepared LIBXSMM led. At 30 × 1043 × 275, Accelerate and prepared LIBXSMM had equal median absolute times to the clock resolution. The scorer breaks exact median-time ties by library name. Their individual paired COB ratios were 0.971× and 0.951× respectively. COB lost at these two geometries across all three independent comparisons. The earlier research and integrated studies had geometric means of 1.032× and 1.019× against their six external competitors.

Each of the 11,616 warm observations belongs to a position- and predecessor-balanced sequence, across two seeds per shape. Every implementation received identical A, B and C pointer values. Two benchmark labels referred to the exact same incumbent function; their eight shape medians ranged from 1.000× to 1.016×. The 160 separate process-first measurements charged explicit LIBXSMM setup and generation plus the first multiply. All methods performed ordinary dynamic loading before that clock began. The CPU ran one caller thread. Guarded normalized timing-fixture checks covered 98,134,032 cells with a maximum absolute difference from FP64 of 0.00002186.

The two KleidiAI implementations had separately passed 39 fixtures and 122,382 full-output FP64 comparisons each, including true cropped strides and guard tests. Their immutable upstream revision is `ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1` from Arm's [KleidiAI repository](https://github.com/ARM-software/kleidiai/tree/ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1). An optional MOP4 path exceeded the tested M5 Max's instruction support, as recorded in the separate feature probe. The results cover one machine, eight selected FP32 products, seven external library families and this single-thread protocol.

The immutable inputs, raw observations, checksums, scoring code and provider build and correctness receipts are included. Run `python3 score.py .` after extracting the archive to reproduce all ratios. A native replay on supported Apple silicon uses `python3 native-replay.py RELEASE_REPO LIBXSMM_CHECKOUT PEERS.csv NEW_DIRECTORY`. The peer CSV uses `method,path` for all ordinary external adapters including KleidiAI-fixed and KleidiAI-elastic. Build the two qualified KleidiAI variants with `python3 provider/build.py PINNED_UPSTREAM NEW_PROVIDER_BUILD` and execute `python3 provider/qualify.py NEW_PROVIDER_BUILD` as documented by the qualification script.
