# Changing matrices reversed COB’s M5 Max lead over the external field

The shipped compact option still improved on its own predecessor when the data changed. With every matrix page locked in memory, COB averaged **1.051× its predecessor** across the original eight shapes. Against the fastest independent method per shape, **the hot-buffer result was 1.026× and the rotating-buffer result was 0.950×**. The separate prefaulted run gave 1.013× and 0.936×, respectively. The frozen rule for extending the hot ranking to changing operands failed in both runs.

Each measured batch made the same number of whole public calls. The hot condition repeatedly used one matrix triple. The rotating condition supplied distinct A, B and C addresses and distinct input values on every call, traversing all 22–121 triples before moving to the next library. Matrices did recur in later sweeps. All methods followed the same round-specific permutation and used the same output addresses. Input generation, reference calculation and dataset allocation occurred before timing. Per-call workspace, packing and output costs remained inside the ordinary public calls. Prepared LIBXSMM reused its generated code and received the correct pointers for each new matrix.

## Results with locked matrix pages

All times are whole-call wall-clock microseconds. The ratio compares the external leader’s latency to COB in matched batches; above one favors COB. Pool bytes are the summed logical FP32 sizes of all A, B and C matrices, excluding layout padding, guard space, code, references and library scratch.

| M × N × K | Triples | Pool bytes | COB hot µs | COB rotating µs | Fastest rotating competitor | COB / field, hot | COB / field, rotating |
|---|---:|---:|---:|---:|---|---:|---:|
| 24 × 1088 × 280 | 100 | 134,988,800 | 14.1 | 38.8 | MaxMulSK | 0.995× | 0.841× |
| 27 × 2240 × 96 | 121 | 134,606,208 | 11.2 | 24.1 | MpGEMM | 1.073× | 1.033× |
| 29 × 2624 × 216 | 52 | 135,021,952 | 37.2 | 93.3 | MpGEMM | 1.066× | 0.924× |
| 32 × 3904 × 368 | 22 | 138,457,088 | 108.0 | 210.3 | MpGEMM | 0.998× | 1.013× |
| 25 × 2287 × 115 | 104 | 134,390,880 | 13.1 | 33.2 | KleidiAI-elastic | 1.102× | 0.939× |
| 26 × 3487 × 307 | 29 | 135,621,748 | 56.7 | 118.2 | MaxMulSK | 1.099× | 1.007× |
| 30 × 1043 × 275 | 103 | 134,462,380 | 21.2 | 43.7 | MaxMulSK | 0.823× | 0.868× |
| 31 × 3135 × 187 | 49 | 135,088,492 | 46.8 | 84.4 | MpGEMM | 1.081× | 0.994× |
| 24 × 1031 × 383 (extra) | 79 | 135,503,644 | 33.9 | 54.1 | KleidiAI-fixed | 0.790× | 0.930× |

The extra 24 × 1031 × 383 boundary challenge was frozen separately and excluded from the eight-shape averages. Its rotating ratio was 1.040× in the prefaulted run and 0.930× in the wired run. On the original 24 × 1088 × 280 layout, wired COB took 14.1 µs hot and 38.8 µs rotating, with prepared LIBXSMM and MaxMulSK leading the respective external fields. On 29 × 2624 × 216, COB switched from a 1.066× hot advantage against MpGEMM to 0.924× under rotation.

## Residency, accuracy and ranking uncertainty

The pools span 134,390,880–138,457,088 logical bytes. macOS reported 16,384-byte pages, 128-byte cache lines, 16 MiB L2 per group of six Super cores and 8 MiB L2 for the Performance tier. The logical ring exceeded the largest of those reported L2 sizes by at least eightfold. The process ran user-initiated QoS and observed one Mach task thread at timing boundaries; macOS controlled core placement. System-level cache size, cache hits and traffic reaching DRAM were not measured.

In the wired run, `mlock` covered every A, B and C mapping. All matrix pages were resident in every sampled `mincore` reading. There were zero major and 9,337 minor process faults across 36 timing epochs, including library/runtime activity outside the protected input arenas. The first run prefaulted all matrices; one intermediate observation found two of 8,246 pages absent in one geometry/seed. These are warmed-code, memory-resident observations, distinct from timing process startup or incurring page faults to load a data file.

Each run recorded 4,356 timed batches and 318,956 complete multiplies across two seeds, with eight external configurations from seven library families. The independent scorer checked 773,295,534 FP64 cells per run, covering every output of every distinct slot for every registered method. The maximum absolute error was 0.0000312905 on these normalized data. All inputs were write-protected. All observed output guards and post-batch FP64 probes passed. The integrated arithmetic’s separate scaled-input limitations remain documented in the compact-source report.

When the fastest external was selected independently for each seed, the eight-shape rotating ratios were 0.932× and 0.931× prefaulted, then 0.945× and 0.932× with wired pages. Same-function aliases exposed timing noise. The wired rotating per-shape median A/B ratios ranged from 0.934× to 1.052×, with three of eight outside ±3%. The prefaulted run had five outside that band. Small per-shape differences should be interpreted in light of these controls. Neither run met the frozen requirement of winning separately on both seeds with all eight null checks within 3%.

The source and provider binaries matched the earlier two-panel study. The archive contains the same-hardware, same-input hot comparison alongside the rotating data, per-shape scores for wall and caller CPU clocks, residency and accuracy receipts, native source and checksums. Recompute the two published scores with `python3 score.py prefaulted --output reproduced-hot-ring` and `python3 score.py wired --output reproduced-wired`. Native reproduction on a supported Apple machine uses `python3 reproduce.py RELEASE_REPO LIBXSMM_CHECKOUT PEERS.csv NEW_DIRECTORY`. The CSV uses `method,path` for Accelerate, OpenBLAS, ArmPL, MaxMulSK, MpGEMM, KleidiAI-fixed and KleidiAI-elastic. The script builds both public opt-in versions from the frozen parent commit and patch. Use `--only wired` or `--only prefaulted` to run a single mode pair.
