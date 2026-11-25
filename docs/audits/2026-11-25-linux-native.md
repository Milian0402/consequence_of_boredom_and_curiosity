# Single-thread FP32 GEMM on native Linux x86-64 and Arm64

GitHub's `ubuntu-24.04` runner reported Intel Xeon 6975P-C. Its `ubuntu-24.04-arm` runner reported Arm Neoverse-N2. Each process was pinned to one permitted logical CPU and checked for a single thread. GCC compiled both COB versions with the same native target. Comparisons used the distribution's one-thread OpenBLAS 0.3.26. Every reported ratio is a median of paired elapsed times through one-shot multiply entries, including packing and allocation. These results apply to the reported machines and shapes.

The first portable change improved the plain C fallback but left the Xeon far behind OpenBLAS. The second revision added explicit AVX-512 and AVX2/FMA tiles and a cache-blocked one-shot path, alongside an Arm NEON edge kernel. On five medium shapes the Xeon then measured 31.78–36.12× the speed of the preceding COB kernel. At 768 square the paired comparison had medians of 199.74 ms for the older code and 5.91 ms for the new code. On Arm, 383 × 385 × 389 improved 1.258×, and the streaming one-row path improved 10.7–11.6× on two widths. The initial one-row competitor used SGEMM. The final run used the equivalent transposed SGEMV. [Portable C run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45628670772) · [Native SIMD run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45629641726)

The third revision tested one Strassen level with ordinary native COB leaves on five eligible shapes. Six counterbalanced pairs were measured against each comparator per shape. The baseline was the native SIMD revision `5e40c1c`. All outputs from the baseline, candidate and OpenBLAS were compared with FP64 DGEMM from the exact FP32 inputs for two seeds. All passed the 0.002 maximum absolute error threshold. Candidate maximum error on the eligible products was 0.00041034. A separate 2048-square cancellation input and forced allocation-failure tests also passed. [Strassen and SGEMV run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45630540529)

| M × N × K | Xeon vs previous COB | Xeon vs OpenBLAS | N2 vs previous COB | N2 vs OpenBLAS |
| --- | ---: | ---: | ---: | ---: |
| 1536 square | 1.118× | 0.884× | 1.153× | 1.059× |
| 2048 square | 1.136× | 0.869× | 1.152× | 1.057× |
| 3072 square | 1.136× | 0.863× | 1.102× | 1.023× |
| 4096 square | 1.190× | 0.833× | 1.285× | 0.998× |
| 3072 × 3328 × 2944 | 1.132× | 0.844× | 1.105× | 1.006× |

The candidate won all 30 eligible pairs against previous COB on each architecture. At the off-route 1024 square and 2049 × 2112 × 1984 controls, it trailed OpenBLAS on both machines. The Xeon controls moved about 5% against previous COB despite using the classical path, so small ratios merit a fresh repeat. For 1 × 1536 × 1024 and 1 × 2049 × 1025 the final Xeon row kernel measured 0.996× and 1.005× versus OpenBLAS SGEMV. The Arm ratios were 1.013× and 1.193×. Larger portfolio and independent-machine comparisons remain open.
