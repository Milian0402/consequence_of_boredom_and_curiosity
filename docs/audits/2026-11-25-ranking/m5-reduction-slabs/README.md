# A 64-step SME reduction slab slowed both reused and changing M5 inputs

The experimental kernel lost on **all three predefined pilot layouts, on both seeds**, compared with the existing compact path at `69b239d6ab9b9a86425ef6874634e3a1952a7f73`. The shipped path retained its lead under both input contracts. The candidate achieved **0.801× incumbent throughput** with a reused matrix triple and **0.818×** while rotating through at least 128 MiB of distinct, VM-locked A/B/C data. It failed its prespecified pilot rule. The experimental patch remains outside production and the pilot spent no new holdout layouts.

For each direct 16×64 output tile, the incumbent consumes the full depth and writes each output once. The candidate groups four adjacent column tiles into a 256-column output band, sweeps 64 K values at a time, and revisits every output row in that band for each slab. A complete 64×256 logical B slab is 64 KiB, and up to 32 rows of C occupy 32 KiB per band. It retains the same full-call A packing and never packs B. Between slabs it stores partial FP32 accumulators to C and reloads them into ZA, preserving each element’s fused accumulation order. The change increases C traffic. No hardware counter separated the traffic cost from other effects.

| M × N × K | Input contract | Incumbent µs | Candidate µs | Speedup | Seed 1 / 2 | Same-function A/B |
|---|---|---:|---:|---:|---:|---:|
| 24 × 1088 × 280 | hot | 16.1 | 24.8 | 0.870× | 0.935× / 0.851× | 0.938× |
| 24 × 1088 × 280 | rotate | 34.5 | 41.5 | 0.820× | 0.816× / 0.831× | 0.982× |
| 29 × 2624 × 216 | hot | 24.1 | 33.3 | 0.737× | 0.712× / 0.802× | 1.093× |
| 29 × 2624 × 216 | rotate | 66.4 | 80.1 | 0.823× | 0.807× / 0.850× | 1.017× |
| 31 × 3135 × 187 | hot | 28.9 | 35.4 | 0.801× | 0.800× / 0.801× | 0.968× |
| 31 × 3135 × 187 | rotate | 71.5 | 93.6 | 0.810× | 0.774× / 0.836× | 0.995× |

The first two geometries had shown changing-data losses to the external field. The third was a predeclared compact control. Each ordinary public build received the same operands and output addresses. Two labels resolved to the identical incumbent function pointer. The schedule ran all six within-process orders three times per seed, with balanced hot/rotating epoch order. The geometric-mean caller-CPU ratios were 0.800× hot and 0.819× under rotation. Hot same-function nulls varied by more than 3% on each geometry, and all three rotating nulls were within 2%. Both candidate seeds were below one on every geometry and workload.

The harness checked 67,981,542 output cells against independent FP64 calculations and required the candidate’s FP32 results to match the incumbent bit-for-bit on each distinct input. Maximum absolute error on the normalized fixtures was 0.00002435. All 648 timed batches (43,416 multiplies) passed guard and post-timing spot checks. Each matrix arena was write-protected for inputs and fully resident at every `mincore` observation after `mlock`. Major process faults stayed at zero in all 12 timing epochs. The study measures warmed-code timing on one M5 Max, with no direct measurement of DRAM traffic.

Run `python3 score.py .` from the extracted archive to reproduce the ratios and pilot decision. The native replay is `python3 reproduce.py SOURCE_REPO LIBXSMM_CHECKOUT NEW_OUTPUT`. The scorer verifies the entire six-permutation schedule, resident pools, input and output checks, per-seed observations and declared promotion rule.
