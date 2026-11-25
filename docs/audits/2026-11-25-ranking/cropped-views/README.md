# Cropped matrix views on Intel(R) Xeon(R) 6975P-C

The installed generic COB library with streamed B panels retained **0 of its two previously reported direct-library wins** under the registered transfer rule on this single core. It beat the fastest measured independent interface by median on **0/3** previously unmeasured cropped geometries. Each geometry had an aligned companion with identical logical inputs.

| M × N × K | Case | Fastest peer on crop | Aligned peer / COB | Cropped peer / COB | Cropped pairs won | Default / COB on crop |
|---|---|---|---:|---:|---:|---:|
| 108 × 1728 × 320 | repeat-win-deeper | mkl-jit | 1.038× | 0.928× | 0/6 | 1.296× |
| 137 × 2179 × 347 | repeat-win-ragged | mkl-jit | 1.067× | 0.936× | 0/6 | 1.269× |
| 49 × 2861 × 157 | repeat-loss-thin | mkl | 0.897× | 0.867× | 0/6 | 1.409× |
| 43 × 2291 × 137 | fresh-short | mkl | 0.991× | 0.903× | 0/6 | 1.471× |
| 87 × 2809 × 239 | fresh-mid | mkl-jit | 0.932× | 0.853× | 0/6 | 1.233× |
| 127 × 1847 × 281 | fresh-upper | mkl-jit | 1.018× | 0.933× | 0/6 | 1.275× |

Across all six geometries, COB’s geometric-mean cropped/aligned time ratio was 1.005×. The corresponding ratios were 0.910× for oneMKL and 0.913× for its direct/JIT-capable interface. On the two repeated wins, the oneMKL interfaces improved by roughly 10% and 13% respectively on the cropped inputs, while COB’s times changed by around 1%.

Ratios are medians of the six matching peer-to-COB time ratios. Values greater than one favor COB. Each layout selects its fastest independent interface by median whole-call time among OpenBLAS, oneMKL, its separate sequential direct/JIT-capable interface, and AMD AOCL. Transfer requires an outright COB median win and at least four winning pairs in **both** the aligned and cropped layouts. The two prior wins came from an earlier installed-release run; the companion here gives the same-host baseline. [Previous release acceptance](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45695245069)

All six cropped layouts and all 36 matched warm batches favored enabled COB over its own default. All six cropped medians and all 36 matched warm batches favored the fastest independent interface over enabled COB. The inputs used independent row strides and three distinct nonzero base-address residues modulo 64 bytes for every cropped layout. A and B were read-only. Every untouched part of C was checked, including rows and columns outside the view. Four full-reference fixtures per method and layout and the six timing batches independently passed the 0.002 absolute FP64 limit. The worst observed error was 2.38291975e-05 across 168,392,140 output checks. Each of the 504 warm batches measured 16 complete calls on one pinned CPU; the separate first calls are in raw.csv.

The supplementary packed-A challenger, the default release, and all peer timings appear in the packet. The benchmark changes both row strides and offsets together. A measured difference therefore describes the complete cropped layout. The results support conclusions for this CPU, set of implementations and tested shapes.

Run `python3 verify.py raw` in the packet to recalculate all medians, paired ratios, kernel route checks, guard counts, and transfer decisions. The included provenance records each original artifact member digest and the full GitHub ZIP digest. [Public release](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6) · [Benchmark source and protocol](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/bce9e5367a2b1b414207460f635ac498663b213f/tools/world-rank/views) · [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45699142561)
