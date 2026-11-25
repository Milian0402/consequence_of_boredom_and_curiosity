# Standalone wide FP32 multiplication on Neoverse-N2

The new Arm kernel packs 16 columns of B through the full reduction. It **missed the registered improvement gate**. Its geometric-mean speedups over the otherwise identical tuned COB library were **1.045× and 1.032×** across eight fresh shapes on two separately allocated runners. It won 7/8 and 6/8 fresh paired medians. Both used one pinned logical CPU.

| Native run | Seed-block speedups over incumbent | Worst fresh ratio | Warm one-shot wins | Warm wins including retained plans | First-in-process one-shot wins |
|---|---|---:|---:|---:|---:|
| 1 | 1.043×, 1.046× | 0.999× | 4/8 | 4/8 | 6/8 |
| 2 | 1.028×, 1.036× | 0.981× | 2/8 | 2/8 | 6/8 |

Each field separately selects the fastest qualified external interface by the lowest arithmetic median on that layout. A field win in the table uses the median of six paired competitor/candidate times. The direct field comprises OpenBLAS, ArmPL, BLASFEO, per-call LIBXSMM dispatch and strict oneDNN per-call primitive creation. The retained field additionally permits a cached oneDNN primitive and a held LIBXSMM kernel. First-in-process comparisons use six initialized workers with one initial complete product each. The warm measurements use sixteen complete calls in each matching worker.

| Fresh M × N × K | Incumbent / candidate, runs 1 and 2 | Fastest one-shot, run 1 (ratio, pairs won) | Fastest one-shot, run 2 (ratio, pairs won) |
|---|---:|---|---|
| 28 × 1153 × 107 | 1.123×, 1.115× | libxsmm-dispatch (0.913×, 0/6) | libxsmm-dispatch (0.892×, 0/6) |
| 47 × 2051 × 153 | 1.025×, 1.024× | openblas (1.014×, 6/6) | blasfeo (1.009×, 6/6) |
| 64 × 2304 × 224 | 0.999×, 0.991× | blasfeo (0.997×, 0/6) | blasfeo (0.993×, 0/6) |
| 75 × 2815 × 263 | 1.061×, 1.055× | blasfeo (0.991×, 0/6) | blasfeo (0.987×, 0/6) |
| 99 × 1537 × 311 | 1.094×, 1.092× | blasfeo (1.019×, 6/6) | blasfeo (1.022×, 6/6) |
| 118 × 3329 × 359 | 1.038×, 1.007× | armpl (1.021×, 6/6) | blasfeo (0.998×, 2/6) |
| 143 × 2473 × 229 | 1.011×, 1.001× | armpl (1.002×, 5/6) | armpl (0.997×, 1/6) |
| 157 × 3889 × 307 | 1.019×, 0.981× | openblas (0.987×, 0/6) | openblas (0.944×, 0/6) |

The candidate won every warm pair against all seven external interfaces on both runners at 47 × 2051 × 153, 99 × 1537 × 311. These are shape-specific replicated leads.

Values above one favor the candidate. The gate required at least 1.05× geometric-mean improvement over the incumbent, gains in both seed blocks, six winning fresh medians and no fresh layout below 0.95×. The two bypass controls matched the incumbent bit-for-bit in all four fixtures on both runners. Separate witnesses recorded all ten eligible routes and both bypasses per runner. Both ordinary executables on both runners were killed by SIGSYS during explicit attempts to create another task, and the numerical workers enforced the same filter. They checked 509,991,400 total logical output values against FP64, with a maximum absolute error of 2.66741654e-05 under the 0.002 limit. All nine methods qualified on all eight fresh layouts. On both hosts, the BLASFEO adapter process exited with signal 11 on the 137 × 2179 × 347 anchor and on both bypass controls. Those three non-fresh external comparisons have a smaller field. The logs record the failures without assigning a cause to the upstream library.

Two earlier hosted attempts stopped during the incumbent test suite and produced zero timing observations. The forced-Strassen tests expected their own allocation path at 64 and 128 rows. The tuned N2 small kernel correctly dispatched first without allocating. A two-line test-only correction checked this path using the existing small-kernel fixture, which exercises both public output and forced Strassen. Native Linux Arm clean builds confirmed byte-identical installed arithmetic archives before and after the correction. The two pre-timing archives, their checksums and test failures are listed in no-data/receipts.json.

Each native-N directory includes actual shape/stride/offset data, per-call metadata, first/warm times, counterexamples, numerical checks, eligibility and its ordinary/trace build hashes. Run `python3 verify.py raw` within either directory to reproduce all reported gates, ratios and field rankings.

[Native workflow](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45703366648) · [Frozen benchmark and source patch](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/d79902f7602b7890bdbe408b09b42927ed0ec7c3/tools/world-rank/arm-wide) · [Public COB baseline](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6)
