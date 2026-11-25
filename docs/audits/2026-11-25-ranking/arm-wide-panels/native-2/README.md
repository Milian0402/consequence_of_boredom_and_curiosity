# Single-core Arm SGEMM on Neoverse-N2

The candidate that streams 16 columns of B at a time **missed its registered improvement gate**. On eight fresh layouts its geometric-mean paired speedup over the tuned public COB incumbent was **1.032×**. It won 6/8 paired medians. Its weakest ratio was 0.981×, and the separate seed-block geometric means were 1.028× and 1.036×.

Against the fastest admitted independent one-shot interface per fresh layout, the candidate won 2/8 warm paired medians. It won 6/8 when peers were selected independently using first-in-process time. Each first-time comparison uses the same six fresh worker processes as its warm counterpart, with one complete initial call per process after library initialization. The retained-plan field below also includes warmed LIBXSMM and oneDNN.

| Layout | M × N × K | Candidate warm, µs | Incumbent / candidate | Best one-shot (paired ratio, wins) | Including retained (paired ratio) | First-call one-shot (paired ratio) |
|---|---|---:|---:|---|---|---|
| repeat-win-ragged | 137 × 2179 × 347 | 4246.6 | 1.039× | openblas (0.976×, 0/6) | openblas (0.976×) | openblas (1.020×) |
| repeat-intel-gap | 79 × 1793 × 223 | 1327.6 | 0.999× | blasfeo (0.987×, 0/6) | blasfeo (0.987×) | openblas (1.110×) |
| fresh-short-tail | 28 × 1153 × 107 | 160.6 | 1.115× | libxsmm-dispatch (0.892×, 0/6) | libxsmm-dispatch (0.892×) | blasfeo (0.941×) |
| fresh-thin-odd | 47 × 2051 × 153 | 630.9 | 1.024× | blasfeo (1.009×, 6/6) | blasfeo (1.009×) | openblas (1.097×) |
| fresh-exact | 64 × 2304 × 224 | 1366.3 | 0.991× | blasfeo (0.993×, 0/6) | blasfeo (0.993×) | openblas (1.139×) |
| fresh-minus-one | 75 × 2815 × 263 | 2334.5 | 1.055× | blasfeo (0.987×, 0/6) | blasfeo (0.987×) | openblas (1.076×) |
| fresh-middle-plus-one | 99 × 1537 × 311 | 1883.5 | 1.092× | blasfeo (1.022×, 6/6) | blasfeo (1.022×) | openblas (1.124×) |
| fresh-long-depth | 118 × 3329 × 359 | 5830.1 | 1.007× | blasfeo (0.998×, 2/6) | blasfeo (0.998×) | armpl (1.054×) |
| fresh-upper-irregular | 143 × 2473 × 229 | 3272.2 | 1.001× | armpl (0.997×, 1/6) | armpl (0.997×) | armpl (1.065×) |
| fresh-near-row-limit | 157 × 3889 × 307 | 7981.7 | 0.981× | openblas (0.944×, 0/6) | openblas (0.944×) | openblas (0.994×) |
| control-below-width | 61 × 983 × 177 | 472.2 | 1.001× | openblas (0.949×, 0/6) | openblas (0.949×) | openblas (0.716×) |
| control-above-rows | 161 × 2631 × 283 | 5079.4 | 0.997× | openblas (0.944×, 0/6) | openblas (0.944×) | openblas (0.896×) |

Each ratio is the median of six matched peer-to-candidate time ratios. Values above one favor the candidate. Each field selects its strongest interface by the lowest arithmetic median of six times for that timing contract. The two repeats and two bypass controls are shown separately from the eight-case acceptance measure.

The candidate and incumbent used ordinary uninstrumented executables linked to their respective installed CMake archives. The candidate ran on all ten eligible layouts, and both controls took the incumbent path. The controls matched the incumbent bit-for-bit in all four accuracy fixtures. Across 105 admitted method/layout combinations, workers checked 254,995,700 outputs against FP64 with maximum absolute error 2.66741654e-05. Every successful worker checked actual strides and base residues, protected both inputs read-only, checked the surrounding C parent, and disallowed new tasks using seccomp.

All nine methods qualified on all eight fresh cases. The three exclusions were blasfeo on repeat-win-ragged, blasfeo on control-below-width, blasfeo on control-above-rows. These workers exited with signal 11 before admission. The exit status identifies a worker fault. Its cause within the adapter or its dependencies remains undetermined. The field comparisons for those non-fresh cases use the remaining qualified methods.

Run `python3 verify.py raw` to reproduce the gates and all matched arrays. The original ZIP and every member checksum are identified in provenance.json. [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45703366648) · [Frozen protocol and implementation](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/d79902f7602b7890bdbe408b09b42927ed0ec7c3/tools/world-rank/arm-wide) · [Public source baseline](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6)
