# Single-core Arm SGEMM on Neoverse-N2

The candidate that streams 16 columns of B at a time **missed its registered improvement gate**. On eight fresh layouts its geometric-mean paired speedup over the tuned public COB incumbent was **1.045×**. It won 7/8 paired medians. Its weakest ratio was 0.999×, and the separate seed-block geometric means were 1.043× and 1.046×.

Against the fastest admitted independent one-shot interface per fresh layout, the candidate won 4/8 warm paired medians. It won 6/8 when peers were selected independently using first-in-process time. Each first-time comparison uses the same six fresh worker processes as its warm counterpart, with one complete initial call per process after library initialization. The retained-plan field below also includes warmed LIBXSMM and oneDNN.

| Layout | M × N × K | Candidate warm, µs | Incumbent / candidate | Best one-shot (paired ratio, wins) | Including retained (paired ratio) | First-call one-shot (paired ratio) |
|---|---|---:|---:|---|---|---|
| repeat-win-ragged | 137 × 2179 × 347 | 4168.6 | 1.055× | openblas (0.992×, 0/6) | openblas (0.992×) | openblas (1.015×) |
| repeat-intel-gap | 79 × 1793 × 223 | 1326.8 | 1.001× | blasfeo (0.987×, 0/6) | blasfeo (0.987×) | armpl (1.188×) |
| fresh-short-tail | 28 × 1153 × 107 | 159.7 | 1.123× | libxsmm-dispatch (0.913×, 0/6) | libxsmm-retained (0.879×) | blasfeo (0.954×) |
| fresh-thin-odd | 47 × 2051 × 153 | 630.5 | 1.025× | openblas (1.014×, 6/6) | openblas (1.014×) | openblas (1.082×) |
| fresh-exact | 64 × 2304 × 224 | 1362.5 | 0.999× | blasfeo (0.997×, 0/6) | blasfeo (0.997×) | armpl (1.252×) |
| fresh-minus-one | 75 × 2815 × 263 | 2325.2 | 1.061× | blasfeo (0.991×, 0/6) | blasfeo (0.991×) | armpl (1.177×) |
| fresh-middle-plus-one | 99 × 1537 × 311 | 1881.6 | 1.094× | blasfeo (1.019×, 6/6) | blasfeo (1.019×) | armpl (1.196×) |
| fresh-long-depth | 118 × 3329 × 359 | 5602.6 | 1.038× | armpl (1.021×, 6/6) | armpl (1.021×) | armpl (1.057×) |
| fresh-upper-irregular | 143 × 2473 × 229 | 3241.6 | 1.011× | armpl (1.002×, 5/6) | armpl (1.002×) | armpl (1.062×) |
| fresh-near-row-limit | 157 × 3889 × 307 | 7582.6 | 1.019× | openblas (0.987×, 0/6) | openblas (0.987×) | openblas (0.999×) |
| control-below-width | 61 × 983 × 177 | 473.7 | 1.000× | openblas (0.945×, 0/6) | openblas (0.945×) | openblas (0.724×) |
| control-above-rows | 161 × 2631 × 283 | 5057.9 | 1.000× | openblas (0.951×, 0/6) | openblas (0.951×) | openblas (0.917×) |

Each ratio is the median of six matched peer-to-candidate time ratios. Values above one favor the candidate. Each field selects its strongest interface by the lowest arithmetic median of six times for that timing contract. The two repeats and two bypass controls are shown separately from the eight-case acceptance measure.

The candidate and incumbent used ordinary uninstrumented executables linked to their respective installed CMake archives. The candidate ran on all ten eligible layouts, and both controls took the incumbent path. The controls matched the incumbent bit-for-bit in all four accuracy fixtures. Across 105 admitted method/layout combinations, workers checked 254,995,700 outputs against FP64 with maximum absolute error 2.66741654e-05. Every successful worker checked actual strides and base residues, protected both inputs read-only, checked the surrounding C parent, and disallowed new tasks using seccomp.

All nine methods qualified on all eight fresh cases. The three exclusions were blasfeo on repeat-win-ragged, blasfeo on control-below-width, blasfeo on control-above-rows. These workers exited with signal 11 before admission. The exit status identifies a worker fault. Its cause within the adapter or its dependencies remains undetermined. The field comparisons for those non-fresh cases use the remaining qualified methods.

Run `python3 verify.py raw` to reproduce the gates and all matched arrays. The original ZIP and every member checksum are identified in provenance.json. [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45703366648) · [Frozen protocol and implementation](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/d79902f7602b7890bdbe408b09b42927ed0ec7c3/tools/world-rank/arm-wide) · [Public source baseline](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6)
