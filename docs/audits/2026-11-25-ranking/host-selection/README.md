# Choosing a schedule on each CPU

Two prior matrix layouts were used to choose among the existing aligned Winograd path, signed updates using the same library, and the direct library. The rule kept the existing path on both CPUs. It produced no incremental speedup on the next three layouts. [Rule frozen before timing data were inspected](protocol.md)

| CPU | Chosen path | Timed calibration work | Paired speed relative to fastest independent entry on deep / wide / compact holdouts |
|---|---|---:|---:|
| Intel Xeon 6975P-C | Released COB with oneMKL | 6.66 s | 0.986× / 1.010× / 1.071× |
| Arm Neoverse N2 | Released COB with OpenBLAS | 24.90 s | 1.055× / 1.043× / 1.004× |

The two older Intel layouts disagreed on whether calling oneMKL directly was faster, and the signed path gained less than the required 1.5% geometric mean on both CPUs. Calibration cost counts 36 timed warm multiplications on each host. It excludes setup and validation. The three follow-ups reuse the recorded calls for the chosen path. [Intel result](x86_64.json) · [Arm result](aarch64.json) · [Measurements and qualification](../signed-winograd/results-linux/README.md)

To reproduce a host decision, download its artifact from the [native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015), extract it, and run the following command with the directory containing that host's signed-score.json and timings.csv.

```sh
python3 -B select.py /path/to/extracted/signed-leaves
```
