# Reproducing a native benchmark

The [results index](../../docs/audits/2026-11-25-ranking/README.md) gives the
CPU, shapes, timing contract, numerical checks and offline replay command for
each study. Each recorded benchmark commit is part of this repository’s
history. Use the recorded commit for a native rerun because some studies build
experimental kernels from that checkout as well as a pinned release archive.

For example, the irregular Hopcroft–Kerr study ran at
`37ead44b9f3a144dada8a92b17bd447b138e6689`. From a full clone, create a
separate checkout with the following command.

```sh
git worktree add --detach ../cob-ragged-benchmark 37ead44b9f3a144dada8a92b17bd447b138e6689
```

The workflow records the native build and run commands for both Linux
architectures. It is triggered by pushes to `bench/native-linux-gemm`.
The `tools/world-rank/phase.txt` file in the benchmark checkout selects the
study. The explicit oneMKL handle study used
`0fcba28139f3d0f64492d5e9938f50d32280625d`. [Native workflow](../../.github/workflows/native-gemm.yml)
