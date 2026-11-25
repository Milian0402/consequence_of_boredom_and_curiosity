# Replaying the ragged HK15 comparison

This benchmark compares a research-only HK15 ragged-edge patch with its tuned standalone release, both at production commit `22d925ed5a31f7246eb2269675e9a3e85d502410`. The ordinary and diagnostic executables use separate builds. The diagnostics check the actual 15-product schedule and disjoint edge operations. The installed competition includes public COB with two levels of Winograd using OpenBLAS and the platform vendor, alongside the independent single-thread field that qualifies on the host. [Frozen panel and acceptance criteria](protocol.md)

On a clean Linux x86-64 or Arm64 runner, use the toolchain prerequisites and dependency steps in `.github/workflows/native-gemm.yml`, job `ragged-hk`. The benchmark needs a separately checked-out, unmodified production source, a vendor directory containing the verified serial OpenBLAS FP32 and FP64 plugin and the platform-specific dependencies. Run this command from the benchmark repository root.

```sh
bash tools/world-rank/ragged-hk/run.sh "$production_source" "$vendor_directory" "$result_directory"
```

The study writes full output error and memory checks to `accuracy.csv`. `timings.csv` has one cold-after-initialization call and one immediately following warm call for each timed process. `goldens.csv` identifies the 20 serial OpenBLAS DGEMM reference files with SHA-256 digests and independent scalar FP64 spot checks. Every process checks both logical input digests, all live FP32 output cells, cropped destination padding and one-thread CPU affinity. `routes.csv` links the candidate, incumbent, released vendor-assisted configurations and FastMMW operations to separately checked runtime witnesses. `sources/methods.csv` and `qualified.csv` document host availability and case-level exclusions. [Arm and x86 Linux workflow](../../../.github/workflows/native-gemm.yml)
