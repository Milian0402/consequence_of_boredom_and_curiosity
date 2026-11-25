# Padding rows before a direct vendor SGEMM call

On two independent Xeon 6975P-C runners, rounding the output row count to a multiple of eight made all five genuinely padded fresh layouts slower than the same sequential oneMKL call. The geometric means were 0.886× and 0.887× direct oneMKL. Every padded comparison lost in both replicas (60 paired observations). On the three other fresh layouts the adapter passed the original arguments through to oneMKL. A raw timing lead on a pass-through case measures that same underlying implementation.

An additional runner assigned after the two Intel runs was an AMD EPYC 9V45. On the five actually padded fresh layouts the geometric mean was 0.954× against direct AOCL. All five paired medians favored direct AOCL, with 29 individual wins out of 30. The proposed padding rule missed the advancement criterion on both CPU families. [AMD paired observations](attempt-2-job-139503993618.json)

The adapter selected the native provider by x86 CPU vendor, copied A and allocated a strided scratch C only on the padded path, and included all copies and allocation in the timing. The untimed witness intercepted the underlying provider call, checked each actual dimension and stride, and checked every output against an exact integer result. All timed methods also passed the full four-fixture FP64 checks. [Registered decision rule](protocol.md) · [Per-layout records and complete native field](../packed-a-x86/README.md)

Run the following command from this directory, substituting either recorded job directory, to reproduce the paired ratios and decision.

```sh
python3 -B score.py ../packed-a-x86/attempt-1-job-139500093897/raw
```
