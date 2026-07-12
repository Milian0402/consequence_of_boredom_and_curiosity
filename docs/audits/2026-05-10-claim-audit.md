# 2026-05-10 Claim Audit

> Historical snapshot. A July 2026 current-head re-audit found losses to
> current OpenBLAS and MIT-licensed MpGEMM. This file records what the May audit
> showed, but it no longer supports a current broad fastest claim.

## Finish Line

This pass audits the scoped finish line, not the unqualified research goal.

Audited claim:

COB is the fastest tested licensed/open-source implementation for
single-thread FP32 row-major SGEMM on Apple Silicon, in the routed shape ranges.

This is narrower than "fastest matrix multiplication software available." It
does not include multi-threaded GEMM, GPU kernels, non-FP32 types, transposed
operands, arbitrary `alpha`/`beta`, or shapes outside the routed suites.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Single-threaded matrix multiplication | README scope, `docs/claims.md`, benchmark environment pins BLAS thread variables to `1` for external CBLAS targets. | Covered for repo scope |
| FP32 row-major SGEMM | Public API and claim docs restrict the contract to row-major FP32 `C = A * B`. | Covered |
| Apple Silicon routed shape ranges | `tools/claim_audit.sh` ran square, medium, and skinny routed suites on Apple M5 Max. | Covered for audited suites |
| Correctness | `make test` passed 127 GEMM shapes after a clean rebuild. | Covered |
| Fresh current audit | `COB_AUDIT_REPEATS=11 COB_AUDIT_OUT_DIR=/private/tmp/cob-final-claim-audit-20260510 sh tools/claim_audit.sh`. | Covered |
| Licensed/open-source baseline boundary | `docs/claims.md` lists BLIS, OpenBLAS, BLASFEO, Eigen, Rust `matrixmultiply`, `coral-aarch64`, LIBXSMM, `tract-linalg`, and KleidiAI as tested baselines. | Covered by prior audits and docs, not all rerun today |
| Proprietary/unclear-license exclusions | Accelerate and MpGEMM are tracked separately in `docs/claims.md`. | Covered |
| Universal "fastest fastest" claim | Requires all available source-inspectable baselines and all relevant shapes/API contracts. | Not proven |

## Commands Run

```sh
make clean
make all
make test
COB_AUDIT_REPEATS=11 COB_AUDIT_OUT_DIR=/private/tmp/cob-final-claim-audit-20260510 sh tools/claim_audit.sh
git diff --check
```

Results:

- Clean build succeeded at commit `014557a`.
- Correctness passed: `all GEMM tests passed (127 shapes)`.
- Audit bundle: `/private/tmp/cob-final-claim-audit-20260510`.
- `git diff --check` passed.

## Audit Context

From `/private/tmp/cob-final-claim-audit-20260510/context.txt`:

- Commit: `014557afd1ac6ab417dc73359a1d536ed9eb406b`.
- Machine: `Mac17,6`.
- Kernel: Darwin `25.4.0`.
- CPU: `arm64`, 18 active CPUs.
- L1d: 64 KB.
- L2: 8 MB.
- Page size: 16 KB.
- Cache line: 128 B.
- SME and SME2 available.
- Audit repeats: 11.
- Suites: `square medium skinny`.

## Fresh Audit Findings

The May 10 audit compares COB public paths against the benchmark's available
non-COB baseline, Accelerate, and against COB's own packed variants.
Accelerate is proprietary, so Accelerate gaps do not invalidate the
licensed/open-source claim; they remain useful calibration.

Notable results:

- `skinny.one-shot`, `skinny.packed-B`, and `skinny.packed-AB` had no
  non-COB target gaps in the fresh audit.
- `square.packed-AB` had no non-COB target gaps.
- `medium.packed-AB` had one small Accelerate gap:
  `768x768x4096`, ratio `0.992279x`, gap `0.78%`.
- One-shot and packed-B still have several Accelerate gaps, especially medium
  pack-overhead-heavy cases. These remain outside the licensed/open-source
  baseline set.
- Several rows had high best-vs-median drops, so any small delta should be
  rechecked with paired or cold reruns before tuning.

The audit also shows that COB's own packed-AB path is often the fastest public
COB API when both operands can be reused. That is an API-contract distinction,
not a one-shot regression by itself.

## External Baseline State

The current licensed/open-source claim still relies on the previous external
baseline audits recorded in `docs/claims.md`, not a full May 10 rerun of every
external project. Those recorded baselines include:

- BLIS.
- OpenBLAS.
- BLASFEO.
- Eigen.
- Rust `matrixmultiply`.
- `coral-aarch64`.
- LIBXSMM exact `beta = 0`.
- `tract-linalg` comparable public paths.
- KleidiAI comparable public one-shot path.

The May 10 web scan did not change the claim boundary:

- KleidiAI remains a relevant open-source Arm baseline and should continue to
  be tracked on future audits.
- `sapphire-compute` advertises MIT licensing and Apple Silicon SGEMM
  performance, but the published package claim was not enough to add it to the
  audited source-inspectable baseline set in this pass.
- MpGEMM remains source-available calibration with unclear local licensing and
  is not folded into the licensed/open-source claim.

## Conclusion

The scoped release claim is presentable:

COB is fastest among the tested licensed/open-source baselines for
single-thread FP32 row-major SGEMM on Apple Silicon, in the routed shape ranges.

The broader "fastest fastest open-source matrix multiplication software
available" goal is not fully proven. The remaining path to broaden it is:

1. Rerun every external licensed/open-source baseline from a clean checkout on
   the same May 10 commit.
2. Add any newly discovered source-inspectable Apple Silicon SGEMM package to
   the audit.
3. Treat MpGEMM as in-scope only if its license becomes clear.
4. If MpGEMM becomes in-scope, write an original fixed-shape SME kernel for the
   remaining `m = 64` large-`K` gaps before claiming victory over it.
