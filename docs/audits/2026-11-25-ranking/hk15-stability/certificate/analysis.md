# Exact shortcut for the HK15 magnitude guard

The benchmark patch applies to `85a007747ff54241f47c4754216e0a65748f2847`. It changes only `src/cob_hk15.inc`. The original classifier remains the fallback.

For each operand the shortcut nominates a power of two U in the original legal interval from 2⁻⁴⁸ to 2⁴⁸. It then establishes three inequalities.

- Every logical magnitude is either zero or within [2⁻⁴⁸, U], including magnitudes outside the sampled windows. Integer magnitude checks reject NaNs, infinities and otherwise disallowed values.
- Every logical row and column contains a magnitude at least U/4. A witness in any position suffices. The shortcut searches short prefixes and returns to the original classifier if it cannot find such a witness.
- Every A row and B column has at least ceil(3k/4) entries of magnitude at least U/8. The count uses all k positions.

The true maximum M is positive and bounded by U. Each row or column maximum is at least U/4 ≥ M/4. Likewise, every entry counted at U/8 also reaches M/8, so the original support test must pass. These implications allow a positive shortcut to enter the unchanged signed-coherence calculation. A failed shortcut always executes the original classifier. Padding never participates. An input with a late exception can therefore cost an additional scan.

The floating-point GEMM kernels, fallback arithmetic and coherence thresholds are identical. AppleClang's public `cob_sgemm_rowmajor` retains its 432-byte local frame plus saved registers. Only the HK shape's classification routine probes its roughly 15 KiB scratch.

The ordinary benchmark matrix is the installed public SGEMM, built with Release, ragged HK15, portable and recursive Strassen on the M5 Max. The separate timing-input driver performs only classification and emits no times.
