# Rational rank-48 and rank-49 schedules

The 4 × 4 block experiment uses the 201-balanced rank-48 and 156-operation
rank-49 standard-basis programs from the same PLinOpt revision. Both use
identical FP32 transform code generation, direct views for unmodified inputs,
temporary storage for the remaining forms, a serial vendor GEMM for each
product, and the same uneven-edge handling. Every allocation, transform and
combination is inside the timed call. The leaves are sequential oneMKL on
x86-64 and OpenBLAS on Arm64. [Numerical schedules](NOTICE.md)

The primary comparison is rank 48 against rank 49 for each of the four rows in
`panel.csv`. Direct vendor results give the external comparison. The first
odd layout appeared in the earlier release study. This is a screen of specific
published schedules and their implementation. Prioritize an independent
confirmation only if the 48-product algorithm passes every numerical, guard,
route and thread check and beats the 49-product algorithm by paired median on
at least three layouts on that processor. A lead over a direct library also
requires at least five of six paired wins on that layout. Record failures
and exclude them from speed claims.

Check all 4,096 bilinear coefficients exactly for each schedule. On each CPU
and layout require the expected number of core and edge leaf products and
an exact analytic full-output and padding test. The common worker checks two
random seeds, cancellation and scaled inputs against FP64, with absolute error
at most 0.002, intact guards and inputs, one thread and one pinned CPU.
Measure three counterbalanced warm rounds for each seed and keep the first
unbatched calls separate. No COB release defaults depend on this screen.
