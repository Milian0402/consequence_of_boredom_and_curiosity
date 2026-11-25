# Can two calibration shapes choose the next three?

This supplemental analysis uses the five layouts in the [signed-update native
run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45689461015).
The rule was written after the jobs started, before inspecting their timing data.
Its inputs are verified one-thread, full-output-qualified complete calls. It
evaluates a choice among measured paths. A production dispatcher implementing
the choice needs its own whole-call test.

Use the same rule on each processor. The provider is the sequential oneMKL
interface on x86-64 and OpenBLAS on Arm64. The three choices are the released
aligned Winograd algorithm, the signed-update variant with the same provider,
and a direct call to that provider. All three must qualify for the two
calibration shapes. Use `deep-odd-prior` and `wide-odd-prior` exclusively
for selection. For each alternative compute the median of the six paired
release-time/alternative-time ratios per layout. Require both medians to be
at least 0.995 and their geometric mean to exceed 1.015. Choose the alternative
with the greatest qualifying geometric mean, with the release as fallback.
For exact ties between alternatives prefer the direct library.

Lock that choice for both odd follow-ups and the compact control. Where the
choice fails qualification, score a failure. Show wins, losses and geometric
means against the release and the fastest independent one-shot entry. A choice
of direct BLAS counts as the same implementation when comparing it to itself.
For a follow-up public-dispatch trial, require a geometric mean exceeding 1.02
over the release across the three follow-ups and no layout below 0.98. Report
the calibration cost separately. This is a test of transfer across shapes,
with the two CPU outcomes reported separately.
