# One fused Strassen level inside a Hopcroft–Kerr subproduct

Two large rectangles lost to the Winograd hybrids on both server CPUs in
run 45657809673. The harmonic sizes of their Hopcroft–Kerr subproducts are
960 and 1021.33. Both fall below the existing fused Strassen cutoff of 1024.
The larger rectangle that won on both machines has subproducts of size 1216.

The primary experiment sets the cutoff to 960 inside eligible Hopcroft–Kerr
subproducts. The exploratory arm uses 896. The application-level cutoff stays
at 1024. The five cases in `hk-child.csv` comprise the two earlier losses,
a fresh case on each side of 960 below 1024, and a fresh control above 1024.

Compare complete public calls against the previous public HK implementation,
the two competing Winograd hybrids, and the fastest independent library
on each CPU and shape. Record all six paired observations per comparison,
both seeds, and every result on the control. Separate instrumented builds
record how many of the fifteen subproducts actually used the fused kernel.
Random, cancellation and scaled inputs must pass the existing full-output
FP64 and buffer-guard checks on every layout. Every timed worker uses one
thread pinned to one logical CPU.

Report the primary candidate's result for each previous loss and whether
the faster hybrid still wins there. Evaluate transfer on the fresh shapes
separately. At the 952-child layout only the exploratory cutoff changes the
route. At 1088 all three arms should select the same path. Interpret any
speed differences there alongside repeat variability. Any cutoff selected
after inspecting timings requires a new panel to confirm its benefit.
