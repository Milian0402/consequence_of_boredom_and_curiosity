# Transfer of the 960 child cutoff to fresh strided layouts

The five-shape cutoff experiment directly confirmed the routing mechanism and
improved COB on three affected layouts on both tested CPUs. Its 960 cutoff
reversed both previous hybrid losses on Neoverse N2. Intel oneMKL remained
faster on the Xeon.

The three layouts in `hk-parent.csv` were selected before seeing that run's
results. Its SHA-256 is
`707584c88806ea83b13cc0b5e63815a967751a099cc49b0b27e8fdba413b386b`.
The two transfer cases have child harmonic means 985.63 and 1010.58. They
use different aspect ratios, odd child dimensions and padding. The third
case has harmonic mean 1037.94. Both old and new code use fused Strassen
there. A separate trace must show the new helper ran 15, 15 and 0 times.

Time the uninstrumented public entry, including packing and workspace, and
the incumbent at revision `9df006c`. Compare the primary candidate with the
fastest direct library, both existing vendor-backed Winograd hybrids and the
independently authored FastMMW routines at one and two levels. Include any
copies and padding in the FastMMW call. Verify one-thread execution, every
logical output against FP64 on random, cancellation and scaled inputs, and
the row and allocation guards. Report medians and all six paired observations
per comparison on each CPU. Count only the first two cases as evidence for
the change to the cutoff.
