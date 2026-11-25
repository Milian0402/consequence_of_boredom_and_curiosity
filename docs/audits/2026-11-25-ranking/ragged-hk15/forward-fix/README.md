# Forward fix for the static OpenBLAS proof

The measured script required its Linux executable to define the Fortran `sgemm_` entry point. The installed COB library calls the C interface `cblas_sgemm` and the serial-thread getter. OpenBLAS places the Fortran entry in a separate archive member, so normal static linking can omit it. Both OpenBLAS configurations passed their two CMake tests and installed in the hosted run. The logs ended before the script identified its failed predicate. The local reconstruction reproduced a failure of the Fortran-symbol check in both otherwise working configurations.

The proposed patch checks that GNU ld resolved each reference from the installed COB archive to a definition in the selected static OpenBLAS archive, then checks the ordinary executable’s defined functions. It also prints the command and line of a failed release guard. The existing native fixture verifies the serial runtime and the actual leaf calls. Six negative tests cover missing functions and incorrect library or caller identity. The patch applies to benchmark commit `37ead44b9f3a144dada8a92b17bd447b138e6689`. It changes benchmark verification only.

**Local Linux AArch64 correctness.** The existing production checkout was frozen at `22d925ed5a31f7246eb2269675e9a3e85d502410`. We built static serial OpenBLAS 0.3.34 from its retained upstream commit `e0166008be8e466242aa76b2ff75ce3f0fbf574a`. In both configurations, GNU ld traced the multiply to `cblas_sgemm.o` and the thread getter to `memory.o`. Its ordinary and observed executables each checked all 4,202,499 outputs of one cropped 2049 × 2051 × 2053 multiply against a serial DGEMM reference. Each protected 50,173 other output cells and preserved read-only inputs. The ordinary and observed output bytes agreed within each configuration.

| Released configuration | Actual leaf calls | Largest absolute error |
| --- | ---: | ---: |
| Two-level Winograd | 49 core + 2 fringes | 0.000537 |
| Aligned two-level Winograd | 49 core + 3 fringes | 0.000465 |

No hosted job used this patch. The two OpenBLAS release modes remain absent from the hosted rankings. This packet covers only the static OpenBLAS check. ArmPL uses a separate dynamic provider check.

Run `python3 replay.py` inside this folder for an offline replay of the patch, regression tests, native link records and correctness receipts. The file `correctness.json` records the static archive and worker SHA-256 values, output digests, error bounds and scoped provenance. The evidence reports correctness and supplies no additional performance measurements.
