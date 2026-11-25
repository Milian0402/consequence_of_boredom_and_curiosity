# Effects of strides and starting addresses on one-thread SGEMM

This study isolates six address-layout changes in an earlier [cropped-view run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45699142561). It repeats three previously measured geometries. The first two, 108 × 1728 × 320 and 137 × 2179 × 347, had led the external field for aligned inputs and lost that lead on cropped inputs. At 43 × 2291 × 137, normal sequential oneMKL led on the crop. This third geometry was labeled `fresh-short` in the original experiment.

The two timed methods are the installed generic COB `a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6` static archive with `COB_ENABLE_X86_STREAM_B=ON`, and ordinary sequential oneMKL 2026.2.1 from the pinned vendor recipe. The original full field included MKL's alternative direct-call interface. This experiment measures attribution between the two specified interfaces.

## Interventions and endpoints

Mask bits 0 through 5 independently select the A, B and C leading dimensions and the A, B and C starting offsets from page-aligned parents. Bit 0 chooses the low or high A leading dimension. Bit 3 chooses the low or high absolute starting offset of A in floats. The same conventions apply to B and C. Each low value comes from the earlier aligned layout, and each high value from its cropped companion. High origins equal the previous row offset multiplied by the **high** leading dimension, plus its column offset, even at mixed corners where the corresponding leading dimension is low. The actual first-address residue equals four times this absolute offset, modulo 64.

The three geometries each contribute 64 distinct memory layouts. For each geometry masks 0 and 63 reproduce its old endpoints exactly, including the accessible rounded parent sizes. The 62 mixed corners combine the six interventions. At mixed corners a logical row may straddle a nominal parent row. The worker treats each allocation as a flat buffer and strides between logical row beginnings. Each leading dimension is at least the logical row width, so all logical rows are disjoint.

| Geometry | Leading dimensions at mask 0 → 63, A/B/C | Starting offsets in floats at mask 0 → 63, A/B/C |
| --- | --- | --- |
| 108 × 1728 × 320 | 324/1732/1732 → 359/1799/1821 | 0/0/0 → 385/3623/5485 |
| 137 × 2179 × 347 | 354/2186/2186 → 388/2252/2274 | 0/0/0 → 803/6777/9118 |
| 43 × 2291 × 137 | 140/2294/2294 → 182/2368/2390 | 0/0/0 → 753/2391/4798 |

## Measurement and analysis

Both methods use the same logical input values at all 64 masks for each geometry, distribution and seed. A 64-bit FNV-1a checksum of logical binary32 input values in canonical little-endian row-major order records the identity of both inputs. The process protects both complete input parents as read-only, places inaccessible pages before and after every parent, checks all C elements outside the live footprint, and verifies every output against independently accumulated FP64 values at a maximum absolute error of 0.002. For each cell and method, four qualification processes use two random seeds, a cancellation fixture with the first seed and a scaled fixture with the second. A separate linked witness records the active COB kernel at every cell. Each worker reports actual strides, offsets and modulo-64 addresses, its pinned logical CPU and single-thread status.

The original two seeds are 73194513 and 38641729. Six fresh-process batches per method and cell each contain 16 warm complete SGEMMs, with one first complete call recorded separately. Within the first block, for replicate t=0,1,2, visit slots s=0…63 and within each slot visit the geometry (h+t) mod 3 for h=0,1,2. For geometry i, the mask is (37s+11i+17t) mod 64. The second block reverses each corresponding 192-cell traversal. The pair order alternates using (i+s+t+block) mod 2 as the first method's index. Each method therefore leads three of six pairs at every cell.

The frozen panel SHA-256 is `5742b14e8e2fd71993ce98070fabf85b648f1ad69efb7f16ff07f2a8eb246631`. [Prospective contrasts, interactions and endpoint replication criteria](estimands.md)

The conclusions concern these binary stride and offset changes on the measured CPU, data and software. Page-aligned parent addresses fix offset residues but virtual-address placement and shared-runner background activity may differ between processes. The protocol measures their combined effect over six paired batches on a pinned logical CPU; it does not identify physical cache sets, absolute hardware causes or a worldwide rank.
