# Which matrix layout changes affect multiplication time?

The experiment ran on Intel(R) Xeon(R) 6975P-C using COB’s released stream-B kernel and ordinary sequential oneMKL. For each input and the output, it switches the row stride and the starting address independently. Each of three shapes uses the same logical input matrices across 64 storage layouts. A ratio of 0.90× means the changed layout took 10% less time, and 1.10× means it took 10% more.

**Both changes to B made oneMKL faster on all three shapes.** The larger row stride improved elapsed time by 6.5%, 2.7%, 6.9% in table order. Moving B’s start independently improved it by 3.6%, 11.4%, 2.1%. All six round estimates on every shape agreed on the direction of both effects.

## Whole-layout change

| Shape M×N×K | Method | Current cropped/original | Earlier cropped/original | Six current rounds |
| --- | --- | ---: | ---: | ---: |
| 108×1728×320 | COB | 1.0481× | 1.0073× | 0.9942×–1.1186× |
| 108×1728×320 | oneMKL | 0.9499× | 0.9030× | 0.9050×–1.0462× |
| 137×2179×347 | COB | 0.9770× | 1.0057× | 0.8633×–1.0132× |
| 137×2179×347 | oneMKL | 0.8376× | 0.8643× | 0.7563×–0.8699× |
| 43×2291×137 | COB | 1.0042× | 0.9977× | 0.9393×–1.0815× |
| 43×2291×137 | oneMKL | 0.9095× | 0.9089× | 0.7597×–1.0616× |

The two historical endpoints are exact matches in shape, stride and relative start address. The historical CPU was Intel(R) Xeon(R) 6975P-C.

## Individual layout features

The high setting uses the cropped layout’s stride or start offset. Within each round, each contrast holds the other five features fixed across 32 matched pairs. The table gives the geometric mean of their time ratios over six rounds.

| Shape | Method | A stride | B stride | C stride | A start | B start | C start |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 108×1728×320 | COB | 1.0014× | 1.0025× | 0.9994× | 1.0041× | 1.0023× | 1.0028× |
| 108×1728×320 | oneMKL | 1.0039× | 0.9346× | 1.0005× | 1.0020× | 0.9642× | 1.0046× |
| 108×1728×320 | COB/oneMKL | 0.9974× | 1.0726× | 0.9988× | 1.0021× | 1.0396× | 0.9982× |
| 137×2179×347 | COB | 0.9994× | 1.0023× | 1.0046× | 0.9993× | 0.9960× | 0.9978× |
| 137×2179×347 | oneMKL | 1.0018× | 0.9728× | 0.9988× | 0.9982× | 0.8855× | 0.9984× |
| 137×2179×347 | COB/oneMKL | 0.9976× | 1.0303× | 1.0059× | 1.0011× | 1.1247× | 0.9994× |
| 43×2291×137 | COB | 1.0047× | 0.9924× | 1.0006× | 0.9958× | 1.0181× | 0.9970× |
| 43×2291×137 | oneMKL | 0.9971× | 0.9310× | 1.0003× | 0.9915× | 0.9794× | 0.9993× |
| 43×2291×137 | COB/oneMKL | 1.0077× | 1.0659× | 1.0003× | 1.0044× | 1.0395× | 0.9977× |

The COB/oneMKL values describe how a layout change alters the time ratio of the two implementations. Values above one shift the relative performance in oneMKL’s favor.

## Stride and start address on the same operand

Each number is the ratio of the stride effect at the nonzero starting offset to the stride effect at zero offset. The other four features are averaged over all 16 settings.

| Shape | Method | A stride × A start | B stride × B start | C stride × C start |
| --- | --- | ---: | ---: | ---: |
| 108×1728×320 | COB | 0.9932× | 1.0018× | 0.9917× |
| 108×1728×320 | oneMKL | 0.9985× | 1.0918× | 0.9915× |
| 137×2179×347 | COB | 0.9926× | 0.9967× | 0.9967× |
| 137×2179×347 | oneMKL | 0.9985× | 1.0133× | 1.0049× |
| 43×2291×137 | COB | 0.9975× | 1.0249× | 1.0041× |
| 43×2291×137 | oneMKL | 1.0060× | 1.1379× | 1.0164× |

The strongest within-B interaction on oneMKL occurred at 43×2291×137. Raising B’s stride changed elapsed time to 0.8728× with its original starting address, and to 0.9931× with the cropped address. An optimization that changes both needs a whole-call timing on each combination.

## Effects that changed direction

For COB, A stride, B stride, C stride, A origin, B origin, C origin changed direction across shapes. Individual rounds disagreed in direction for 17 of the 18 shape–feature combinations.

For oneMKL, A stride, C stride, A origin, C origin changed direction across shapes. Individual rounds disagreed in direction for 12 of the 18 shape–feature combinations.


The complete results include the remaining 12 feature interactions, the slope at each starting offset, all six round estimates, and both seed-block estimates. With three fixed shapes on one CPU, the observed ranges describe sensitivity to layout and trial order.

## Verification

The audit checked 747,084,800 output positions against independent FP64 products over 1,536 preflight observations and 2,304 timed observations. Each timing includes 16 complete calls, totaling 36,864 warm calls. Input fingerprints matched across layouts and methods. The largest observed absolute output error was 2.38292e-05 against a limit of 0.002. [Native run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45702348592) · [Benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/9e2483cc4b1839a2e0a392f2cfdb656d807d006e/tools/world-rank/factorial)
