# Optimization Attempt Timeline

Last updated: 2026-05-27

This tracks the matrix-multiplication speed work so far. The narrow comparison
scope is single-threaded FP32 row-major GEMM for `C = A * B` with `alpha = 1`
and `beta = 0`.

Entries on 2026-05-05 are ordered top-to-bottom from newest to oldest in the
recent work block, then older historical material follows. For exact ordering,
use the git history for this file; the current recent sequence is anchored by:

- `d2468c3` Broaden medium SME direct route.
- `aef6118` Document wide m64 large-K tuning.
- `0a20d56` Tune wide m64 large-K chunk.
- `19c331b` Document wide m64 k1536 chunk tuning.
- `1b23220` Use full K chunk for wide m64 k1536.
- `9ad4f50` Document wide m64 K chunk tuning.
- `ea7c963` Increase wide m64 SME K chunk.
- `fc657ef` Document B-pack prefetch distance increase.
- `7fae628` Increase B-pack prefetch distance.

## Timeline

### 2026-05-27 local-uncommitted: m224 medium SME direct edge accepted

The lower medium edge now reaches `m = 224` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The dispatcher folds
`m = 224/256/288/320/352/384` into one compact SME direct-`B` gate and keeps
the lower neighbor, `n = 1216`, and higher-K rows off this route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m224-extra-n-sme.c` showed all
twenty candidate rows positive. Repeat-301, `iters=8` confirmed the rectangle:
`224x1280x832` median `1.3818x`, `224x1280x960` `1.3674x`,
`224x1280x1024` `1.3567x`, `224x1280x1088` `1.3597x`, and
`224x1280x1152` `1.2634x`. The weakest confirmed median in the rectangle was
`224x1344x960` at `1.1116x`, still with a positive holdout.

The guards kept the route bounded: `224x1216x832`, `224x1280x1536`,
`224x1280x2048`, lower neighbor `192x1280x832`, and routed upper neighbor
`256x1280x832` stayed neutral/noisy or behavior-identical.

Correctness coverage adds the twenty accepted `m = 224` rows.

### 2026-05-27 local-uncommitted: m256 medium SME direct edge accepted

The lower medium edge now reaches `m = 256` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The dispatcher folds
`m = 256/288/320/352/384` into one compact SME direct-`B` gate and keeps the
lower neighbor, `n = 1216`, and higher-K rows off this route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m256-extra-n-sme.c` showed all
twenty candidate rows positive. Repeat-301, `iters=8` confirmed the rectangle:
`256x1280x832` median `1.3305x`, `256x1280x960` `1.3332x`,
`256x1280x1024` `1.3322x`, `256x1280x1088` `1.3600x`, and
`256x1280x1152` `1.3675x`. The weakest confirmed median in the rectangle was
`256x1344x1088` at `1.2545x`, still with a positive holdout.

The guards kept the route bounded: `256x1216x832`, `256x1280x1536`,
`256x1280x2048`, lower neighbor `224x1280x832`, and routed upper neighbor
`288x1280x832` stayed neutral/noisy or behavior-identical.

Correctness coverage adds the twenty accepted `m = 256` rows.

### 2026-05-27 local-uncommitted: m288 medium SME direct edge accepted

The lower medium edge now reaches `m = 288` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The dispatcher folds
`m = 288/320/352/384` into one compact SME direct-`B` gate and keeps the lower
neighbor, `n = 1216`, and higher-K rows off this route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m288-extra-n-sme.c` showed all
twenty candidate rows positive. Repeat-301, `iters=8` confirmed the rectangle:
`288x1280x832` median `1.3090x`, `288x1280x960` `1.2894x`,
`288x1280x1024` `1.2899x`, `288x1280x1088` `1.3074x`, and
`288x1280x1152` `1.2966x`. The weakest confirmed median in the rectangle was
`288x1344x832` at `1.2367x`, still with a positive holdout.

The guards kept the route bounded: `288x1216x832`, `288x1280x1536`,
`288x1280x2048`, lower neighbor `256x1280x832`, and routed upper neighbor
`320x1280x832` stayed neutral/noisy or behavior-identical.

Correctness coverage adds the twenty accepted `m = 288` rows.

### 2026-05-27 local-uncommitted: m320 medium SME direct edge accepted

The lower medium edge now reaches `m = 320` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The dispatcher folds
`m = 320/352/384` into one compact SME direct-`B` gate and keeps the lower
neighbor, `n = 1216`, and higher-K rows off this route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m320-extra-n-sme.c` showed all
twenty candidate rows positive. Repeat-301, `iters=8` confirmed the rectangle:
`320x1280x832` median `1.2629x`, `320x1280x960` `1.2639x`,
`320x1280x1024` `1.2654x`, `320x1280x1088` `1.2689x`, and
`320x1280x1152` `1.3009x`. The weakest confirmed median in the rectangle was
`320x1408x1088` at `1.1006x`, still with a positive holdout.

The guards kept the route bounded: `320x1216x832`, `320x1280x1536`,
`320x1280x2048`, lower neighbor `288x1280x832`, and routed upper neighbor
`352x1280x832` stayed neutral/noisy or behavior-identical.

Correctness coverage adds the twenty accepted `m = 320` rows.

### 2026-05-27 local-uncommitted: m352 medium SME direct edge accepted

The lower medium edge now reaches `m = 352` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The dispatcher folds
`m = 352/384` into one compact SME direct-`B` gate while keeping the lower
neighbor and higher-K rows off the new route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m352-extra-n-sme.c` showed the
rectangle positive, with `352x1472x832` noisy enough to require confirmation.
Repeat-301, `iters=8` confirmed all twenty rows. The lower-width rows were
strong: `352x1280x832` median `1.2437x`, `352x1280x960` `1.2427x`,
`352x1280x1024` `1.2451x`, `352x1280x1088` `1.2603x`, and
`352x1280x1152` `1.2694x`. The noisy `352x1472x832` row confirmed at median
`1.2040x`, bootstrap95 `[1.2164,1.2919]`, with positive holdout.

The guards kept the route bounded: `352x1216x832`, `352x1280x1536`,
`352x1280x2048`, lower neighbor `320x1280x832`, and routed upper neighbor
`384x1280x832` stayed neutral/noisy or behavior-identical.

Correctness coverage adds the twenty accepted `m = 352` rows.

### 2026-05-27 local-uncommitted: m384 medium SME direct edge accepted

The lower medium edge now reaches `m = 384` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The exact
`384x1280x1024` row was already on SME direct-`B`; this pass fills the rest of
the medium rectangle while keeping higher-K and adjacent-height guards off the
new route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m384-extra-n-sme.c` showed the new
rows positive, with the existing exact `384x1280x1024` behavior-identical.
Repeat-301, `iters=8` confirmed the route. Representative rows:
`384x1280x832` median `1.2209x`, bootstrap95 `[1.2059,1.2236]`;
`384x1280x960` `1.2434x`, bootstrap95 `[1.1876,1.2128]`;
`384x1280x1088` `1.0835x`, bootstrap95 `[1.0976,1.1454]`;
`384x1344x960` `1.1970x`, bootstrap95 `[1.1045,1.1663]`;
`384x1408x1152` `1.0870x`, bootstrap95 `[1.0635,1.0999]`;
and `384x1472x1152` `1.0443x`, bootstrap95 `[1.0615,1.1149]`.
The strongest rows were around `1.23x`.

The guards kept the rule bounded: `384x1216x832`, existing exact
`384x1280x1536`, `384x1280x2048`, lower neighbor `352x1280x832`, and routed
upper neighbor `416x1280x832` stayed neutral/noisy or behavior-identical.

Correctness coverage adds the nineteen new `m = 384` rows; the existing
`384x1280x1024` coverage already covered the remaining row in the rectangle.

### 2026-05-27 local-uncommitted: m416 medium SME direct edge accepted

The lower medium edge now reaches `m = 416` for `n = 1280/1344/1408/1472` at
all tested 64-step K points from `832` through `1152`. The dispatcher now folds
the accepted `m = 416/448/480` rectangles into one compact SME direct-`B`
gate.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m416-extra-n-sme.c` showed all
twenty candidate rows positive. Repeat-301, `iters=8` confirmed the full range:
the weakest full-run row, `416x1344x1152`, still had median `1.0897x` with
bootstrap95 `[1.0514,1.0888]`, while the strongest rows were around
`1.24x`. A focused repeat-301 rerun confirmed the noisy `n = 1472` tail:
`416x1472x960` median `1.2032x`, `416x1472x1024` `1.2169x`,
`416x1472x1088` `1.2208x`, and `416x1472x1152` `1.2139x`, all with positive
holdouts.

The guards kept the route bounded: `416x1216x832`, `416x1280x1536`,
`384x1280x832`, and routed neighbor `448x1280x832` stayed neutral/noisy or
behavior-identical after the compact gate rewrite.

The same pass filled the missing `k = 1024/1088` correctness coverage for the
already accepted `m = 448/480` broad gates. A repeat-301 audit against
`/private/tmp/cob-next-audit/gemm-baseline-m480-extra-n-sme.c` confirmed all
sixteen intermediate rows, with medians from `1.1594x` to `1.2248x`.

Correctness coverage adds the twenty accepted `m = 416` rows plus sixteen
intermediate-K rows for `m = 448/480`.

### 2026-05-27 local-uncommitted: m448 medium SME direct edge accepted

The lower medium edge now reaches `m = 448` for `n = 1280/1344/1408/1472` at
`k = 832/960/1024/1088/1152`, matching the accepted `m = 480` rectangle.

A broad repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m448-extra-n-sme.c` showed all
twelve candidate rows positive. Repeat-301, `iters=8` confirmed the route:
`448x1280x832` median `1.2062x`, bootstrap95 `[1.1889,1.2643]`;
`448x1280x960` `1.2137x`, bootstrap95 `[1.1936,1.2777]`;
`448x1280x1152` `1.2341x`, bootstrap95 `[1.1859,1.2658]`;
`448x1344x832` `1.1664x`, bootstrap95 `[1.1696,1.2409]`;
`448x1344x960` `1.1831x`, bootstrap95 `[1.1529,1.2169]`;
`448x1344x1152` `1.1018x`, bootstrap95 `[1.0508,1.1305]`;
`448x1408x832` `1.1903x`, bootstrap95 `[1.1905,1.2017]`;
`448x1408x960` `1.2106x`, bootstrap95 `[1.1988,1.2122]`;
`448x1408x1152` `1.2245x`, bootstrap95 `[1.2022,1.2191]`;
`448x1472x832` `1.1701x`, bootstrap95 `[1.1672,1.1773]`;
`448x1472x960` `1.1898x`, bootstrap95 `[1.1755,1.1890]`; and
`448x1472x1152` `1.2062x`, bootstrap95 `[1.1867,1.2036]`.

The guards kept the route bounded: `448x1216x832`, `448x1280x1536`,
`416x1280x832`, and `480x1280x832` stayed neutral/noisy or
behavior-identical.

Correctness coverage initially added the twelve edge `m = 448` rows; the
`k = 1024/1088` rows were filled in by the later m416 pass.

### 2026-05-27 local-uncommitted: m480 medium SME direct edge accepted

The lower medium edge now reaches `m = 480` for `n = 1280/1344/1408/1472` at
`k = 832/960/1024/1088/1152`, routing the full tested rectangle through SME
direct-`B`.

A broad repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m480-extra-n-sme.c` showed all
twelve candidate rows positive. Repeat-301, `iters=8` confirmed the route:
`480x1280x832` median `1.2182x`, bootstrap95 `[1.2012,1.3029]`;
`480x1280x960` `1.2031x`, bootstrap95 `[1.1804,1.2672]`;
`480x1280x1152` `1.2196x`, bootstrap95 `[1.1762,1.2772]`;
`480x1344x832` `1.1540x`, bootstrap95 `[1.1536,1.2233]`;
`480x1344x960` `1.1757x`, bootstrap95 `[1.1517,1.2192]`;
`480x1344x1152` `1.1812x`, bootstrap95 `[1.1727,1.2499]`;
`480x1408x832` `1.1809x`, bootstrap95 `[1.1702,1.2456]`;
`480x1408x960` `1.1911x`, bootstrap95 `[1.1363,1.2089]`;
`480x1408x1152` `1.2060x`, bootstrap95 `[1.1680,1.2441]`;
`480x1472x832` `1.1656x`, bootstrap95 `[1.1576,1.2235]`;
`480x1472x960` `1.1792x`, bootstrap95 `[1.1621,1.2263]`; and
`480x1472x1152` `1.1932x`, bootstrap95 `[1.1177,1.1837]`.

The guards kept the route bounded: `480x1216x832`, `480x1280x1536`,
`448x1280x832`, and `512x1280x832` stayed neutral/noisy or
behavior-identical.

Correctness coverage initially added the twelve edge `m = 480` rows; the
`k = 1024/1088` rows were filled in by the later m416 pass.

### 2026-05-27 local-uncommitted: m512 medium SME direct edge accepted

The lower medium edge now reaches `m = 512` for `n = 1280/1344/1408` at
`k = 832/960`, while keeping `k = 1152`, `n = 1472`, and the existing exact
`512x1280x1536` route unchanged.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m512-extra-n-sme.c` showed all six
candidate rows positive. Repeat-301, `iters=8` confirmed the route:
`512x1280x832` median `1.1801x`, bootstrap95 `[1.1642,1.1853]`;
`512x1280x960` `1.1924x`, bootstrap95 `[1.1780,1.1962]`;
`512x1344x832` `1.1431x`, bootstrap95 `[1.1398,1.1503]`;
`512x1344x960` `1.1605x`, bootstrap95 `[1.1402,1.1581]`;
`512x1408x832` `1.1765x`, bootstrap95 `[1.1665,1.1782]`; and
`512x1408x960` `1.1920x`, bootstrap95 `[1.1783,1.1946]`.

The guards kept the route bounded: `512x1280x1152`, `512x1344x1152`,
`512x1408x1152`, and `512x1472x832` stayed neutral/noisy. Existing exact
`512x1280x1536`, lower neighbor `480x1280x832`, and upper neighbor
`544x1280x832` were behavior-identical/noisy.

Correctness coverage adds the six accepted `m = 512` rows.

### 2026-05-27 local-uncommitted: m544 medium SME direct edge accepted

The lower medium edge extends to `m = 544` for `n = 1280/1344/1408` at
`k = 832/960`, while keeping `k = 1152` and `n = 1472` on the packed route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m544-extra-n-sme.c` showed all six
candidate rows positive. Repeat-301, `iters=8` confirmed the route:
`544x1280x832` median `1.1718x`, bootstrap95 `[1.1799,1.2477]`;
`544x1280x960` `1.2025x`, bootstrap95 `[1.2082,1.2839]`;
`544x1344x832` `1.1374x`, bootstrap95 `[1.1180,1.1863]`;
`544x1344x960` `1.1594x`, bootstrap95 `[1.1438,1.2114]`;
`544x1408x832` `1.1632x`, bootstrap95 `[1.1675,1.2334]`; and
`544x1408x960` `1.1776x`, bootstrap95 `[1.1418,1.2190]`.

The guards kept the route bounded: `544x1280x1152`, `544x1344x1152`,
`544x1408x1152`, and `544x1472x832` stayed neutral/noisy. Existing neighbors
`512x1280x832` and `576x1280x832` were behavior-identical/noisy.

Correctness coverage adds the six accepted `m = 544` rows.

### 2026-05-27 local-uncommitted: m576 medium SME direct edge accepted

The lower medium edge extends to `m = 576` for `n = 1280/1344/1408` at
`k = 832/960`, while keeping `k = 1152` and `n = 1472` on the packed route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m576-extra-n-sme.c` showed all six
candidate rows positive. Repeat-301, `iters=8` confirmed the route:
`576x1280x832` median `1.1806x`, bootstrap95 `[1.1880,1.2871]`;
`576x1280x960` `1.1687x`, bootstrap95 `[1.1013,1.1947]`;
`576x1344x832` `1.1431x`, bootstrap95 `[1.1232,1.1938]`;
`576x1344x960` `1.1439x`, bootstrap95 `[1.1122,1.1931]`;
`576x1408x832` `1.1573x`, bootstrap95 `[1.1013,1.1814]`; and
`576x1408x960` `1.1799x`, bootstrap95 `[1.1498,1.2247]`.

The guards kept the route bounded: `576x1280x1152`, `576x1344x1152`,
`576x1408x1152`, and `576x1472x832` stayed neutral/noisy. Existing upper
neighbor `608x1280x832` stayed behavior-identical/noisy; lower neighbor
`544x1280x832` showed positive no-route noise, so it is left for a separate
direct probe rather than included here.

Correctness coverage adds the six accepted `m = 576` rows.

### 2026-05-27 local-uncommitted: m608 medium SME direct edge accepted

The lower medium edge extends below `m = 640`, but only for exact
`608x1280x832/960` and `608x1408x832/960`. The `n = 1344` rows were too weak
to include.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m608-extra-n-sme.c` showed strong
signal at `n = 1280` and `n = 1408`, while `n = 1344`, `k = 1152`, and
`n = 1472` stayed neutral or weak. Repeat-301, `iters=8` confirmed the narrowed
route: `608x1280x832` median `1.1463x`, bootstrap95 `[1.0931,1.1278]`;
`608x1280x960` `1.1689x`, bootstrap95 `[1.1574,1.1739]`;
`608x1408x832` `1.1528x`, bootstrap95 `[1.1087,1.1317]`; and
`608x1408x960` `1.0849x`, bootstrap95 `[1.0382,1.0771]`.

The guards kept the rule bounded: `608x1344x832/960`, all tested
`k = 1152` rows, and `608x1472x832` stayed neutral/noisy. Existing neighbors
`576x1280x832` and `640x1280x832` were behavior-identical/noisy.

Correctness coverage adds the four accepted `m = 608` rows.

### 2026-05-27 local-uncommitted: m640 medium SME direct edge accepted

The lower medium edge extends to `m = 640` for `n = 1280/1344/1408` at
`k = 832/960`, while keeping `k = 1152` and `n = 1472` on the packed route.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m640-extra-n-sme.c` showed all six
candidate rows positive, though noisy. Repeat-301, `iters=8` confirmed the
route: `640x1280x832` median `1.1523x`, bootstrap95 `[1.1219,1.1886]`;
`640x1280x960` `1.1595x`, bootstrap95 `[1.1337,1.2092]`;
`640x1344x832` `1.1180x`, bootstrap95 `[1.0945,1.1680]`;
`640x1344x960` `1.1375x`, bootstrap95 `[1.1125,1.1802]`;
`640x1408x832` `1.1460x`, bootstrap95 `[1.1102,1.1702]`; and
`640x1408x960` `1.1339x`, bootstrap95 `[1.0891,1.1711]`.

The guards kept the route bounded: `640x1280x1152`, `640x1344x1152`,
`640x1408x1152`, and `640x1472x832` stayed neutral/noisy. Existing neighbors
`608x1280x832` and `672x1280x832` were behavior-identical/noisy.

Correctness coverage adds the six accepted `m = 640` rows.

### 2026-05-27 local-uncommitted: m992 medium SME direct edge accepted

The upper side of the broad medium route now fills the 32-row gap between
`m = 960` and `m = 1024`. The accepted `m = 992` subset mirrors the nearby
`m = 1024` shape set: `n = 1280` at `k = 832/960`, `n = 1344` at
`k = 832/960/1152`, and `n = 1472` at `k = 832/960`.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m992-extra-n-sme.c` showed the
candidate rows winning while `n = 1408`, `992x1280x1152`, and
`992x1472x1152` stayed neutral/noisy. Repeat-301, `iters=8` confirmed the
route: `992x1280x832` median `1.1080x`, bootstrap95 `[1.1044,1.1128]`;
`992x1280x960` `1.1213x`, bootstrap95 `[1.1081,1.1233]`;
`992x1344x832` `1.0793x`, bootstrap95 `[1.0702,1.0802]`;
`992x1344x960` `1.0815x`, bootstrap95 `[1.0386,1.0618]`;
`992x1344x1152` `1.0865x`, bootstrap95 `[1.0143,1.0411]`;
`992x1472x832` `1.0870x`, bootstrap95 `[1.0714,1.0862]`; and
`992x1472x960` `1.0972x`, bootstrap95 `[1.0860,1.1043]`.

The guards kept the rule bounded: `992x1280x1152`, `992x1408x832/960/1152`,
and `992x1472x1152` stayed neutral/noisy. Existing neighbors
`960x1280x832`, `1024x1280x832`, and `1024x1408x832` were
behavior-identical/noisy.

Correctness coverage adds the seven accepted `m = 992` rows.

### 2026-05-27 local-uncommitted: m672 medium SME direct edge accepted

The lower medium edge extends one more step below `m = 704`, but only for a
narrower subset: `672x1280x832/960`, `672x1344x832`, and
`672x1408x832/960` now route through SME direct-`B`.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m672-extra-n-sme.c` found strong
signal for `n = 1280` and `n = 1408`, but `672x1344x960` was too weak and
`672x1472x832` stayed neutral. Repeat-301, `iters=8` confirmed the narrowed
route: `672x1280x832` median `1.1390x`, bootstrap95 `[1.0980,1.2036]`;
`672x1280x960` `1.1529x`, bootstrap95 `[1.1323,1.1805]`;
`672x1344x832` `1.1159x`, bootstrap95 `[1.1093,1.1189]`;
`672x1408x832` `1.1499x`, bootstrap95 `[1.1411,1.1520]`; and
`672x1408x960` `1.1988x`, bootstrap95 `[1.1681,1.2653]`.

The omitted rows kept the rule bounded: `672x1344x960`, all tested
`k = 1152` rows, and `672x1472x832` stayed neutral/noisy. Existing neighbors
`640x1280x832` and `704x1280x832` were behavior-identical/noisy.

Correctness coverage adds the five accepted `m = 672` rows.

### 2026-05-27 local-uncommitted: m800 medium SME direct edge accepted

The lower medium edge now extends through `m = 800` with the same bounded shape
set as `m = 768`: `n = 1280` at `k = 832/960`, `n = 1344/1408` at
`k = 832/960/1152`, and `n = 1472, k = 832`.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m800-extra-n-sme.c` showed the
candidate rows winning strongly while `800x1280x1152`,
`800x1472x960/1152`, and accepted-neighbor rows stayed neutral/noisy.
Repeat-301, `iters=8` confirmed the route: `800x1280x832` median `1.1193x`,
bootstrap95 `[1.1133,1.1235]`; `800x1280x960` `1.1144x`, bootstrap95
`[1.0616,1.0886]`; `800x1344x832` `1.0920x`, bootstrap95
`[1.0566,1.0780]`; `800x1344x960` `1.1145x`, bootstrap95
`[1.1009,1.1158]`; `800x1344x1152` `1.1351x`, bootstrap95
`[1.1115,1.1299]`; `800x1408x832` `1.1236x`, bootstrap95
`[1.1204,1.1303]`; `800x1408x960` `1.1312x`, bootstrap95
`[1.1121,1.1268]`; `800x1408x1152` `1.1470x`, bootstrap95
`[1.1306,1.1444]`; and `800x1472x832` `1.1078x`, bootstrap95
`[1.1026,1.1121]`.

The omitted rows stayed bounded: `800x1280x1152`, `800x1472x960`, and
`800x1472x1152` were neutral/noisy. Existing neighbors `768x1280x832` and
`832x1280x832` were behavior-identical/noisy.

Correctness coverage adds the nine accepted `m = 800` rows.

### 2026-05-27 local-uncommitted: m736 medium SME direct edge accepted

The lower medium edge now fills the 32-row gap between the accepted `m = 704`
and `m = 768` routes. Exact `m = 736`, `n = 1280/1344/1408`, `k = 832/960`
now uses the SME direct-`B` medium kernel.

A repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m736-extra-n-sme.c` showed all six
target rows positive while `k = 1152`, `n = 1472`, and accepted-neighbor
guards stayed neutral/noisy. Repeat-301, `iters=8` confirmed the route:
`736x1280x832` median `1.1411x`, bootstrap95 `[1.1129,1.1311]`;
`736x1280x960` `1.0913x`, bootstrap95 `[1.0630,1.0848]`;
`736x1344x832` `1.1133x`, bootstrap95 `[1.0939,1.1087]`;
`736x1344x960` `1.1296x`, bootstrap95 `[1.1063,1.1221]`;
`736x1408x832` `1.0576x`, bootstrap95 `[1.0336,1.0921]`; and
`736x1408x960` `1.0531x`, bootstrap95 `[1.0753,1.1388]`.

The guards kept the route bounded: `736x1280x1152`, `736x1344x1152`,
`736x1408x1152`, and `736x1472x832` stayed neutral/noisy. Existing routed
neighbors `704x1280x832` and `768x1280x832` were behavior-identical/noisy.

Correctness coverage adds the six accepted `m = 736` rows.

### 2026-05-27 local-uncommitted: n1216 SME-before-strided probe rejected

The `n = 1216` AMX strided-`B` route had several low one-shot medians in the
fresh grid, so a temp source tried routing exact candidates through SME direct
before strided-`B`.

The broad repeat-101 screen was mixed, but repeat-301, `iters=8` rejected the
narrowed candidate: `960x1216x832` median `0.9908x`, bootstrap95
`[0.9660,0.9926]`; `960x1216x960` `0.9945x`, bootstrap95
`[0.9790,0.9993]`; and `1120x1216x832` `0.9961x`, bootstrap95
`[0.9664,1.0134]`. Keep these rows on the existing strided-`B` route.

### 2026-05-27 local-uncommitted: exact m1184 n1280 SME direct accepted

The upper-edge `n = 1280` pair extends one more 32-row step: exact
`1184x1280x832` and `1184x1280x960` now use the SME direct-`B` medium kernel.

Repeat-201, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-m1184-n1280-sme.c` validated the
pair: `1184x1280x832` median `1.0983x`, bootstrap95 `[1.0920,1.1047]`,
B-faster `197/201`, holdout median `1.0965x`; and `1184x1280x960` median
`1.1028x`, bootstrap95 `[1.0801,1.1017]`, B-faster `188/201`, holdout median
`1.1014x`.

The guard rows kept the rule exact: `1184x1280x1152` stayed neutral/noisy,
accepted-neighbor `1152x1280x832` was behavior-identical/noisy, next-row
`1216x1280x832` stayed neutral, `1184x1344x832` stayed neutral, and
`1184x1472x960` regressed on mean-log CI.

Correctness coverage adds `1184x1280x832` and `1184x1280x960`.

### 2026-05-27 local-uncommitted: exact m1152 n1280 SME direct accepted

The next upper edge did not support a broad `m = 1152` rule, but exact
`1152x1280x832` and `1152x1280x960` were strong enough to route through SME
direct-`B`.

The broad repeat-101 screen against
`/private/tmp/cob-next-audit/gemm-baseline-m1152-extra-n-sme.c` only made the
`n = 1280, k = 832/960` pair look credible; `n = 1344`, `n = 1472`, `k =
1152`, and the `1408` side guards were weak, noisy, or neutral. Focused
repeat-301, `iters=8` confirmed the accepted pair: `1152x1280x832` median
`1.0980x`, bootstrap95 `[1.0829,1.0983]`, B-faster `286/301`, holdout median
`1.0966x`; and `1152x1280x960` median `1.1060x`, bootstrap95
`[1.0859,1.1007]`, B-faster `281/301`, holdout median `1.1065x`.

Correctness coverage adds `1152x1280x832` and `1152x1280x960`.

### 2026-05-27 local-uncommitted: m1120 medium SME direct upper edge accepted

The `m = 1120` upper edge was noisier than the previous two row bands, so the
accepted rule is slightly narrower: `n = 1280` at `k = 832/960`,
`n = 1344` at `k = 832/960/1152`, and only `1472x960`.

Repeat-201, `iters=4` against
`/private/tmp/cob-next-audit/gemm-baseline-m1120-extra-n-sme.c` narrowed the
gate: `1120x1472x832` had positive median but negative mean-log CI, and the
`1408` plus `k = 1152` side guards stayed neutral/noisy. A repeat-301,
`iters=8` confirmation validated the accepted rows: `1120x1280x832` median
`1.0978x`, bootstrap95 `[1.0731,1.0933]`; `1120x1280x960` `1.0847x`,
bootstrap95 `[1.0432,1.0657]`; `1120x1344x832` `1.0730x`, bootstrap95
`[1.0638,1.0748]`; `1120x1344x960` `1.0243x`, bootstrap95
`[1.0277,1.1031]`; `1120x1344x1152` `1.0680x`, bootstrap95
`[1.0208,1.0763]`; and `1120x1472x960` `1.0938x`, bootstrap95
`[1.0823,1.0926]`.

The unchanged-neighbor guards `1088x1280x832` and `1152x1280x832` were noisy
or neutral in the broad screen, so the rule stops at this exact `m = 1120`
subset for now.

Correctness coverage adds the six accepted `m = 1120` rows.

### 2026-05-27 local-uncommitted: m1088 medium SME direct upper edge accepted

The `m = 1088` upper edge now mirrors the accepted `m = 1056` subset:
`n = 1280` at `k = 832/960`, `n = 1344` at `k = 832/960/1152`, and
`n = 1472` at `k = 832/960`.

Repeat-201, `iters=4` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-m1088-extra-n-sme.c` validated the
seven target rows: `1088x1280x832` median `1.0890x`, `1088x1280x960`
`1.0785x`, `1088x1344x832` `1.0536x`, `1088x1344x960` `1.0913x`,
`1088x1344x1152` `1.1076x`, `1088x1472x832` `1.0804x`, and
`1088x1472x960` `1.0905x`. All target rows had positive bootstrap intervals
in the repeat-201 run.

The guard rows kept the rule bounded: `1088x1408x832` and `1088x1408x960`
stayed neutral/noisy, `1088x1280x1152` and `1088x1472x1152` stayed neutral,
and accepted-neighbor `1056x1280x832` plus next-row `1120x1280x832` were
behavior-identical/noisy.

Correctness coverage adds the seven accepted `m = 1088` rows.

### 2026-05-27 local-uncommitted: 512x1152x2048 packed-B MC512 rejected

The packed-B audit tried one exact row-block change in
`cob_sgemm_amx_packed_b_mc`: route `512x1152x2048` through the 512-row AMX
large packed-B block, matching the existing exact `512x1280x2048` rule.

Packed-B repeat-201, `iters=8` against
`/private/tmp/cob-next-audit/gemm-baseline-512x1152x2048-packed-mc512.c` was
too weak to route: target `512x1152x2048` measured median `1.0155x`, but
bootstrap95 was `[0.9957,1.0166]` and holdout median was only `1.0120x` with
weak sign. Guards were not supportive: `512x1216x2048` `0.9965x`,
`512x1280x2048` neutral/noisy at `1.0019x`, `512x1152x1536` `0.9935x`,
`512x1152x3072` `0.9963x`, and `768x1152x2048` noisy at `1.0387x`.

Packed-AB repeat-201, `iters=8` rejected the idea harder: target
`512x1152x2048` measured median `0.9606x`, bootstrap95 `[0.9188,1.0402]`,
with holdout median `0.9885x`. Leave the exact MC512 packed-B helper unchanged.

### 2026-05-27 local-uncommitted: m1056 medium SME direct upper edge accepted

The medium extra-`N` upper edge now extends one 32-row step past the exact
`m = 1024` subset. The accepted `m = 1056` rule mirrors that subset:
`n = 1280` at `k = 832/960`, `n = 1344` at `k = 832/960/1152`, and
`n = 1472` at `k = 832/960`.

Repeat-201, `iters=4` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-m1056-extra-n-sme.c` was positive
on all target rows: `1056x1280x832` median `1.1064x`, `1056x1280x960`
`1.1012x`, `1056x1344x832` `1.0632x`, `1056x1344x960` `1.0769x`,
`1056x1344x1152` `1.0534x`, `1056x1472x832` `1.0843x`, and
`1056x1472x960` `1.0878x`. The `1344` rows were rechecked at repeat-301,
`iters=8`: `1056x1344x832` median `1.0798x`, bootstrap95
`[1.0683,1.0811]`; `1056x1344x960` `1.0889x`, bootstrap95
`[1.0546,1.0782]`; and `1056x1344x1152` `1.0945x`, bootstrap95
`[1.0590,1.0829]`.

The guard rows kept the rule bounded: `1056x1408x832` and `1056x1408x960`
stayed neutral/noisy, `1056x1280x1152` and `1056x1472x1152` stayed neutral,
and accepted-neighbor guards `1024x1280x832` and `1088x1280x832` were
behavior-identical/noisy.

Correctness coverage adds the seven accepted `m = 1056` rows.

### 2026-05-27 local-uncommitted: m704 medium SME direct lower edge accepted

The new medium extra-`N` edge route extends one more bounded row band downward:
`m = 704`, `1280 <= n <= 1408`, and `k = 832` or `960` now uses the SME
direct-`B` medium kernel.

The first repeat-101, `iters=4` screen against
`/private/tmp/cob-next-audit/gemm-baseline-m704-extra-n-sme.c` was strongly
positive on all six target cells, with medians from `1.1118x` to `1.1451x`.
Repeat-201 confirmation stayed positive for `704x1344x832` `1.0656x`,
`704x1344x960` `1.0893x`, `704x1408x832` `1.0543x`, and `704x1408x960`
`1.1385x`; the `1280` rows were noisier in that run, so they were rechecked
separately.

Focused repeat-301, `iters=8` confirmed the `n = 1280` pair cleanly:
`704x1280x832` median `1.1364x`, bootstrap95 `[1.1330,1.1431]`, B-faster
`295/301`, holdout median `1.1353x`; and `704x1280x960` median `1.1448x`,
bootstrap95 `[1.1406,1.1522]`, B-faster `297/301`, holdout median `1.1471x`.

The guard rows kept the rule bounded: `640x1280x832` stayed neutral/noisy,
`704x1280x1152` stayed neutral, `704x1472x832` stayed neutral, and accepted
`768x1280x832` was behavior-identical/noisy.

Correctness coverage adds the six accepted `m = 704` rows.

### 2026-05-27 local-uncommitted: exact 384x896x1536 SME direct accepted

The lower-row sibling of the `512x896` SME-direct band has one useful exact
case. The dispatcher now routes only `384x896x1536` through the SME direct-`B`
medium kernel.

Repeat-201, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-384x896-sme.c` measured
`384x896x1536` at median `1.2278x`, mean-log `1.2270x`, bootstrap95
`[1.1892,1.2625]`, B-faster `187/201`, sign-p `8.52e-40`, and holdout median
`1.2337x`.

The exact gate stayed bounded: `384x896x1024` was neutral at median `1.0000x`,
`384x896x2048` was neutral/noisy at `1.0034x`, `384x960x1536` stayed neutral
at `0.9987x`, and the already-routed `512x896x1536` stayed behavior-identical
at `1.0082x`.

Correctness coverage adds `384x896x1536`.

### 2026-05-27 local-uncommitted: exact n1152 k832 SME-before-strided accepted

The older broad `n = 1152` SME-before-AMX-strided probe was correctly rejected,
but its notes called out two isolated positive rows. A fresh exact probe now
routes only `1024x1152x832` and `1088x1152x832` around AMX strided-`B` and
into the existing SME direct-`B` medium kernel.

Repeat-201, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-n1152-exact-sme-before-strided.c`
validated both exact targets: `1024x1152x832` median `1.0219x`, bootstrap95
`[1.0191,1.0365]`, B-faster `155/201`, holdout median `1.0243x`; and
`1088x1152x832` median `1.0185x`, bootstrap95 `[1.0092,1.0263]`, B-faster
`145/201`, holdout median `1.0213x`.

The guard rows kept the rule exact: `832x1152x832` was neutral at median
`1.0008x`, `960x1152x832` regressed at `0.9890x`, `1152x1152x832` was
neutral/noisy at `0.9929x`, `1024x1152x960` regressed at `0.9772x`,
`1024x1152x1152` stayed noisy at `1.0364x` with weak CI/sign, and
`1024x1216x832` stayed neutral/noisy at `1.0054x`.

Correctness coverage adds `1024x1152x832` and `1088x1152x832`.

### 2026-05-27 local-uncommitted: 512x896 SME direct K-band accepted

The exact `512x896x1536` SME-direct route exposed a nearby gap at lower `K`.
The dispatcher now treats `512x896` as an SME direct-`B` band for
`832 <= k <= 1536`, rather than a single exact `k = 1536` exception.

Repeat-201, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-512x896-kband-sme.c` measured the
newly routed rows as strong wins: `512x896x832` median `1.1726x`,
bootstrap95 `[1.1677,1.2946]`; `512x896x960` `1.1686x`, bootstrap95
`[1.1465,1.2743]`; `512x896x1024` `1.1755x`, bootstrap95
`[1.1973,1.3391]`; `512x896x1152` `1.1817x`, bootstrap95
`[1.2074,1.3325]`; and `512x896x1280` `1.1924x`, bootstrap95
`[1.2213,1.3652]`. The already-routed `512x896x1536` was behavior-identical
in this comparison.

The same run kept the band bounded: `512x896x2048` stayed neutral/noisy at
median `0.9909x`, `512x960x1152` regressed at `0.9986x` with negative
mean-log CI, and `768x896x1152` stayed neutral/noisy at `1.0036x`.

Correctness coverage adds the five new `512x896` lower-`K` rows.

### 2026-05-27 local-uncommitted: m768/m1024 medium SME direct edge route accepted

The medium SME-direct extra-`N` band previously covered `832 <= m <= 960`
for `n = 1280..1472, k = 832..1152`. A subagent sweep found edge candidates at
`m = 768` and `m = 1024`; the accepted source change adds only the rows that
survived paired A/B and leaves the mixed edge rows on AMX.

Repeat-201, `iters=4` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-m768-m1024-direct-edge.c` validated
the accepted `m = 768` rows: `768x1280x832` median `1.1253x`,
`768x1280x960` `1.1343x`, `768x1344x832` `1.1082x`,
`768x1344x960` `1.0630x`, `768x1344x1152` `1.1059x`,
`768x1408x832` `1.1267x`, `768x1408x960` `1.0453x`,
`768x1408x1152` `1.1266x`, and `768x1472x832` `1.1186x`. The rejected
same-edge guards stayed neutral/noisy: `768x1280x1152` `0.9988x`,
`768x1472x960` `0.9989x`, and `768x1472x1152` `1.0026x`.

The accepted `m = 1024` rows measured `1024x1280x960` median `1.1084x`,
`1024x1344x832` `1.0832x`, `1024x1472x832` `1.0950x`, and
`1024x1472x960` `1.0826x` in the same repeat-201 run. Higher-iteration
confirmation (`repeat=301`, `iters=8`) kept the borderline rows positive:
`1024x1280x832` median `1.0772x`, bootstrap95 `[1.0509,1.0878]`;
`1024x1344x960` `1.0694x`, bootstrap95 `[1.0401,1.0825]`; and
`1024x1344x1152` `1.0881x`, bootstrap95 `[1.0546,1.0721]`. Rejected
`m = 1024` guards were neutral or negative: `1024x1280x1152` `0.9971x`,
`1024x1408x832` `1.0036x`, `1024x1408x960` `1.0018x`, and
`1024x1472x1152` `0.9981x`.

Correctness coverage adds the 16 accepted edge rows.

### 2026-05-27 local-uncommitted: exact 512x896x1536 SME direct route accepted

The medium audit found a strong one-shot exception at `512x896x1536`: routing
only that exact shape through the SME direct-`B` medium kernel beat the current
packed path without creating a usable threshold around it.

Focused repeat-301, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-512x896-sme.c` measured median
`1.2025x`, mean-log `1.2208x`, bootstrap95 `[1.1808,1.2652]`, B-faster
`247/301`, sign-p `1.19e-30`, holdout median `1.2028x`, holdout bootstrap95
`[1.1640,1.2811]`, holdout B-faster `126/151`, and holdout sign-p
`2.04e-17`.

The guard set did not justify broadening the gate: `512x896x1024` was neutral
at median `0.9991x`, `512x896x2048` was noisy at `1.0052x`,
`512x960x1536` was noisy at `1.0065x`, `512x1088x1536` was neutral at
`1.0019x`, and `768x896x1536` was noisy at `1.0083x`. Treat this as an exact
one-shot rule beside the existing `512x1024x1536` SME-direct exception.

Correctness coverage adds `512x896x1536`.

### 2026-05-27 local-uncommitted: exact 1024 square large-block rejected

A fresh square audit again made `1024x1024x1024` look weak on separate-process
medians, so a temp source added exact `1024^3` to
`cob_sgemm_amx_large_block_shape` without broadening the old `n >= 1024`
large-block threshold that had previously been rejected.

Repeat-301, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-square1024-largeblock.c` did not
clear the dispatch-gate bar: `1024x1024x1024` measured median `1.0023x`,
bootstrap95 `[0.9994,1.0180]`, B-faster `159/301`, sign-p `0.356`, and
holdout median `1.0109x`. Behavior-identical guards stayed neutral/noisy:
`768^3` `1.0003x`, `1280^3` `0.9980x`, and `1536^3` `0.9994x`. Revert the
temp exact gate; standalone square audit medians are not enough evidence here.

### 2026-05-27 local-uncommitted: n1152/n1216 k1536 packed fallback rejected

A medium audit showed noisy one-shot dips at `n = 1152/1216, k = 1536`, so a
temp source forced those widths away from the AMX strided-B route and through
the packed fallback. This mirrors some accepted packed-path decisions at higher
K, but it did not hold at `k = 1536`.

Repeat-201, `iters=8` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-n1152-n1216-k1536-packed.c` was a
hard rejection: `512x1152x1536` median `0.8324x`, `768x1152x1536` `0.8708x`,
`1024x1152x1536` `0.8994x`, `512x1216x1536` `0.8346x`,
`768x1216x1536` `0.8887x`, and `1024x1216x1536` `0.9051x`. Existing
`k = 2048` packed-route guards stayed neutral/noisy. Keep `k = 1536` on
strided-B for these widths.

### 2026-05-27 local-uncommitted: n768 packed fallback screen rejected

A compile-flag screen lowered `COB_SGEMM_AMX_STRIDED_B_MAX_N` from `832` to
`512`, forcing `n = 768` one-shot rows below the existing high-K large-block
gates away from AMX strided-B and through the packed fallback. Repeat-201,
`iters=8` A/B was a clear loss at the low-K rows: `512x768x1536` median
`0.8680x`, `768x768x1536` `0.8989x`, `1024x768x1536` `0.9123x`,
`512x768x2048` `0.8499x`, `768x768x2048` `0.8818x`, and `1024x768x2048`
`0.9082x`. The `k = 3072` rows were neutral because they already sit on the
large-block packed path. Keep the current strided-B coverage for `n = 768`
below the high-K gates.

### 2026-05-27 local-uncommitted: exact 768x512x1536 AMX fallback accepted

A medium audit pointed at noisy one-shot `n = 512, k < 2048` rows where the
current path packs B and then uses the SME packed-B kernel to avoid the AMX
strided-B conflict stride. A broad `COB_DISABLE_APPLE_SME=1` probe was too
coarse, but it showed one repeatable candidate: exact `768x512x1536`.

The source now skips the SME packed-B detour only for `768x512x1536`, letting
the existing AMX packed fallback consume the already-packed B panel. Focused
repeat-301, `iters=8` A/B against
`/private/tmp/cob-next-audit/gemm-baseline-n512-amx-exact.c` measured the
target at median `1.0602x`, bootstrap95 `[1.0153,1.1012]`, B-faster
`179/301`, sign-p `0.00121`, and holdout median `1.0662x`. Neighboring guards
`512x512x1536`, `1024x512x1536`, `1280x512x1536`, `768x512x1024`, and
`768x512x2048` were behavior-identical or noisy under the exact gate.

### 2026-05-27 local-uncommitted: m96/m128 k1024 fallback recheck rejected

A noisy skinny claim audit made the m96/m128 `k >= 1024` SME B-reuse rows look
weak against packed-path ceilings, so the old route decision was rechecked
directly. The candidate disabled
`COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N` by compiling it to a very high
threshold, forcing those rows back through the later one-shot fallback path.

Repeat-201, `iters=8` paired A/B was a clear rejection: `96x4096x1024` median
`0.6320x`, `96x8192x1024` `0.6212x`, `96x4096x2048` `0.6219x`,
`128x4096x1024` `0.6601x`, `128x8192x1024` `0.6562x`, and
`128x8192x4096` `0.7076x`, all with B-faster `0/201`. Keep the existing SME
B-reuse route for m96/m128 `k >= 1024`.

### 2026-05-27 local-uncommitted: direct-NC 768-column retune rejected

The accepted high-width m64 direct-NC route uses 1024-column chunks. A
compile-flag probe tested `COB_SGEMM_M64_SME_DIRECT_NC=768`, since older notes
only ruled out `512` on the existing direct-NC routes. Repeat-201, `iters=4`
A/B was neutral/noisy across most of the band: `64x2560x12288` median
`0.9981x`, `64x3072x4096` `1.0022x`, `64x3072x7168` `0.9987x`,
`64x3584x7168` `0.9996x`, `64x3712x7168` `1.0003x`, `64x4032x7168`
`0.9996x`, and `64x4032x8192` `1.0036x`. The tempting audit row
`64x3840x7168` leaned negative at `0.9953x` with a negative holdout. Keep the
current 1024-column direct-NC chunk.

### 2026-05-27 local-uncommitted: long-k512 NC and unroll probes rejected

The remaining `64x32768x512` calibration gap was tested with route-local
changes instead of broad reuse knobs. A temp source first added a default-off
`m = 64, k = 512` NC override for the long-`n` B-reuse path. With
`NC = 1024`, repeat-201, `iters=8` paired A/B regressed `64x4096x512` to
`0.9334x`, left `64x16384x512` neutral at `1.0001x`, and measured
`64x32768x512` at `0.9901x`. With `NC = 256`, the low widths were only
weak/noisy positive, `64x16384x512` regressed to `0.9872x`, and
`64x32768x512` stayed neutral at `0.9999x`.

A second temp source added compile-flagged 2x `p`-loop unrolling to the two
non-tuple B64 helpers used by the long-k512 route:
`cob_sgemm_16x64_sme_strided_b_pack_b32` and
`cob_sgemm_16x64_sme_from_packed_b64`. Repeat-201, `iters=8` A/B with the
unroll enabled measured `64x4096x512` `0.9948x`, `64x8192x512` `0.9994x`,
`64x16384x512` `0.9767x`, and `64x32768x512` `0.9936x`. Shared-helper guards
were also not useful: `96x4096x512` regressed to `0.9942x`, while the
remaining m96/m128 rows stayed neutral/noisy. Revert the temp source changes.

### 2026-05-27 local-uncommitted: tuple-prefetch distance retune rejected

Fresh compile-flag A/B checks tried moving
`COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE` away from the accepted value `16`.
The macro is shared by the exact `n = 4096, k >= 7168` B-reuse path and the
newer selected wide prefetched tuple-pack paths, so the guard set covered all
of those routes.

Distance `24` regressed the `n = 4096` target region: repeat-201, `iters=4`
measured `64x4096x7168` median `0.9493x`, `64x4096x8192` `0.9610x`, and
`64x4096x12288` `0.9857x`. Wide guards were neutral/noisy except a weak
`64x4160x2048` positive line.

Distance `8` was also not shippable: repeat-201, `iters=4` measured the
`n = 4096` rows at `0.9955x`, `0.9935x`, and `0.9959x`, while
`64x4160x2048` regressed to `0.9655x`. `64x7168x8192` showed a noisy positive
median at `1.0728x`, but its holdout reversed to `0.9725x`. Keep the current
distance `16`.

### 2026-05-27 local-uncommitted: long-k512 source-B pack prefetch rejected

A temp source added source-B software prefetching to the existing non-tuple
`cob_sgemm_16x64_sme_strided_b_pack_b32` helper used by the long-`n`,
`k = 512` B-reuse path. This tested prefetching the original B stream while
keeping the long-k512 route on the non-tuple helper, avoiding the already
rejected tuple-helper swap.

Paired repeat-201, `iters=8` A/B against
`/private/tmp/cob-next-audit/gemm-baseline-k512-pack-prefetch.c` measured
`64x4096x512` median `0.9887x`, `64x8192x512` `1.0011x`,
`64x16384x512` `0.9199x`, `64x32768x512` `0.8072x`,
`96x4096x512` `0.9241x`, `96x8192x512` `0.9969x`,
`128x4096x512` `0.9983x`, and `128x8192x512` `0.9996x`. The source change was
reverted because the larger m64 k512 widths and the first m96 guard regressed
hard while the remaining shapes were neutral/noisy.

### 2026-05-27 local-uncommitted: exact n24576 k1536 prefetch accepted

The long-wide `64x24576x1536` MpGEMM stock row was still close enough to be
worth probing after the `n = 7168` prefetch work. A broad `n >= 24576,
k = 1536` source-B pack prefetch rule was rejected because it regressed
`64x32768x1536` hard in the first repeat-201 run (`0.8749x`). Narrowing the
rule to exact `n = 24576, k = 1536` kept the useful target and avoided that
tail risk.

Focused repeat-301, `iters=4` validation against
`/private/tmp/cob-next-audit/gemm-baseline-longwide-k1536-prefetch.c` measured
`64x24576x1536` median `1.0617x`, bootstrap95 `[1.0178,1.0818]`, B-faster
`201/301`, sign-p `5.97e-09`, and holdout median `1.0489x`. Neighboring widths
`64x24064x1536`, `64x24320x1536`, `64x24832x1536`, `64x25088x1536`, the
`64x32768x1536` tail, and `64x24576x2048` were neutral/noisy under the exact
gate.

### 2026-05-27 local-uncommitted: packed-B reuse prefetch rejected

A follow-up tested software prefetching inside
`cob_sgemm_16x64_sme_from_packed_b64_tuple`, the helper that consumes already
packed B for the second through fourth 16-row panels in the m64 reuse path.
This was intended to complement the accepted source-B pack prefetching, but it
did not hold up. Repeat-201, `iters=4` paired A/B against
`/private/tmp/cob-next-audit/gemm-baseline-packed-reuse-prefetch.c` was
neutral/noisy or worse across the main reuse rows: `64x4096x7168` median
`0.9969x`, `64x7168x2048` `1.0062x` with neutral holdout, `64x7168x16384`
`1.0104x` with neutral holdout, `64x24576x1536` `1.0061x` with neutral
holdout, `64x8192x2048` `0.9971x`, `64x8192x1024` `1.0160x` with neutral
holdout, `96x4096x1024` `1.0195x` with neutral holdout, and
`128x8192x1024` `1.0124x` with neutral holdout. Revert the temp source change;
keep prefetching limited to selected source-B pack helpers.

### 2026-05-27 local-uncommitted: exact n7168 high-K prefetch accepted

Follow-up after the wide `k = 2048` prefetch change tested whether the same
tuple-prefetch pack helper should cover wider high-K B-reuse rows. A broad
`k >= 8192` rule was not good enough because `n = 8192` was neutral/noisy, but
an exact `n = 7168, k >= 8192` rule held against neighboring guards. The source
now uses the prefetched helper for wide `m = 64` B-reuse when `k == 2048`, and
also when `n == 7168 && k >= 8192`.

The first broad repeat-201, `iters=2` A/B measured strong wins at
`64x7168x8192` median `1.0671x` and `64x7168x16384` `1.0774x`, but
`64x8192x8192` was `0.9800x` and `64x8192x16384` was only weak/noisy. The
narrowed repeat-201, `iters=2` validation kept the target wins:
`64x7168x8192` median `1.0850x`, bootstrap95 `[1.0436,1.1026]`, holdout
median `1.0786x`; and `64x7168x16384` median `1.1043x`, bootstrap95
`[1.0726,1.1276]`, holdout median `1.1321x`. Guards `64x6144x8192`,
`64x8192x8192`, `64x6144x16384`, `64x8192x16384`, `64x7168x4096`, and
`64x7168x2048` stayed neutral/noisy under the exact rule.

### 2026-05-26 local-uncommitted: wide m64 k2048 prefetch accepted

The `64x7168x2048` MpGEMM calibration gap did not respond to wide B-reuse
`NC/KC` retuning, but did respond to reusing the existing tuple-prefetch pack
helper. The source now uses `cob_sgemm_16x64_sme_strided_b_pack_b32_tuple_prefetch2`
for the wide `m = 64, k = 2048` B-reuse route, while keeping neighboring
`k = 1536` and `k = 4096` paths on the old tuple pack helper.

Broad repeat-201, `iters=4` A/B against
`/private/tmp/cob-next-audit/gemm-baseline-wide-prefetch.c` showed the target
moving in the right direction: `64x6144x2048` median `1.0833x`,
`64x7168x2048` `1.0617x`, and `64x5120x2048` `1.0315x`, while `64x8192x2048`
was neutral and `64x7168x1536` / `64x7168x4096` stayed off the new path.
Focused repeat-301, `iters=8` validation was stronger across the wide
`k = 2048` band: `64x5120x2048` median `1.0894x`, `64x6144x2048` `1.0845x`,
`64x7168x2048` `1.0978x`, and `64x8192x2048` `1.0418x`, all with positive
holdouts. Edge repeat-201, `iters=8` also showed `64x4160x2048` at median
`1.1876x` and `64x4096x2048` positive/noisy at `1.0271x`; `k = 1536` and
`k = 3072` guards were neutral. An isolated repeat-31 one-shot bench after the
source change measured `64x7168x2048` at median `1076.82 GF/s`, best
`1120.48 GF/s`; this is much closer to MpGEMM's earlier `1095.65 GF/s` stock
result and beats it on best sample, but still leaves a small median gap in that
separate-process comparison.

The existing `COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE=16` stayed the best
overall distance for this shared helper. Distance `8` regressed the target and
low/mid `k = 2048` rows despite helping `8192`; distance `32` helped
`4160/5120` but regressed `8192` and the existing `64x4096x7168` prefetch path.

### 2026-05-26 local-uncommitted: wide m64 NC/KC follow-up rejected

After `137b02c`, a fresh MpGEMM calibration refresh at
`/private/tmp/cob-mpgemm-after-n4096-prefetch-20260526` showed COB beating the
stock MpGEMM skinny rows on five of six medians in the current local run:
`64x2112x7168` `1431.14 GF/s`, `64x24576x1536` `1076.61 GF/s`,
`64x32768x512` `885.20 GF/s`, `64x7168x16384` `1035.00 GF/s`,
`64x4096x7168` `1083.34 GF/s`, and `64x7168x2048` `1003.77 GF/s`.
The remaining obvious MpGEMM calibration gap is still `64x7168x2048`
against MpGEMM's earlier `1095.65 GF/s` stock result.

Wide B-reuse chunk probes did not close that gap. `COB_SGEMM_M64_SME_WIDE_KC`
at `2048`, `1536`, and `768` was neutral or worse on `64x7168x2048` and
nearby guards. `COB_SGEMM_M64_SME_REUSE_NC=256` helped some low-edge
`k = 1536` rows and `64x8192x2048`, but regressed the target
`64x7168x2048`; `NC=768` was neutral on the target and regressed
`64x8192x2048` plus `64x7168x1536`.

A source-narrowed `k = 1536` tight-NC candidate also stayed uncommitted. The
broad `4160 <= n <= 7168` rule was only consistently useful at the low edge,
and an exact `64x4160x1536` repeat-301, `iters=8` validation was only a small
win: median `1.0138x`, bootstrap95 `[1.0075,1.0250]`, B-faster `171/301`,
sign-p `0.021`, and holdout median `1.0151x`. That is too weak for another
exact dispatch exception.

### 2026-05-26 local-uncommitted: n4096 pack-B prefetch split accepted

Focused high-K `m = 64, n = 4096` probes found one small route-local tweak
worth keeping. The existing B-reuse tuple-prefetch path now uses a separate
`COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE=16`, while direct streaming-B
keeps `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE=32`.

The accepted split was first identified with a compile-flag sweep, then
validated as a source-narrowed change against the saved baseline
`/private/tmp/cob-next-audit/gemm-baseline-n4096-kc.c`. Broad repeat-201,
`iters=4` A/B measured target speedups of `64x4096x7168` median `1.0148x`,
`64x4096x8192` `1.0063x`, and `64x4096x12288` `1.0113x`; guards
`64x4160x7168`, `64x5120x7168`, and `64x2112x7168` were neutral because the
new constant only fires on the exact `n = 4096, k >= 7168` reuse path. Heavier
target-only repeat-301, `iters=8` confirmation stayed positive:
`64x4096x7168` median `1.0035x`, `64x4096x8192` `1.0033x`, and
`64x4096x12288` `1.0045x`, with positive mean-log/bootstrap and holdout on all
three rows.

Rejected alternatives from the same pass: forcing `n = 4096, k >= 7168` from
B-reuse to direct-NC regressed `64x4096x7168/8192/12288` to about
`0.79..0.84x`; disabling the special `n = 4096` tuple-prefetch path regressed
the same region at about `0.93..0.97x`; `COB_SGEMM_M64_SME_REUSE_NC=256` was
neutral, `COB_SGEMM_M64_SME_REUSE_NC=1024` was not a target win and hurt the
`64x4160x7168` guard, `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE=64` regressed the
target/guard set, and using `KC=1024` for this reuse path was neutral to worse.

### 2026-05-26 local-uncommitted: MpGEMM calibration refreshed

The MpGEMM checkout was refreshed at `/private/tmp/mpgemm_latest`, commit
`8d83011`. No `LICENSE*` or `COPYING*` file was found within depth 2, so MpGEMM
still stays outside the licensed/open-source claim set, but it remains the main
source-available calibration target. Its README states that MpGEMM is an
SME-focused GEMM library and reports an average `1.23x` speedup over Apple
Accelerate on the paper's Apple M4 Pro workloads.

Build and correctness passed with MpGEMM's stock Makefiles:
`make -C /private/tmp/mpgemm_latest/src`,
`make -C /private/tmp/mpgemm_latest/benchmark correct.x singlePerformance.x`,
and `/private/tmp/mpgemm_latest/benchmark/correct.x`. The stock
`singlePerformance.x` FP32 row-major results were:
`64x2112x7168` `1283.29 GF/s`, `64x24576x1536` `1031.78 GF/s`,
`64x32768x512` `880.48 GF/s`, `64x7168x16384` `1018.18 GF/s`,
`64x4096x7168` `1076.82 GF/s`, and `64x7168x2048` `1095.65 GF/s`.

COB one-shot repeat-31 on the same stock shapes measured:
`64x2112x7168` median `1308.42 GF/s`, `64x24576x1536` median
`720.10 GF/s` with best `998.52 GF/s`, `64x32768x512` median
`843.80 GF/s`, `64x7168x16384` median `980.27 GF/s`,
`64x4096x7168` median `942.11 GF/s`, and `64x7168x2048` median
`1025.12 GF/s`. An isolated repeat-101 rerun improved `64x24576x1536` to
median `960.22 GF/s`, best `1025.87 GF/s`, and measured `64x4096x7168` at
median `948.77 GF/s`, best `1062.21 GF/s`.

Follow-up probes from the same pass found no code change to keep. For exact
`64x4096x7168/8192/12288`, forcing direct-NC instead of the existing high-K
B-reuse path was a hard regression: medians `0.8434x`, `0.8421x`, and
`0.7918x`. For exact `n = 1024`, forcing the packed-AMX fallback was a hard
regression around `0.56..0.62x`, and forcing B-reuse with `NC=256` or
`NC=1024` regressed all tested high-K rows. The reusable output helper
`tools/mpgemm_calibration.sh` was added so this source-available baseline can
be refreshed without hand-editing benchmark files.

### 2026-05-26 local-uncommitted: m64 post-route follow-up probes rejected

After `da3a87e`, a fresh route-aware one-shot grid over `m = 64` and
`n = 1024..4096`, `k = 1536..16384` showed the remaining weak area was still
large-K skinny direct routing, especially high-K `n = 1024`, `64x2112x7168`,
and a few noisy direct-NC rows. The grid output was
`/private/tmp/cob-m64-next/oneshot-route.csv`; a small full benchmark subset in
`/private/tmp/cob-m64-next/target-full.csv` showed `64x1024x7168` could still
trail Accelerate in a noisy repeat-7 run, so the next probes focused there.

Exact `n = 1024` direct-NC chunking with `COB_SGEMM_M64_SME_DIRECT_NC=512` was
not shippable. It gave weak medians at the low edge but no clean holdout/sign
support: `64x1024x4096` median `1.0197x`, holdout `1.0170x`, sign-p `0.0109`;
`64x1024x7168` median `1.0170x`, holdout `1.0164x`, sign-p `0.00184`; and
`64x1024x8192` was neutral. Applying `NC=512` to existing direct-NC routes was
also neutral or negative except a noisy `64x3584x7168` median `1.0262x` with
weak holdout sign support. Keep the current direct-NC chunk size and gates.

Narrow high-K `n = 1024` B-reuse was also rejected. Routing `n = 1024` through
reuse from `k >= 8192` regressed `64x1024x8192` to median `0.9823x` and
`64x1024x12288` to `0.9594x`; only `64x1024x16384` looked positive at median
`1.0860x`, but sign-p `7.05e-05` and holdout sign-p `0.00265` were far below
the acceptance bar for another exact gate.

The `n = 1024` K-chunk probe was not stable. A compile-flag run with
`COB_SGEMM_SKINNY_SME_LARGE_KC=512` looked promising at `64x1024x6144`
(`1.0963x`, holdout `1.0900x`, sign-p `5.15e-19`) and `64x1024x8192`
(`1.0756x`, holdout `1.0857x`, sign-p `5.57e-13`), but it regressed the
low-width `k = 2048` guards because the flag was global. A source-narrowed
`6144..8192` gate did not reproduce, and a focused exact `64x1024x6144`
repeat-401 rerun collapsed to median `1.0015x`, holdout `0.9994x`, sign-p
`0.92`. Leave `n = 1024` on the existing `KC=1024` direct route.

Two other route probes were rejected. For exact `64x1280x1536`, forcing the
AMX chunked packed-B path was a hard regression at median `0.5255x`, even
though a noisy full benchmark subset showed a small Accelerate gap. For
`64x2112x7168` and neighboring high-K points, forcing the larger `KC=1024`
direct route regressed the target: `64x2112x7168` median `0.9635x` and
`64x2112x8192` `0.9684x`. These results reinforce that the remaining gap is
unlikely to fall to another simple dispatch or cache-blocking gate.

### 2026-05-26: exact 2560 m64 route add-ons accepted

Follow-up calibration after the wide-reuse/direct-NC pass found two exact
`n = 2560` wins. The final source routes exact `64x2560x3072` through the
existing SME B-reuse path, and routes `m = 64, n = 2560, k >= 12288` through
the direct-NC streaming-B path. Benchmark route labels mirror both gates, and
correctness coverage now covers 135 GEMM shapes.

The accepted reuse point was exact, not a smooth band. With only
`64x2560x3072` routed, repeat-201, `iters=3` measured median `1.0799x`,
bootstrap95 `[1.0647,1.1109]`, B-faster `154/201`, sign-p `1.76e-14`, and
holdout median `1.0885x`. A wider `2304..2816, k = 3072` probe rejected the
neighbors hard: `64x2304x3072` median `0.9154x`, `64x2432x3072` `0.9126x`,
`64x2496x3072` `0.8126x`, `64x2624x3072` `0.8370x`, and
`64x2816x3072` `0.9056x`; `64x2880x3072` was neutral and unchanged.

The accepted direct-NC point is also exact in `n`, and is limited to the
validated higher-K threshold. Focused repeat-401, `iters=3` checks measured
`64x2560x12288` median `1.0516x`, bootstrap95 `[1.0608,1.1141]`, B-faster
`280/401`, sign-p `1.28e-15`, and holdout median `1.0444x`; and
`64x2560x16384` median `1.0398x`, bootstrap95 `[1.0428,1.0619]`, B-faster
`363/401`, sign-p `1.12e-67`, and holdout median `1.0410x`. Lower high-K
points were left out: `64x2560x5120` had weak full-run support despite a
positive holdout, `64x2560x6144` was inconsistent across cold runs,
`64x2560x7168` did not clear the usual holdout-median bar, and
`64x2560x8192` stayed positive on some runs but did not survive the final
no-change guard cleanly enough to route.

Rejected follow-up probes from this continuation: `COB_SGEMM_M64_SME_REUSE_NC`
at `256` was neutral/noisy and hurt `64x4096x2048`; `NC=1024` was a clear loss.
`COB_SGEMM_M64_SME_WIDE_K1536_KC=1024` regressed low-edge `k = 1536` shapes,
while `512` had tempting but weak/noisy exact medians. `WIDE_KC=512` and
`1536` for `k >= 2048` failed cold reruns or regressed guard widths. A broader
direct-NC high-K width band around `n = 2560` was not stable enough to ship;
keep the source exact unless fresh paired evidence changes that.

### 2026-05-26: m64 SME wide-reuse and 3072 direct-NC routes accepted

This session changed only m64 SME routing. A new
`cob_sgemm_m64_sme_wide_reuse_shape` helper routes the existing SME B-reuse
path for `m = 64` when `k >= 1024, n >= 4160`; when `n == 4096` and
`1024 <= k < 7168`, leaving the existing `n == 4096, k >= 7168` path
unchanged; exact `n == 2560, k == 3072`; when `3072 <= n < 4096` at
`k == 1024`; and when `n == 3584` with `1024 <= k < 7168`. The existing
direct-NC helper now also covers exact `n == 2560, k >= 12288` and exact
`n == 3072, k >= 4096`, in addition to the previous
`3584 <= n < 4096, k >= 7168` band. Benchmark route labels mirror source
dispatch, and correctness coverage now covers 135 GEMM shapes.

Final direct-NC validation accepted the exact `n = 3072` high-K pocket. The
repeat-161, `iters=2` final check for `64x3072x4096` measured median
`1.0368x`, bootstrap95 `[1.0514,1.0871]` by mean-log interval, B-faster
`130/161`, sign-p `1.27e-15`, and holdout median `1.0399x`. Earlier
repeat-161 checks were stronger at higher K: `64x3072x7168` median `1.1535x`
with holdout `1.1577x`, `64x3072x8192` `1.4175x` with holdout `1.3976x`, and
`64x3072x12288` `1.3450x` with holdout `1.2799x`.

Final reuse validation showed the low/wide rule was route-worthy:
`64x3072x1024` median `1.5952x`, holdout `1.6403x`, sign-p `4.76e-43`;
`64x3328x1024` `1.7308x`, holdout `1.5774x`, sign-p `1.16e-34`;
`64x3584x2048` `1.1019x`, holdout `1.0953x`, sign-p `2.98e-26`;
`64x3840x1024` `1.6749x`, holdout `1.6025x`, sign-p `6.84e-49`;
`64x4096x2048` `1.1129x`, holdout `1.1390x`, sign-p `1.77e-33`;
`64x4160x1024` `1.8372x`, holdout `1.8675x`, sign-p `6.84e-49`; and
`64x5120x7168` `2.0563x`, holdout `2.0965x`, sign-p `6.84e-49`.

Guard validation kept the accepted shape rules narrow. `64x3072x3072` stayed
neutral/negative at median `0.9857x` with holdout `0.9996x`, so it was dropped. `64x3328x2048`
was not granted an exact gate despite median `1.0142x` and holdout `1.0473x`.
After the unstable exact 3712 route was removed, `64x3712x4096` stayed neutral
at median `1.0080x`, holdout `1.0067x`, and `64x3712x5120` stayed neutral at
`1.0002x`, holdout `1.0119x`. `64x3840x3072` stayed unchanged at median
`0.9988x`, holdout `1.0079x`; `64x4096x7168` stayed on the existing high-K
reuse path at median `1.0039x`, holdout `1.0039x`.

Verification passed `make test` across 135 shapes, `make all`,
`git diff --check`, `cmake --build build-cmake` with only existing
SDK/Accelerate warnings, and `ctest --test-dir build-cmake --output-on-failure`.
Route smoke showed the expected labels: `64x2560x3072` `sme_skinny_reuse`,
`64x2560x8192` `sme_skinny_strided`, `64x2560x12288`
`sme_skinny_strided_nc`, `64x2496x3072` `sme_skinny_strided`,
`64x2624x8192` `sme_skinny_strided`, `64x3072x1024`
`sme_skinny_reuse`, `64x3072x3072` `sme_skinny_strided`,
`64x3072x4096` `sme_skinny_strided_nc`, `64x3584x2048`
`sme_skinny_reuse`, `64x3712x4096` `sme_skinny_strided`,
`64x3840x1024` `sme_skinny_reuse`, `64x4096x2048`
`sme_skinny_reuse`, `64x4096x7168` `sme_skinny_reuse`,
`64x4160x1024` `sme_skinny_reuse`, and `64x5120x7168`
`sme_skinny_reuse`.

### 2026-05-26 local-uncommitted: m64 SME route probes rejected

Several m64 probes from the same session were rejected before the final narrow
reuse and direct-NC rules landed. Disabling SME with
`-DCOB_DISABLE_APPLE_SME=1` was a hard regression on m64 high-K shapes:
`64x1024x7168` median `0.5442x`, `64x1088x7168` `0.6219x`,
`64x2112x7168` `0.5759x`, and `64x4096x7168` `0.4176x`.

Raising `COB_SGEMM_SKINNY_SME_LARGE_KC` to `2048` was neutral or regressive:
`64x1024x7168` median `0.9024x`, `64x1088x7168` `1.0057x`,
`64x2112x7168` `1.0077x` with weak support, and `64x4096x7168` `0.9976x`.
Direct-stream prefetch for exact `n = 1024, k >= 4096` regressed
`64x1024x7168` to `0.9576x`, with guards unchanged. Forcing high-K
`n = 1024` through B-reuse also regressed: `64x1024x4096` median `0.9074x`
and `64x1024x7168` `0.8868x`.

A broad direct-NC threshold of `n >= 2048, k >= 7168` had one useful exact
`n = 3072` pocket, but was neutral/noisy elsewhere; the final source kept only
exact `n = 3072` plus the previous `3584..4032` high-K direct-NC band. Broad
reuse for `n = 3712, k >= 3072` was unstable: early `4096/5120` runs looked
positive, but mixed final validation turned exact `3712x4096` and
`3712x5120` neutral until the route was removed, and `3712x6144` was a clear
focused-rerun regression at median `0.9424x`. Do not route these 3712 shapes
without fresh cold proof.

### 2026-05-06 local-uncommitted: m64 prefetch-boundary branch hoist rejected

A temp source in `/private/tmp/cob_prefetch_guard_exp` split the prefetched
SME streaming-B K loops into a main loop with unconditional B prefetches and a
short no-prefetch tail. This removed the `p + distance < k` check from the hot
loop in both `cob_sgemm_16x64_sme_strided_b32_prefetch2` and the tuple
B-pack-prefetch helper, targeting the map/dispatch pressure seen in the m64
large-K counter pass without changing any route gates.

Correctness passed across 127 shapes, but paired validation was neutral/noisy:
`64x1088x7168` median `0.9958x`, `64x2112x7168` `1.0014x` with holdout
`0.9949x`, `64x3328x12288` `1.0000x`, `64x3712x12288` `0.9985x`,
`64x4032x8192` `0.9984x`, `64x4096x7168` `0.9965x`, `64x4096x8192`
`0.9945x`, and guard `384x1280x1536` `1.0026x`. Several high-variance shapes
auto-extended to 61 repeats but still lacked useful sign-test or holdout
support. Leave the compact in-loop prefetch guard in source.

### 2026-05-06 local-uncommitted: medium SME B-panel-first traversal rejected

The broad B-panel-first loop-order probe for the SME direct-B kernel was not
shippable globally: square and lower-width medium shapes were neutral or
regressed, and the m64 large-K direct shapes stayed noisy. The useful region was
initially only the already-routed exact medium pair `384x1280x1536` and
`512x1280x1536`, but the exact gated version did not survive cold validation.

The exploratory broad source first looked promising in repeat-101 validation:

- `384x1280x1536`: median `1.1108x`, bootstrap95 `[1.0999,1.1358]`,
  B-faster `95/101`, sign-p `1.07e-21`, holdout median `1.1057x`.
- `512x1280x1536`: median `1.0718x`, bootstrap95 `[1.0753,1.1091]`,
  B-faster `90/101`, sign-p `1.42e-16`, holdout median `1.0457x`.

After implementing a narrow candidate and comparing against clean `2a79a71`,
the result weakened: `384x1280x1536` fell to median `1.0138x` with holdout
median `1.0059x`, and `512x1280x1536` reversed to median `0.9941x`. The code
was removed instead of shipping another brittle exact gate.

Rejected broadening evidence from the same probe: `384x384x384`,
`768x768x768`, `960x896x960`, and `384x512x3072` did not show an acceptable
win, while `512x512x3072` and `832x1472x1152` failed the acceptance floor on
repeat-61. Keep the current A-panel-first SME direct-B traversal.

### 2026-05-06 local-uncommitted: m64 SME prefetch locality rejected

The m64 direct-prefetch path was retested with only the `__builtin_prefetch`
locality hint changed, leaving the accepted distance, issue rate, and route
gates unchanged.

- Locality `0` was a hard regression on the prefetched m64 large-K direct
  shapes: `64x1088x7168` median `0.4779x`, `64x2112x7168` `0.3817x`,
  `64x3328x12288` `0.4611x`, `64x3712x12288` `0.3960x`, and
  `64x4032x8192` `0.3517x`.
- Locality `2` was neutral/noisy: `64x2112x7168` median `1.0030x`,
  `64x3328x12288` `0.9924x`, `64x3712x12288` `1.0040x`,
  `64x4032x8192` `1.0022x`, and `64x4096x7168` `1.0015x`, all without
  useful sign-test or holdout support.

Keep `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE=32` with locality `1`.

### 2026-05-06 local-uncommitted: SME direct epilogue branch hoist rejected

A temp source hoisted the `accumulate` branch out of the per-row epilogue loop
in `cob_sgemm_16x64_sme_strided_b32` and its prefetched variant. Correctness
passed, but paired repeat-31 validation did not show a usable win:
`64x1088x7168` median `0.9958x`, `64x2112x7168` `0.9958x`,
`64x4032x8192` `0.9878x`, `64x4096x7168` `1.0003x`,
`384x1280x1536` `0.9957x`, and `832x1472x1152` `1.0021x`.

The remaining m64 large-K gap is not explained by the row epilogue branch in
the current C intrinsic kernels. Leave the compact epilogue as-is.

### 2026-05-06 local-uncommitted: native CPU compile flag rejected

The candidate side was compiled with `COB_AB_B_FLAGS=-mcpu=native` to check
whether local M5 code generation alone would improve the current kernels.
Repeat-31 paired validation was neutral/noisy across the tested set:
`64x2112x7168` median `0.9891x`, `64x4096x7168` `1.0154x`,
`384x1280x1536` `1.0301x` with weak holdout, `512x1280x1536` `0.9804x`,
`1280^3` `0.9974x`, and `2048^3` `1.0066x`.

Do not make `-mcpu=native` a default build flag based on current evidence.

### 2026-05-06 local-uncommitted: direct SME K16 unroll pragma rejected

A temp source added `#pragma clang loop unroll_count(16)` to the plain and
prefetched `cob_sgemm_16x64_sme_strided_b32` K loops. This was a cheap check
inspired by MpGEMM's hand-scheduled K16 assembly blocks, without changing route
gates or kernel layout. Correctness passed, but repeat-31 paired validation did
not support the change: `64x1088x7168` median `1.0020x`,
`64x2112x7168` `0.9993x`, `64x3328x12288` `1.0093x`,
`64x3712x12288` `1.0270x` with weak sign-test support,
`64x4032x8192` `0.9675x`, `64x4096x7168` `0.9900x`, and
`384x1280x1536` hard-regressed to `0.9034x`.

The compiler pragma is not an adequate substitute for a real hand-scheduled
SME kernel.

### 2026-05-06: packed-AB paired A/B mode added

The paired A/B harness now supports `COB_AB_MODE=packed-AB`, so future
fully-prepacked API changes can be measured in the same interleaved process as
one-shot and packed-B changes. The mode packs A and B once for each side, times
`cob_sgemm_rowmajor_packed_ab`, keeps the same checksum guard, and still
reports paired median ratio, bootstrap interval, sign-test p-value, and
split-half holdout.

Smoke validation with `COB_AB_MODE=packed-AB` against identical source compiled
and ran on `128` and `64x2048x1536`; checksums matched and output mode labels
reported `packed-ab`. Large-shape no-change ratios remained noisy, as expected,
but the setup/timing path is now available for real packed-AB candidates.

### 2026-05-06: m64 high-K direct SME N-chunking accepted

The remaining m64 large-K counter evidence pointed at dispatch/scheduling
pressure rather than simple B-memory waits. A bounded structural probe split
the direct SME streaming-B route into 1024-column chunks only for
`3584 <= n < 4096, k >= 7168`, so each B slab is reused across the four
16-row A panels before moving to the next slab. This avoids the B-pack reuse
path that previously regressed `64x2112x7168` and leaves `n = 4096` on its
existing B-reuse route.

The paired harness had first regressed after the packed-AB API addition because
`tools/paired_ab_bench.sh` did not rename the new packed-A public symbols. The
script now renames `cob_sgemm_pack_a`, `cob_sgemm_free_packed_a`, and
`cob_sgemm_rowmajor_packed_ab` on both A and B sides, restoring same-process
A/B builds.

Final paired validation against clean `e9466e1` source with the fixed harness:

- Guard `64x3520x7168`: median `1.0069x`, bootstrap95 `[0.9817,1.0268]`,
  sign-p `1`, holdout median `1.0149x`; unchanged.
- Guard `64x4096x7168`: median `0.9978x`, bootstrap95 `[0.9756,1.0104]`,
  sign-p `1`, holdout median `0.9954x`; unchanged on the existing reuse route.
- Accepted band: `64x3584x7168` median `1.0865x`, bootstrap95
  `[1.0671,1.1000]`, sign-p `1.46e-20`, holdout `1.0842x`;
  `64x4032x7168` median `1.0500x`, `[1.0443,1.0704]`, sign-p `7.85e-12`,
  holdout `1.0500x`; `64x3584x8192` median `1.0599x`,
  `[1.0496,1.0733]`, sign-p `1.08e-15`, holdout `1.0569x`;
  `64x3712x12288` median `1.0330x`, `[1.0261,1.0494]`,
  sign-p `1.54e-12`, holdout `1.0330x`.
- A focused repeat-161 rerun with `COB_AB_ITERS=2` cleared the noisy middle
  width: `64x3712x7168` median `1.0492x`, bootstrap95 `[1.0489,1.0738]`,
  B-faster `146/161`, sign-p `3.76e-28`, holdout `1.0567x`.

Route smoke after rebuilding labels `64x3520x7168` as
`sme_skinny_strided`, `64x3584x7168`, `64x3712x7168`, and
`64x4032x8192` as `sme_skinny_strided_nc`, and `64x4096x7168` as
`sme_skinny_reuse`. Correctness coverage adds representative
`64x3584x7168` and `64x4032x8192` shapes.

Follow-up rejected probes:

- Adding exact `64x1024x512` to the AMX skinny chunk packer was a hard
  regression against the current AMX strided-B path: repeat-161 with
  `COB_AB_ITERS=8` measured median `0.5557x`, bootstrap95 `[0.5510,0.5619]`,
  B-faster `0/161`, sign-p `6.84e-49`.
- Forcing exact `64x4096x2048` away from SME direct into one-shot packed AMX
  also regressed: repeat-101 median `0.6662x`, bootstrap95
  `[0.6556,0.7026]`, B-faster `3/101`, sign-p `1.36e-25`.
- Prefetching exact `64x1024x1536` in the SME streaming-B route regressed:
  repeat-161 with `COB_AB_ITERS=6` measured median `0.8744x`, bootstrap95
  `[0.8711,0.9096]`, B-faster `29/161`, sign-p `6.7e-17`.

The skinny audit default was updated to include representative
`sme_skinny_strided_nc` shapes, so future claim audits cover this route without
requiring `COB_AUDIT_SKINNY_EXTRA`.

### 2026-05-06: small-A packed-B B-panel traversal accepted

The public packed-B AMX path now has a small-A B-panel-outer traversal for the
wide skinny packed-B regions. Instead of packing one A panel and walking all B
panels before moving to the next A panel, the new path packs the small A block
once, then reuses each packed-B panel across the resident A panels. The source
gate is limited to `m = 64, n >= 2048, k >= 1536` and
`m = 96/128, n >= 4096, k >= 1024`.

Final paired packed-B validation against clean `82fc74f` source:

- `64x2048x1536`: median `1.0815x`, bootstrap95 `[1.0740,1.0956]`,
  B-faster `98/101`, sign-p `1.36e-25`, holdout `1.0810x`.
- `64x2048x2048`: median `1.2078x`, bootstrap95 `[1.1826,1.2086]`,
  B-faster `98/101`, sign-p `1.36e-25`, holdout `1.2120x`.
- `64x3072x2048`: median `1.0998x`, bootstrap95 `[1.0665,1.1529]`,
  B-faster `89/101`, sign-p `1.08e-15`, holdout `1.0998x`.
- `64x3712x7168`: median `1.1298x`, bootstrap95 `[1.1201,1.1458]`,
  B-faster `98/101`, sign-p `1.36e-25`, holdout `1.1360x`.
- `96x4096x1024`: median `1.0725x`, bootstrap95 `[1.0709,1.0908]`,
  B-faster `100/101`, sign-p `8.05e-29`, holdout `1.0708x`.
- `128x4096x1024`: median `1.1806x`, bootstrap95 `[1.1808,1.1967]`,
  B-faster `101/101`, sign-p `7.89e-31`, holdout `1.1770x`.

Rejected low-edge evidence: exact `64x1024x1536` regressed/noised negative,
`64x1408x1536` was positive but below the acceptance floor on holdout,
and `m = 96/128, k = 512` was neutral/noisy. Keep those on the existing
packed-AMX full traversal.

### 2026-05-06: packed-B dispatch cleanup and gate floor

Current cleanup is intended to be behavior-preserving: `src/gemm.c` and
`bench/bench_gemm.c` extract the SME packed-B exclusion list into named helper
predicates so implementation dispatch and benchmark route labels stay aligned.

The cleanup also records a stricter acceptance floor for future dispatch gates:
prefer median speedup at least `3%`, holdout median at least `2%`, and
sign-p `< 1e-10`. Smaller exact gates remain possible, but should be treated as
exceptions that need unusually strong context and follow-up evidence.

Validation after the cleanup passed `make test`, `make all`,
`cmake --build build-cmake`, `ctest --test-dir build-cmake --output-on-failure`,
`sh -n tools/claim_audit.sh`, `git diff --check`, and a packed-B route smoke
covering both `packed_sme` and AMX fallback shapes.

### 2026-05-06: public packed-AB path added

The `tract-linalg` square smoke exposed a stronger-contract calibration target:
its `packed-both` mode prepacked both operands and reached about `2030 GF/s` on
sampled square sizes, which could beat COB's public packed-B path even though it
was not comparable to COB one-shot or packed-B.

COB now exposes the same class of repeated-use contract through
`cob_sgemm_pack_a`, `cob_sgemm_free_packed_a`, and
`cob_sgemm_rowmajor_packed_ab`. The AMX implementation reuses the existing
packed-A/packed-B `32x32` microkernel without repacking either operand during
the timed multiply. Correctness is covered by the existing 125-shape test suite,
which now checks one-shot, packed-B, and packed-AB outputs against the reference.

Repeat-31 square smoke after the change:

| shape | COB packed-AB median | COB packed-AB best |
| --- | ---: | ---: |
| `64x64x64` | `1715.24 GF/s` | `1753.05 GF/s` |
| `128x128x128` | `1593.38 GF/s` | `1628.73 GF/s` |
| `192x192x192` | `1948.32 GF/s` | `1990.05 GF/s` |
| `256x256x256` | `1830.76 GF/s` | `1861.71 GF/s` |
| `384x384x384` | `1939.98 GF/s` | `1965.23 GF/s` |
| `512x512x512` | `1961.17 GF/s` | `1976.52 GF/s` |
| `768x768x768` | `2031.32 GF/s` | `2073.16 GF/s` |
| `1024x1024x1024` | `2018.31 GF/s` | `2076.87 GF/s` |

This closes the sampled `tract packed-both` stronger-prepacking gap for the
large square calibration while keeping the one-shot and packed-B contracts
separate.

Follow-up `1024/1536/2048` focused runs confirmed the large-block traversal fix
keeps the fully packed path around `2.0 TF/s` best on larger aligned squares.
A second full-aligned traversal change switched non-large packed-AB shapes to
B-panel outer order, improving the `192x192x192` packed-AB best to
`2013.27 GF/s` in repeat-31 focused reruns. A same-session tract rerun measured
`tract packed-both` at `2006.11 GF/s` for `192` and `2026.33 GF/s` for `1024`,
while COB packed-AB measured `2035.53 GF/s` best at `1024`.

Low-repeat OpenBLAS route audit output in
`/private/tmp/cob-openblas-packed-ab-audit-20260506` showed no packed-AB gaps in
square, medium, or skinny suites. BLASFEO's packed-AB refresh in
`/private/tmp/cob-blasfeo-packed-ab-audit-20260506` also showed no gaps. BLIS's
low-repeat refresh in `/private/tmp/cob-blis-packed-ab-audit-20260506` had a
single noisy packed-AB row at `768x1152x1536`, but focused repeat-31 rerun
cleared it: COB packed-AB median `1920.78 GF/s` versus BLIS `1585.71 GF/s`.

### 2026-05-06: licensed baseline audit refreshed at f2826ec

Fresh route-aware baseline audits at commit `f2826ec` found no current
route-worthy gaps against OpenBLAS, BLIS, or BLASFEO in the square, medium, and
skinny families. The working tree before this docs task was clean except for
pre-existing untracked `.claude/`.

OpenBLAS was built with:

```sh
make bench-cblas CBLAS_NAME=OpenBLAS CBLAS_HEADER=cblas.h CBLAS_CFLAGS=-I/opt/homebrew/opt/openblas/include CBLAS_LDFLAGS='-L/opt/homebrew/opt/openblas/lib -lopenblas'
```

The default square smoke had COB one-shot ahead at `64`, `128`, `192`, `256`,
`384`, `512`, `768`, and `1024`. Route-aware audit output went to
`/private/tmp/cob-openblas-audit-20260506` with repeats=7. There were no
one-shot gaps in square/medium/skinny and no packed-B gaps in medium/skinny.
The only short-run packed-B square blip at `384` was `0.57%` and cleared on a
repeat-31 rerun: COB packed-B `1729.78 GF/s` versus OpenBLAS `1620.70 GF/s`.

BLIS was built with:

```sh
make bench-cblas CBLAS_NAME=BLIS CBLAS_HEADER=cblas.h CBLAS_CFLAGS='-I/private/tmp/blis_apple/frame/compat/cblas/src -I/private/tmp/blis_apple/frame/include -I/private/tmp/blis_apple/frame/thread -I/private/tmp/blis_apple/include/aaplmx -I/private/tmp/blis_apple' CBLAS_LDFLAGS=/private/tmp/blis_apple/lib/aaplmx/libblis.a
```

Route-aware audit output went to `/private/tmp/cob-blis-audit-20260506` with
repeats=7. The short audit showed possible one-shot losses at
`512x1280x4096`, `1536x1536x1536`, and `1024x1216x1536`, but repeat-31 focused
reruns cleared all three: COB one-shot `1554.35` versus BLIS `1411.70`,
`1776.41` versus `1651.72`, and `1783.31` versus `1678.46` GF/s,
respectively. There were no packed-B gaps.

BLASFEO was built with:

```sh
make bench-fortran-blas FORTRAN_BLAS_NAME=BLASFEO FORTRAN_BLAS_LDFLAGS=/private/tmp/blasfeo/lib/libblasfeo.a
```

The default square smoke showed BLASFEO far behind. Route-aware audit output
went to `/private/tmp/cob-blasfeo-audit-20260506` with repeats=3, with no
one-shot or packed-B gaps in square/medium/skinny against non-COB baselines.

Remaining-baseline smokes stayed within the current claim boundary. The Rust
`matrixmultiply` artifact
`/private/tmp/matrixmultiply_bench/target/release/matrixmultiply_bench` measured
only about `103-112 GF/s` across square `n = 64..1024`, far behind COB.
The `coral-aarch64` artifact `/private/tmp/coral_bench/target/release/coral_bench`
measured about `39.6`, `47.9`, `101.6`, `81.7`, `115.9`, `88.7`, `114.7`,
and `104.0 GF/s` for square `n = 64..1024`, also far behind COB.

KleidiAI driver `/private/tmp/kleidiai_fp32_driver` remained below COB on
sampled comparable one-shot shapes, including square `384` around `1490 GF/s`,
`768` around `1562 GF/s`, `1024` around `1532 GF/s`, and sampled `m = 64`
one-shot shapes around `480-765 GF/s`. KleidiAI elastic compute-only can be
higher, but excludes packing and is not the same public one-shot contract.

The `tract-linalg` artifact `/private/tmp/tract_bench/target/release/tract_bench`
had pack-each and packed-B samples below COB on sampled squares. Its
`packed-both` mode measured up to about `2030 GF/s` and can beat COB packed-B
on some square sizes; record that as an out-of-contract stronger-prepacking
calibration item for commit `f2826ec`. The later packed-AB entry above adds a
matching COB contract and closes the sampled square calibration gap.

### 2026-05-06: one-shot AMX high-K medium MC256 broadened

After the `m512,k2048` chunked-B change, a refreshed medium audit showed the
largest remaining external gaps in high-`K` AMX large-block one-shot rows. A
compile-flag paired screen with `COB_SGEMM_AMX_MC=256` showed the smaller row
block was now broadly useful in the audited high-`K` medium band, while neutral
rows were left unchanged by capping the source helper to the audited range.

The source helper now uses a 256-row A block for one-shot large-block AMX when:
`k >= 4096, m >= 768, 512 <= n <= 1280`; `k >= 3072, m = 512,
768 <= n <= 1280`; `k >= 3072, m >= 1024, 1152 <= n <= 1280`; plus exact
`768x1024x3072`. This keeps public packed-B blocking unchanged.

Representative paired screen wins (`repeat=61`, `iters=2`) included:
`512x768x4096` `1.0404x`, `512x1024x3072` `1.0213x`,
`512x1152x3072` `1.0189x`, `512x1152x4096` `1.0343x`,
`768x1216x4096` `1.0251x`, `768x1280x4096` `1.0275x`,
`1024x1152x3072` `1.0587x`, `1024x1152x4096` `1.0280x`,
`1024x1216x4096` `1.0166x`, and `1024x1280x4096` `1.0239x`.

Correctness coverage adds `512x1152x4096`, `768x1216x4096`, and
`1024x1280x4096`.

### 2026-05-06: m512 k2048 chunked one-shot B path accepted

The refreshed medium audit still showed one-shot gaps where COB packed-B was
fast after setup, but full one-shot B packing left too much packed-B scratch
outside the M5 P-cluster L2. A local candidate reused the existing chunked AMX
B-pack structure for `m = 512`, packing all A panels once and consuming B in
256-column chunks.

The accepted gate is intentionally exact-width: `m = 512, k = 2048` and
`n = 896`, `1024`, `1152`, or `1280`. The same candidate regressed nearby
widths, so no smooth threshold was inferred.

Paired A/B evidence (`repeat=101`, `iters=4`) for accepted widths:

- `512x896x2048`: median `1.0583x`, bootstrap95 `[1.0564,1.0673]`,
  B-faster `98/101`, holdout median `1.0561x`.
- `512x1024x2048`: median `1.0746x`, bootstrap95 `[1.0734,1.0809]`,
  B-faster `101/101`, holdout median `1.0752x`.
- `512x1152x2048`: median `1.0197x`, bootstrap95 `[1.0149,1.0215]`,
  B-faster `90/101`, holdout median `1.0199x`.
- `512x1280x2048`: median `1.0101x`, bootstrap95 `[1.0055,1.0131]`,
  B-faster `74/101`, holdout median `1.0091x`.

Rejected neighbor widths with the same path: `512x832x2048` `0.8667x`,
`512x960x2048` `0.9041x`, `512x1088x2048` `0.9097x`, and
`512x1216x2048` `0.9754x`. Other guards rejected in the first pass included
`512x512x4096`, `512x768x3072`, and `512x1216x3072`.

Correctness coverage adds `512x896x2048`.

### 2026-05-06: one-shot AMX high-K medium MC256 accepted

The medium audit showed remaining one-shot gaps on high-`K` AMX packed
large-block shapes, especially `768x1024x4096` and `1024x768x4096`. A global
compile-flag probe with `COB_SGEMM_AMX_MC=512` regressed the gap set, but
`COB_SGEMM_AMX_MC=256` improved several high-`K` medium rows, suggesting the
M5 8 MB P-cluster L2 favored a smaller one-shot A block for this band.

The accepted source change keeps public packed-B blocking unchanged and adds a
narrow one-shot helper that uses a 256-row A block only for `m >= 768` with
`n = 768, k >= 3072` or `n = 1024, k >= 4096`.

Higher-work paired A/B evidence (`repeat=101`, `iters=4`) against the previous
source:

- `1024x768x4096`: median `1.0297x`, bootstrap95 `[0.9913,1.0338]`,
  B-faster `82/101`, sign-p `1.66e-10`, holdout median `1.0250x`.
- `1024x1024x4096`: median `1.0410x`, bootstrap95 `[1.0287,1.0707]`,
  B-faster `87/101`, sign-p `4.8e-14`, holdout median `1.0359x`.
- `768x1024x4096`: median `1.0466x`, bootstrap95 `[1.0102,1.0584]`,
  B-faster `83/101`, sign-p `3.73e-11`, holdout median `1.0463x`.
- `768x768x4096`: median `1.0537x`, bootstrap95 `[1.0358,1.0626]`,
  B-faster `87/101`, sign-p `4.8e-14`, holdout median `1.0579x`.

Correctness coverage adds `768x768x3072`, `1024x768x4096`, and
`768x1024x4096`.

### 2026-05-06: m64 large-K counter pass and rejected scheduling probes

The local `mperf` sudo prompt succeeded through the macOS authentication
dialog, unblocking hardware-counter collection on the M5 Max. The default
counter set still needed splitting because `INST_SME_ENGINE_ALU` cannot be
mixed into the same pipeline event set on this chip.

Successful repeat-101 pipeline/SME/memory profiles for `64x2112x7168` and
`64x4096x7168` pointed away from a simple B-memory-wait explanation. Both
shapes showed high map/dispatch stalls, near-zero
`LDST_UNIT_WAITING_SME_ENGINE_MEM_DATA`, no SME cross-page events, and mostly
dispatch/scheduling pressure rather than waiting on SME memory data.

Rejected local probes from the same counter-driven pass:

- Issuing the m64 large-K prefetches only every fourth K iteration regressed.
- K4 unrolling of the prefetched kernels was neutral/noisy.
- K4 unrolling of the from-packed B64 tuple kernel was neutral/noisy.
- Disabling the exact `n = 4096` large-K reuse path hard-regressed
  `64x4096x7168` and `64x4096x8192`.
- Forcing B reuse on `64x2112x7168` hard-regressed the exact gap shape.

### 2026-05-06: exact 384x1280x1536 SME direct route accepted

The one-shot dispatcher now routes exact `384x1280x1536` through the SME
direct-`B` medium kernel. This extends the same low-height, `n = 1280`
exception family as `384x1280x1024`, but keeps higher-K and adjacent-width
shapes out of the route.

Initial multi-K exact SME-direct probe on M5 Max with repeat-61,
`COB_AB_ITERS=4`:

- Exact accepted candidate: `384x1280x1536` median `1.0920x`, bootstrap95
  `[1.0151,1.0997]`, B-faster `43/61`, sign-p `0.00187`, holdout median
  `1.1229x`.
- Rejected higher-K candidates: `384x1280x2048` median `0.9234x` and
  `384x1280x3072` median `0.6179x`.
- Width guards at `384x1216x2048` and `384x1344x2048` stayed neutral/noisy.

Isolated repeat-101 confirmation with `COB_AB_ITERS=6` strengthened the exact
`k = 1536` case: median `1.1875x`, bootstrap95 `[1.1574,1.1888]`, B-faster
`99/101`, sign-p `4.06e-27`, holdout median `1.1724x`. Guards stayed neutral:
existing `k = 1024` behavior median `0.9987x`, `k = 2048` median `1.0000x`
with holdout `0.9996x`, `384x1216x1536` median `0.9993x`,
`384x1344x1536` median `0.9972x`, and `512x1280x1536` median `0.9981x`.

Direct benchmark after the patch candidate: COB one-shot `384x1280x1536`
reached median `1793.29 GF/s` on route `sme_medium_direct`, versus packed-B
`1984.16 GF/s` and Accelerate `1887.44 GF/s`. Treat this as a COB one-shot
improvement, not an external median win claim for that noisy run.

Correctness coverage increases to 118 shapes after tests pass.

### 2026-05-06: exact 384x512x3072 SME direct route accepted

The one-shot dispatcher now routes exact `384x512x3072` through the SME
direct-`B` medium kernel. This is another narrow low-height exception; nearby
widths did not hold up well enough to broaden the route.

Focused paired one-shot evidence on M5 Max with repeat-61,
`COB_AB_ITERS=4`:

- Exact accepted gate: `384x512x3072` B/A median `1.1332x`, bootstrap95
  `[1.1114,1.1366]`, B-faster `58/61`, sign-p `3.29e-14`, holdout median
  `1.1332x`.
- Guards stayed neutral/noisy: `384x512x2048` median `0.9971x`,
  `384x512x4096` median `1.0006x`, `384x768x3072` was neutral under the exact
  candidate, and `512x512x3072` was neutral/noisy.

Confirmation with repeat-101, `COB_AB_ITERS=6`, was weaker/noisier but still
positive: `384x512x3072` median `1.1064x`, bootstrap95 `[1.0363,1.0817]`,
B-faster `86/101`, sign-p `2.83e-13`, holdout median `1.0980x`. A second
repeat-101 after removing the rejected temporary `n = 1024` gate strengthened
the result: median `1.1412x`, bootstrap95 `[1.0788,1.1223]`, B-faster
`89/101`, sign-p `1.08e-15`, holdout median `1.1384x`.

Rejected same-family probes: `384x768x3072` first looked positive with median
`1.0896x` and holdout `1.0928x`, but the rerun weakened to full median
`1.0656x` with holdout `0.9986x`, so reject it. `384x1024x3072` was a hard
regression at median `0.5743x`.

Correctness coverage increases to 117 shapes after tests pass.

### 2026-05-06: exact 384x1280x1024 SME direct route accepted

The one-shot dispatcher now routes exact `384x1280x1024` through the SME
direct-`B` medium kernel. This is a narrow low-height exception: adjacent
widths and the same width at `k = 1536` did not provide route-worthy evidence
in this first probe, though later isolated evidence accepted exact
`384x1280x1536`.

Exploratory paired one-shot evidence on M5 Max with repeat-61,
`COB_AB_ITERS=2`:

- Exact accepted gate: `384x1280x1024` B/A median `1.2179x`, bootstrap95
  `[1.2106x,1.2231x]`, B-faster `61/61`, sign-p `8.67e-19`, holdout median
  `1.2184x`.
- Guards stayed neutral/noisy or too small: `384x1216x1024` median `0.9979x`,
  `384x1344x1024` median `1.0000x`, `512x1280x1024` median `1.0006x`, and
  `384x1280x1536` median `1.0011x`.

Confirmation with repeat-101, `COB_AB_ITERS=4`:

- Exact accepted gate: `384x1280x1024` median `1.2218x`, bootstrap95
  `[1.1705x,1.2057x]`, B-faster `95/101`, sign-p `1.07e-21`, holdout median
  `1.2276x`.
- Guards stayed neutral: `384x1216x1024` median `0.9995x`,
  `384x1344x1024` median `1.0012x`, and `384x1280x1536` median `1.0000x`.

Direct benchmark after the patch: COB one-shot `384x1280x1024` reached median
`2005.24 GF/s` on route `sme_medium_direct`, versus packed-B `1962.25 GF/s`
and Accelerate `1958.43 GF/s`.

Correctness coverage increases to 116 shapes.

### 2026-05-06: exact 512x1024x1536 SME direct route accepted

The one-shot dispatcher now routes exact `512x1024x1536` through the SME
direct-`B` medium kernel. This supersedes the older broad `n = 1024` SME
rejection with a narrow exact gate and keeps the general `n = 1024` exclusion
in place for other shapes.

Focused paired one-shot evidence on M5 Max:

- Exact accepted gate: `512x1024x1536` median `1.0699x`, bootstrap95
  `[1.0408,1.0650]`, B-faster `84/101`, holdout median `1.0805x`.
- Earlier repeat-201 probe was also positive: median `1.0372x`, bootstrap95
  `[1.0351,1.0474]`, B-faster `182/201`, holdout median `1.0358x`.
- Guards stayed neutral/noisy: `512x1024x1024`, `512x1024x2048`,
  `768x1024x1536`, `512x960x1536`, and `512x1088x1536`.

Correctness coverage already included `512x1024x1536`; no test count change.

Follow-up rejected: routing high-`K` medium audit gaps through SME direct-B was
a hard regression. The exact probe covered `768x512x4096`, `768x768x4096`,
`1024x768x4096`, and `768x1024x4096`; paired medians were `0.9484x`,
`0.7460x`, `0.7282x`, and `0.3834x`, with all holdouts negative. Guard rows
at `768x512x3072`, `768x768x3072`, and `1024x768x3072` stayed neutral/noisy.
Keep these high-`K` medium rows on packed AMX.

### 2026-05-06: exact 512x512x3072 SME direct route accepted

The one-shot dispatcher now routes exact `512x512x3072` through the SME
direct-`B` medium kernel instead of the packed-AMX conflict path. This is a
narrow exception to the general `n = 512, k >= 2048` packed-AMX rule.

Focused paired one-shot evidence on M5 Max:

- Exact accepted gate: `512x512x3072` median `1.0713x`, bootstrap95
  `[1.0358,1.0660]`, B-faster `87/101`, holdout median `1.0730x`.
- Earlier repeat-201 probe was also positive: median `1.0171x`, bootstrap95
  `[1.0041,1.0208]`, B-faster `137/201`, holdout median `1.0163x`.
- Guards stayed neutral/noisy: `512x512x2048`, `512x512x4096`,
  `512x768x3072`, `768x512x3072`, and `1024x512x3072`.

Rejected broader SME-direct probe: exact `512x768x3072`, `512x1152x2048`,
`512x1216x2048`, `512x1216x3072`, and `512x1280x2048` did not hold up.
`512x1216x3072` and `512x1280x2048` were hard regressions.

Correctness coverage adds `512x512x3072`.

### 2026-05-06: m64 k512 mid-width SME reuse accepted

The one-shot `m = 64, k = 512` SME B-reuse route now starts at mid-width
multiples of 512 from `n = 2048` through `3584`, while retaining the existing
`n >= 4096` long-`N` route. A broad threshold lowering to every `n >= 2048`
was rejected because off-boundary widths regressed.

Focused paired one-shot evidence on M5 Max with the shaped source predicate:

- `64x2048x512` median `1.0485x`, bootstrap95 `[1.0438,1.0552]`,
  B-faster `100/101`, holdout median `1.0489x`.
- `64x2560x512` median `1.0557x`, bootstrap95 `[1.0513,1.0608]`,
  B-faster `99/101`, holdout median `1.0601x`.
- `64x3072x512` median `1.0874x`, bootstrap95 `[1.0791,1.0926]`,
  B-faster `99/101`, holdout median `1.0873x`.
- `64x3584x512` median `1.0804x`, bootstrap95 `[1.0795,1.0953]`,
  B-faster `99/101`, holdout median `1.0839x`.
- Existing long-`N` guards stayed neutral: `64x4096x512` median `1.0015x`
  and `64x8192x512` median `1.0000x`.

Rejected broad-threshold evidence: with `COB_SGEMM_M64_SME_LONG_N_K512_MIN_N`
lowered to `2048`, `64x2112x512` regressed to `0.8994x` and
`64x2304x512` regressed to `0.9402x`.

Follow-up rejected: adding exact `64x1024x512` to the SME B-reuse route was
too weak to ship. The first repeat-201 run showed median `1.0476x`, but the
`COB_AB_ITERS=4` confirmation weakened to median `1.0184x`, bootstrap95
`[1.0059,1.0257]`, B-faster `61/101`, sign-p `0.046`, and holdout median
`1.0062x` with CI crossing `1.0`. Behavior-identical guards at `1536`, `2048`,
`2112`, and `4096` were neutral to slightly negative/noisy, so exact `n = 1024`
stays on AMX.

Correctness coverage adds `64x2048x512`, `64x2560x512`, `64x3072x512`, and
`64x3584x512`.

### 2026-05-06: exact 64x1024x1536 SME skinny route accepted

The one-shot `m = 64, k = 1536` SME skinny direct route now includes exact
`n = 1024`, while leaving the existing `n >= 1408` gate unchanged. A broad
lowering to `n >= 1024` was rejected because `n = 1088`, `1152`, and `1280`
regressed.

Focused paired one-shot evidence on M5 Max:

- Broad rejected gate: `64x1024x1536` improved `1.0182x`, but
  `64x1088x1536` regressed `0.9397x`, `64x1152x1536` regressed `0.9576x`,
  and `64x1280x1536` regressed `0.9485x`.
- Exact accepted gate: `64x1024x1536` median `1.0303x`, bootstrap95
  `[1.0160,1.0422]`, B-faster `143/201`, holdout median `1.0233x`.
- Confirmation with `COB_AB_ITERS=4`: `64x1024x1536` median `1.0536x`,
  bootstrap95 `[1.0471,1.0632]`, B-faster `96/101`, holdout median `1.0536x`.
- Guards stayed neutral: `64x1088x1536` median `1.0000x` and
  `64x1024x2048` median `1.0083x` in the focused confirmation.

Correctness coverage adds `64x1024x1536`.

Follow-up rejected: forcing exact `64x1024x1536` away from the SME skinny
direct route and through the one-shot packed-AMX path was a hard regression.
The paired run measured median `0.5833x`, bootstrap95 `[0.5383,0.5597]`,
B-faster `0/201`, with holdout median `0.5278x`. Keep this shape on the SME
skinny route despite the remaining noisy Accelerate audit gap.

### 2026-05-06: public packed-B 512x1280x2048 MC512 accepted

The public packed-`B` AMX path now uses a 512-row block for exact
`512x1280x2048`. The old 384-row block split the shape into 384+128 rows even
though the right-hand panel was reusable across the full 512 rows.

The first repeat-201 packed-B run had a positive median but noisy confidence,
so the decision was repeated with `COB_AB_ITERS=4`. The focused rerun was clean:
`512x1280x2048` median `1.0137x`, bootstrap95 `[1.0117,1.0151]`,
B-faster `95/101`, holdout median `1.0133x`. Rejected guards:
`512x1280x1536` regressed to `0.9947x`, `512x1344x2048` regressed to
`0.9984x`, and `512x1280x3072` / `512x1216x2048` stayed neutral/noisy.

Correctness coverage adds `512x1280x2048`.

### 2026-05-06: counter probe workflow added

Added an opt-in benchmark row filter, `COB_BENCH_ONLY`, so hardware-counter
runs can isolate `cob one-shot`, `cob packed-B`, `pack-setup`, or `Accelerate`
instead of measuring the whole comparison benchmark. Added
`tools/counter_probe.sh`, which wraps `mperf-stat` with the M5 `as5` KPEP
database and default events for cycles, instructions, cache/TLB misses, map and
dispatch stalls, and SME wait/ALU counters.

Follow-up tooling: `COB_BENCH_HOT_SECONDS=N` keeps a single benchmark row
running in a hot loop for profiler attachment. This is intended for `sample`,
Instruments, or counter tools and is disabled by default. A Codex-sandboxed
`sample -wait` attempt failed on process-list `sysctl` permissions, so `sample`
should be run from a normal terminal.

Local setup status: `/private/tmp/mperf/mperf-stat` can list M5 `as5` events
when launched with `MPERF_KPEP_DB=as5`, but actual sampling still reports
`Root privileges required` without sudo. The next counter pass should run, for
example:

```sh
sudo env MPERF_KPEP_DB=as5 COB_COUNTER_ONLY=one-shot sh tools/counter_probe.sh 64x4096x7168
```

### 2026-05-06: exact 512x1280x1536 SME direct route accepted

The one-shot dispatcher now routes exact `512x1280x1536` through the SME
direct-`B` medium kernel instead of the packed AMX path. This avoids the
one-shot B-pack setup cost on a shape where packed-B compute was already fast
but total one-shot time trailed Accelerate in the medium audit.

Focused paired one-shot evidence on M5 Max:

- `512x1280x1536` median `1.1286x`, bootstrap95 `[1.0919,1.1212]`,
  B-faster `184/201`, holdout median `1.1207x`.
- Guards stayed neutral/noisy: `512x1280x1024`, `512x1280x2048`,
  `768x1280x1536`, `512x1216x1536`, and `512x1344x1536`.

Correctness coverage adds `512x1280x1536`.

### 2026-05-06: public packed-B 512x1024x1536 AMX fallback accepted

The public packed-`B` SME route now rejects exact `512x1024x1536`, letting the
existing AMX packed-`B` path handle that shape. This addresses a fresh audit
row where the SME packed-`B` median was unstable, while leaving the one-shot
route unchanged because one-shot already used AMX there.

Focused paired packed-`B` evidence on M5 Max:

- `512x1024x1536` median `1.0270x`, bootstrap95 `[1.0236,1.0602]`,
  B-faster `123/201`, holdout median `1.0297x`.
- `768x1024x1536` stayed neutral/noisy at median `0.9944x`.
- `512x1152x1536` was a small behavior-identical/layout regression in the
  probe, so the accepted route is exact to `512x1024x1536`.

Correctness coverage adds `512x1024x1536`.

### 2026-05-06: exact 768x512x4096 large-block AMX accepted

The AMX packed full-width large-block rule now includes exact
`768x512x4096` for both one-shot and public packed-`B`. The broader probe that
started `n = 512` large-blocking at `m >= 768, k >= 4096` was rejected because
`768x512x8192` regressed, so the accepted gate is exact.

Focused paired evidence on M5 Max:

- One-shot `768x512x4096` median `1.0383x`, bootstrap95 `[1.0280,1.0387]`,
  B-faster `178/201`, holdout median `1.0419x`.
- Packed-`B` `768x512x4096` median `1.0431x`, bootstrap95
  `[1.0439,1.0584]`, B-faster `185/201`, holdout median `1.0431x`.
- Rejected guard: `768x512x8192` regressed in both one-shot and packed-`B`
  (`0.9747x` and `0.9788x` medians).

The existing correctness suite already covers `768x512x4096`.

### 2026-05-06: one-shot n1152 k2048 packed path lowered to m512

The one-shot AMX medium dispatch now uses the packed-`B` path for exact
`n = 1152, k = 2048` starting at `m >= 512`. The previous gate required
`m >= 1024`, leaving `512x1152x2048` and `768x1152x2048` on strided-`B` AMX
despite the packed path being much more stable in the route-aware audit.

Focused paired old-vs-new evidence on M5 Max:

- `512x1152x2048` median `1.1715x`, bootstrap95 `[1.1387,1.2021]`,
  B-faster `167/201`, holdout median `1.1598x`.
- `768x1152x2048` median `1.1318x`, bootstrap95 `[1.1339,1.1973]`,
  B-faster `162/201`, holdout median `1.0818x`.
- `1024x1152x2048` was behavior-identical/noisy at median `0.9986x`.

The rule stays exact to `k = 2048`; existing `k >= 3072` behavior is unchanged.
Correctness coverage adds `768x1152x2048`.

### 2026-05-06: packed-B m384 high-N AMX block accepted

The public packed-B AMX path now uses the 384-row large-block schedule for
`m = 384`, `n >= 2048`, and `k >= 1024`. The previous `n >= 2048` packed-B
threshold raised the row block to 512, which made `m = 384` miss the large-block
path entirely and lose B-panel reuse.

Paired old-vs-new packed-B evidence on M5 Max:

- `384x2048x1024` median `1.0537x`, bootstrap95 `[1.0439,1.0923]`,
  B-faster `98/121`.
- `384x3072x1024` median `1.1301x`, bootstrap95 `[1.1138,1.1618]`,
  B-faster `120/121`.
- `384x4096x1024` median `1.1683x`, bootstrap95 `[1.1382,1.1743]`,
  B-faster `119/121`.
- `384x2048x1536` median `1.1845x`, bootstrap95 `[1.1662,1.1963]`,
  B-faster `120/121`.

The guard keeps `k = 512` on the old path and leaves `m >= 512` unchanged.
Validation passed: `make test`, `make bench`, and `git diff --check`.

### 2026-05-05: B-pack prefetch accepted

Added guarded `COB_SGEMM_PACK_B_PREFETCH_DISTANCE` prefetching for full
32-column `B`-panel packing and accepted default distance `16`. Distance `8`
was positive at `1280` and broadly helpful but noisy, while distance `4` was
weaker. Distance `16` gave the best large-square signal.

Key paired evidence, old default vs new default:

- `1536` one-shot median `1.0142x`, bootstrap95 `[1.0109,1.0266]`,
  B-faster `61/81`.
- `2048` one-shot median `1.0155x`, bootstrap95 `[1.0019,1.0203]`,
  B-faster `72/81`.
- `1280` was positive but noisy, with holdout positive.
- `512` and `1024` were neutral/noisy.

Validation passed: `make test` across 50 shapes, `make all`, and
`git diff --check`. A route smoke was noisy at `1024` but showed large-square
route behavior.

### 2026-05-05: route-aware benchmark reporting

Added optional route-aware benchmark reporting with `COB_BENCH_ROUTE=1`, so
CSV/text benchmark rows can include COB dispatch route labels. This helps
boundary/grid analysis without changing GEMM hot paths.

Route labels were smoke-tested on square and skinny shapes. The test count was
corrected to 49 after adding the `64x1024x7168` correctness shape.

Validation passed: `make test`, `make all`, and `git diff --check`.

### 2026-05-04: AMX packed-B path

The first major speed path used Apple Silicon AMX with a public packed-`B` API,
native AMX `B` layout, faster aligned packing/stores, and reuse of packed `B`
panels across row blocks.

Result: this became the strongest exact-scope path in COB. On large square
cases, the packed-`B` path measured around 1.9-2.0 TF/s in later validation.

### 2026-05-04: one-shot direct-B AMX optimizations

The one-shot path was optimized for smaller square cases by avoiding some
scratch allocations and extending the direct-`B` AMX range. The latest pushed
commit in this series was `f335c66 Avoid direct-B AMX conflict at 512 stride`.

Result: validation after that commit passed with `make test`, `make bench`, a
CMake build, and `ctest`. COB improved substantially, but Apple Accelerate
could still win in some tested small cases, including around `n = 192`.

### 2026-05-04 to 2026-05-05: external library comparisons

The default square-size benchmark comparisons showed COB beating `blis_apple`
and OpenBLAS. BLASFEO, Rust `matrixmultiply`, Eigen, and `coral-aarch64` were
far below COB in the tested exact scope.

LIBXSMM's exact `beta = 0` path was also far below COB. Its `beta = 1` plus
zeroed-output workaround was close, but that is not the same exact API.

`tract-linalg` AMX one-shot and packed-`B` paths were below COB. Its
compute-only packed-both path was strong, but that comparison does not use the
same API shape.

### 2026-05-05: MpGEMM comparison

MpGEMM was found at <https://github.com/MpGEMM/mpgemm>, with paper
<https://arxiv.org/abs/2512.21473>. It is source-available, but no license file
was observed.

Result: in same-process benchmarks, MpGEMM beat COB in several one-shot FP32
single-thread square cases, including `n = 256`, `384`, `512`, `768`, and
`1024`.

### 2026-05-05: clean-room SME prototype

A clean-room SME prototype was correct and looked promising in compute-only
runs, reaching around 2.0 TF/s in some measurements. The one-shot version was
slower overall because packing dominated the runtime.

Result: the current follow-up is to test a packed-`B` SME path, where packing
can be amortized instead of paid on every call.

### 2026-05-05: guarded SME packed-B path

Commit `0b3f398 Add SME packed-B GEMM path` added a guarded clean-room Apple
SME2.1 `16x64` path for the public packed-`B` API. It reuses the existing
32-column packed-`B` layout and AMX `A` packing.

The path is runtime gated on SME2.1 and 16 FP32 lanes. The dispatch threshold
was tightened to `m/n/k >= 512` after `n = 384` proved noisy.

Result: `make test` passed with 24 shapes, including a new 512
packed-vs-direct check. `make bench` showed packed-`B` around 1837 GF/s at
`n = 512`, 1982 GF/s at `n = 768`, and 1992 GF/s at `n = 1024` in one run. A
CMake build and `ctest --test-dir build-cmake --output-on-failure` also passed.
This improves COB packed-`B`, but still does not close the one-shot gap to
MpGEMM.

### 2026-05-05: narrow SME route for one-shot 512

Commit `bc00b5c Route 512 one-shot GEMM through SME` added a narrow one-shot
optimization for the 512-wide direct-`B` conflict case. After `B` has already
been packed, that case reuses the guarded SME packed-`B` kernel.

The route is limited to `n == 512`, aligned AMX rows, and `k >= 512`. Broader
SME direct-`B` experiments were backed out because they were noisy and
shape-sensitive.

Result: `make test` passed with 24 shapes. In repeated same-session `n = 512`
benchmarks, one-shot improved from roughly 1.58-1.61 TF/s with SME disabled to
about 1.65-1.70 TF/s with the route enabled. A CMake build and
`ctest --test-dir build-cmake --output-on-failure` also passed. This improves
one-shot 512, but still does not complete the universal fastest claim.

### 2026-05-05: post-0182f31 ACL and SME direct-B attempts

ACL CMake SVE/SME builds crashed on Apple M5 Max with `SIGILL`, traced to a
non-streaming SVE instruction in `CpuGemmAssemblyDispatch::configure`. An ACL
NEON-only build ran, but measured only about 120-126 GF/s single-thread, far
below COB.

An SME packed-`B` loop-order experiment was tested and not kept because results
were mixed and noisy.

A conservative local SME direct-`B` one-shot route was added for medium
contiguous square sizes `832-1216`, excluding `1024`. Paired runs improved the
one-shot benchmark results. `1280` was excluded as unstable.

Final pushed route commit `faf4401` uses blocked `A` packing for the medium
direct-`B` SME path. Validation passed with `make test`, a CMake build, and
`ctest --test-dir build-cmake --output-on-failure`. A lower-size gate experiment
extending the route to `768` was rejected after integrated benchmarks regressed
at `768`, so the final gate remains `832-1216`, excluding `1024`.

### 2026-05-05: rectangular SME direct-B route

Commit `3f4ea03` broadened the direct-`B` SME route from square-only to selected
medium contiguous rectangular cases. It also added an Apple-arm64 rectangular
direct-vs-packed test.

Result: `make test` passed with 26 shapes, plus a CMake build and
`ctest --test-dir build-cmake --output-on-failure`. A temporary rectangular
benchmark showed direct around 1.94-2.01 TF/s on the tested rectangular cases.

A temporary two-step SME inner-loop unroll in
`/private/tmp/cob_sme_unroll2_exp` passed tests, but was rejected because it
slowed important routed sizes and hurt packed-`B` stability.

### 2026-05-05: post-841ee51 rejected experiments

Widening the post-pack SME route beyond `n == 512` was tested in
`/private/tmp/cob_postpack_sme_exp` and rejected because it regressed
`n = 768` and `n = 1536`, and did not beat the existing direct-`B` route for
gated sizes.

Changing `COB_SGEMM_AMX_MC` from `384` to `512` was tested in
`/private/tmp/cob_mc512_exp` and rejected because it hurt packed-`B` at larger
sizes such as `1280` and `1408` without a sufficient one-shot gain.

A narrow `n = 384` SME direct-`B` gate was tested in
`/private/tmp/cob_sme384_exp`. It passed tests, but did not beat the current
integrated AMX one-shot path at `n = 384`, so it was rejected.

A strided-`B` AMX row-blocking experiment was tested in
`/private/tmp/cob_strided_b_block_exp`. It passed tests, but regressed small
one-shot sizes such as `64`, `128`, and `256`, so the existing per-32-row `A`
packing stayed.

### 2026-05-05: median benchmark output and admin follow-up

Commit `7ea85c5` added median-aware throughput output to the benchmark harness,
so benchmark rows now show median GF/s alongside the existing timing summary.

Removing the `n == 1024` exclusion from the SME direct-`B` medium gate was
tested and rejected. The change passed tests, but did not improve `1024` median
throughput and caused instability around the gated range, so it was reverted.

Commit `a27a03d` documented an admin/tooling fix after an edit-confirmation UI
appeared while patch-editing a `/private/tmp` worktree. `AGENTS.md` and
`.codex-reminder.md` now tell agents to treat that as client/tooling friction,
avoid that edit path, and continue in the repo checkout without asking.

### 2026-05-05: packed-B SME dispatch capped

Commit `c699b56` capped the public packed-`B` SME dispatch to `n <= 1216`.
Benchmarks showed the AMX packed-`B` path was faster and more stable for large
packed-`B` cases, especially `n = 1536`.

Result: `make test` passed. A selected benchmark after the change showed
packed-`B` median around 1.94-1.96 TF/s at `n = 1536`, versus the earlier broad
SME route around roughly 1.45-1.53 TF/s in the same session.

### 2026-05-05: one-shot SME direct-B 1280 experiment

Commit `d68430c` extended the medium direct-`B` SME route from `1216` to `1280`
and added an Apple-arm64 direct-vs-packed correctness case for `1280`.

Result: with the temporary `1280` gate, repeated `n = 1280` one-shot medians
were mostly about 1.84-1.89 TF/s. With the original `1216` cap, repeated
`n = 1280` one-shot medians were mostly about 1.65-1.74 TF/s with one severe
outlier. Validation passed with `make test` across 27 shapes, and
`git diff --check` passed.

### 2026-05-05: benchmark repeat controls and n=1024 SME rejection

A one-shot `n = 1024` experiment routed the case through the post-pack SME path
after `B` packing by extending the existing `n == 512` conflict check to also
cover `n == 1024`. It passed `make test` across 27 shapes, but did not
materially improve one-shot `n = 1024` versus the AMX path, so it was reverted.

Commit `598444e` made benchmark repeats configurable with
`COB_BENCH_REPEATS`, changed large-shape default repeats from 4 to 5 for a real
median, and documented the environment variable in `README.md`. Validation
included `make test`, `make all`, `git diff --check`, and
`env COB_BENCH_REPEATS=9 ./build/cob_gemm_bench 1024 1280`.

### 2026-05-05: packed-B medium-width dispatch narrowed

Commit `8143b9f` excluded `n = 832` and `n = 960` from the public packed-`B`
SME route, so those widths use AMX packed-`B` instead.

Result: a focused `COB_BENCH_REPEATS=9` benchmark after the change showed
packed-`B` medians about 2003-2014 GF/s at `n = 832` and about 2018 GF/s at
`n = 960`, better than the prior SME-dispatch baseline observed around
1969 GF/s at `832` and 1977-1988 GF/s at `960`. Validation passed with
`make test` across 28 shapes, and `git diff --check` passed.

### 2026-05-05: SME direct-B row block split

Two strided-`B` SME loop experiments were tested and rejected. A B-block-first
loop order passed tests, but regressed key one-shot sizes, especially `n = 896`
and `n = 1216`, so it was reverted. A pointer-increment rewrite inside the same
loop also passed tests, but regressed `n = 896`, so it was reverted.

Commit `73cb130` added a separate 256-row SME direct-`B` row block while
keeping public packed-`B` SME on the 384-row AMX block.

Result: a focused `COB_BENCH_REPEATS=9` benchmark improved several one-shot
routed sizes, including medians around 2013 GF/s at `n = 960`, 1962 GF/s at
`n = 1088`, 1922 GF/s at `n = 1152`, and 1926 GF/s at `n = 1216`. Validation
passed with `make test` across 28 shapes, and `git diff --check` passed.

### 2026-05-05: skinny SME KC calibration

Refreshed current COB versus MpGEMM O0-driver calibration showed COB still
trailing MpGEMM on six `m = 64` skinny shapes. The largest prior usable gap was
around `64x8192x1024`.

A global `COB_SGEMM_SKINNY_SME_KC=1024` probe helped `64x8192x1024` and
`m = 96/128` shapes with `k >= 1024`, but regressed `64x4096x7168`, so it was
rejected. The accepted change was selective: large `KC = 1024` only for
`m = 64` wide `k = 1024`, exact `m = 64, n = 1024` large-`K` skinny cases, and
`m = 96/128` only when `k == 1024` or `n >= 8192`.

Key paired evidence: `64x7168x1024` was about `1.04x`, `64x8192x1024` was about
`1.04-1.05x`, `96x4096x1024` was `1.0206x` at 101 repeats, and
`128x4096x1024` was `1.0304x` at 101 repeats. `64x4096x7168` stayed neutral
after the selective rule. A broader `n < 4096` rule was rejected after focused
101-repeat regression work on `64x2112x7168`, keeping that shape neutral.

Validation passed with `make test`, `make all`, and `git diff --check`. A
direct-vs-packed test was also added for `64x1024x7168`.

### 2026-05-05: direct SME row-block probes and packed-B cap

Direct SME row-block compile-time probes with `COB_SGEMM_SME_DIRECT_MC=384`
and `512` were rejected. Paired A/B runs over `n = 960`, `1088`, `1152`,
`1216`, and `1280` showed neutral/noisy results or small regressions,
especially at `1280`.

The packed-`B` SME max width was lowered from `1216` to `1152` after paired
packed-`B` A/B showed `1216` improved by about 1.014x at 101 repeats,
bootstrap95 `[1.0105x, 1.0180x]`, with B faster in 71/101 samples and holdout
median 1.0170x.

Result: validation passed with `make test` across 48 shapes, plus `make all`
and `git diff --check`. A relinked focused benchmark in a noisy session showed
packed-`B` at `n = 1216` median 1964 GF/s versus Accelerate 1900.68 GF/s.

### 2026-05-05: paired A/B harness refinement

The paired A/B benchmark harness was refined to add split-half holdout
reporting while keeping the existing speedup-CV, sign-test, and checksum-fail
diagnostics. The bootstrap sampler was also changed to avoid low-bit modulo
artifacts that can degenerate confidence intervals at power-of-two repeat
counts.

This addresses part of the benchmark critique by making overfit-looking wins
easier to spot. Heavier follow-ups remain open, including grid boundary
modeling, hardware counter collection, dispatcher refactoring, and
visualization.

### 2026-05-05: packed-B n=1088 dispatch narrowed

Commit `04f2a91` excluded `n = 1088` from the public packed-`B` SME route, so
that width uses AMX packed-`B` instead. It also added an Apple-arm64
direct-vs-packed correctness case for `1088`.

Result: a focused `COB_BENCH_REPEATS=9` benchmark after the change showed
`n = 1088` packed-`B` medians about 2027-2044 GF/s, better than the prior SME
route around roughly 1990-2010 GF/s in nearby runs. Validation passed with
`make test` across 29 shapes, and `git diff --check` passed.

### 2026-05-05: one-shot n=832 AMX direct-B extension

Commit `50bcead` extended the AMX direct-`B` max `N` from `768` to `832`, so
one-shot `n = 832` uses AMX direct-`B` instead of the SME medium route.

Result: a focused `COB_BENCH_REPEATS=15` benchmark showed `n = 832` one-shot
medians around 2014-2021 GF/s, versus the old cutoff control around roughly
1983-2003 GF/s. Validation passed with `make test` across 29 shapes, and
`git diff --check` passed.

### 2026-05-05: one-shot n=960 AMX direct-B extension

Commit `c99f5d5` extended the AMX direct-`B` max `N` from `832` to `960`, so
one-shot `n = 960` uses AMX direct-`B`.

Result: a focused `COB_BENCH_REPEATS=15` benchmark showed `n = 960` one-shot
medians about 2029-2036 GF/s, improving over the previous 832-cutoff control
range around roughly 1995-2006 GF/s. Accelerate still remained around
2058-2067 GF/s. Later split-gate work kept `n = 896` on the SME medium route
while preserving the `n = 960` AMX win. Validation passed with `make test`
across 29 shapes, and `git diff --check` passed.

### 2026-05-05: one-shot n=896/960 AMX direct-B split gate

Commit `7fec4b5` split the AMX direct-`B` medium gate so AMX handles
`n <= 832` plus an explicit extra `n == 960`, letting `n = 896` fall back to
the SME medium route.

Result: focused `COB_BENCH_REPEATS=15` validation showed one-shot medians about
1976 GF/s at `n = 896` and about 2029 GF/s at `n = 960`. Validation passed
with `make test` across 29 shapes, and `git diff --check` passed.

### 2026-05-05: latest post-1088 rejected experiments

After commits `04f2a91` and `822123c`, several follow-up experiments were
tested and rejected. Extending the SME direct-`B` one-shot gate to `n = 1408`
after the 256-row direct-`B` block change passed tests, but did not improve
one-shot `1408`, so it was reverted. Lowering the AMX large-block threshold
from `n >= 1152` to `n >= 1024` in both one-shot and public packed-`B` AMX
paths passed tests, but did not help one-shot `1024` and created noisy
regressions around `1152` and `1216`, so it was reverted.

Forcing public packed-`B` AMX at `n = 1152` and `n = 1216` by excluding those
widths from SME packed-`B` also passed tests, but was mixed and not clearly
better, so it was reverted. A stack-backed `A` scratch experiment for small AMX
direct-`B` one-shot, including an overrideable macro cutoff and cutoff `224`,
passed tests, but a controlled heap-only comparison was faster for `96-224`, so
it was reverted.

Excluding `n = 1088` from the SME direct-`B` one-shot route was also tested and
rejected. It made one-shot `1088` fall back to the slower packed/large path and
benchmarked clearly worse, with medians around 1841-1857 GF/s versus the
committed SME route around 1966-1970 GF/s, so the experiment was reverted.

Workflow note: patch-editing detached `/private/tmp` worktrees can still trigger
client edit-confirmation friction despite standing repo permission. Future
experiments should prefer the main checkout, or use non-editing toggles when a
detached worktree is only needed for measurement.

Validation after the reverts included `make test`; the tree returned clean.

### 2026-05-05: rectangular benchmark arguments

Commit `4074513` added rectangular `MxNxK` benchmark arguments, such as
`832x960x896`, while preserving the previous square `N` arguments.

Result: validation passed with `make test` and a `COB_BENCH_REPEATS=1`
benchmark smoke run.

### 2026-05-05: one-shot n=1088 AMX direct-B gate

Commit `2306377` routes one-shot `n = 1088` through AMX direct-`B` by adding a
second extra direct-`B` gate. It leaves `n = 1024` on the previous packed path,
because testing AMX direct-`B` at `n = 1024` did not improve it.

Result: the same-session old SME one-shot `1088` baseline was about
1954-1963 GF/s median. The new AMX direct-`B` gate measured about
2025-2033 GF/s median in focused 15-repeat runs, with a later 7-repeat sanity
run around 2009-2011 GF/s median.

### 2026-05-05: one-shot n=1152/1216 AMX direct-B gate

Commit `f45bb5e` routes one-shot `n = 1152` and `n = 1216` through AMX
direct-`B` as large extra direct-`B` sizes. It leaves `n = 1280` on the
previous path because testing there was mixed.

Result: focused 15-repeat square validation showed the old baseline around
1911-1916 GF/s median at `n = 1152` and 1925-1927 GF/s at `n = 1216`. The new
direct-`B` route measured about 1985-1991 GF/s at `n = 1152` and
2027-2032 GF/s at `n = 1216` in final validation.

Validation passed with `make test` across 31 shapes after adding `1152` and
`1216` direct-vs-packed coverage. Rectangular smoke with the `MxNxK` benchmark
showed acceptable behavior for `832x1152x896`, `896x1152x1152`,
`1280x1152x960`, `1152x1216x896`, and `1216x1216x1024`.

### 2026-05-05: packed-B setup benchmark row

Commit `0a7de51` added an opt-in `COB_BENCH_PACK_SETUP=1` benchmark row for
the public packed-`B` setup cost and documented the environment variable in
`README.md`.

Result: validation passed with `make test`, a benchmark build, and a smoke run
using `COB_BENCH_REPEATS=3 COB_BENCH_PACK_SETUP=1 ./build/cob_gemm_bench 64
832x960x896`.

### 2026-05-05: row-wise public B-pack order rejected

A row-wise public `B`-pack order experiment changed source traversal during
setup. It passed tests, but setup throughput collapsed compared with the
existing panel-wise packing order.

Result: the baseline panel-wise setup measured roughly 90-114 GB/s across
`512..1536` in a 15-repeat pack-setup run. The row-wise traversal measured only
about 16-67 GB/s depending on size, so it was reverted. The existing panel-wise
packing order stays.

### 2026-05-05: small-k scalar A-pack rejected

Forcing scalar `A`-panel packing for `k <= 256` passed `make test`, but it was
a hard regression in the focused small-size sweep.

Result: a 15-repeat sweep dropped COB one-shot and packed-`B` medians for
`96..256` to roughly 557, 587, 813, 811, 975, and 934 GF/s. That was far below
the prior AMX-pack small-size range of about 1300-1750 GF/s, so the experiment
was reverted.

### 2026-05-05: probe-only route-gap follow-up

After the packed-`B` row-block change, route-aware gap reporting still showed
`1088` and some large-square shapes as possible gaps.

A focused `1088` route flip, compiled with
`COB_SGEMM_AMX_STRIDED_B_EXTRA_N2=0`, was rejected at 101 repeats. The median
was about `0.9914x`, bootstrap95 `[0.9669, 1.0078]`, and holdout was also not
positive.

Disabling the AMX large-block path for `1280`, `1408`, `1536`, `1792`, and
`2048` by compiling with `COB_SGEMM_AMX_MC=4096` was also rejected because all
tested shapes regressed, including `1408` at about `0.945x` and `2048` at about
`0.911x`.

Result: these were probe-only measurements. No source behavior changed.

### 2026-05-05: public packed-B SME n=1280 rejected

Re-enabling public packed-`B` SME dispatch through `n = 1280` passed tests, but
the benchmark results were unstable and mixed.

Result: a focused 15-repeat run had one good `n = 1280` packed-`B` median
around 1978 GF/s, while a duplicate same-run `n = 1280` packed-`B` median
collapsed around 1536 GF/s. `n = 1408` remained on AMX packed-`B` around
1950 GF/s. The change was reverted, keeping the public packed-`B` SME cap at
`n <= 1216`.

### 2026-05-05: 32x64 AMX tile mapping rejected

After commit `7523ab7`, a 32x64 AMX tile experiment tried to reuse one
packed-`A` load across two adjacent 32-column `B` panels. The mapping used
`z` rows `0..7` with paired `X` loads.

Result: the experiment did not pass correctness. `make test` failed on multiple
shapes, including `32x64x31`, `64x64x64`, `96x64x128`, `64x96x128`,
`128x128x129`, and the large direct-vs-packed checks for `1152`, `1216`, and
`1280`, with large max diffs. The attempted 32x64 mapping over AMX `Z` storage
is invalid. The patch was reverted, and `make test` then passed 31 shapes.

### 2026-05-05: two-panel public B-pack setup rejected

After commit `ad43943`, a public `B`-pack setup experiment packed two adjacent
32-column `B` panels in one pass over each source row. The intent was to improve
source locality without creating the full write-stream explosion from the
earlier row-wise setup order.

Result: the experiment passed `make test`, but pack setup throughput regressed
versus the existing panel-wise packing. In a 15-repeat
`COB_BENCH_PACK_SETUP=1` run over `512..1536`, setup medians were roughly 106,
94, 96, 92, 90, 90, 83, and 68 GB/s. That was below the earlier panel-wise
baseline of roughly 90-114 GB/s, and especially worse at larger sizes. The
experiment was reverted, so the existing panel-wise packing remains.

### 2026-05-05: skinny one-shot GEMM dispatch

Commit `32e0221` improved one-shot dispatch for LLM-style skinny shapes after
comparing COB against MpGEMM-style workloads. The public MpGEMM repository
currently has no license file, but its README calls the project open-source and
targets skinny DeepSeek/LLaMA workloads.

COB's public packed-`B` path was already strong on examples such as
`64x2112x7168`, `64x24576x1536`, and `64x32768x512`. The weak point was
one-shot dispatch, where full `B` packing dominated runtime. The commit added
an AMX direct-`B` skinny fallback for `m <= 128` and `n >= 1024`, plus a
256-column chunked packed-`B` one-shot path for `m = 96..128`. It also added
direct-vs-packed correctness coverage for `64x2112x512`, `96x4096x1024`, and
`128x2048x2048`.

Full chunking was rejected for `m = 64` because it hurt that row count, so
`m = 64` uses direct-`B`. Skinny SME direct-`B` was worse for larger `K`; only
a narrow `k = 512` fallback remains after chunk allocation. The new path
improves COB one-shot skinny performance, but still does not universally beat
Accelerate on skinny shapes.

Result: validation passed with `make test` across 34 shapes, and
`git diff --check` passed. Correction after the latest isolated validation:
the earlier `m = 64` direct-`B` medians were noisy and overstated. Isolated
15-repeat runs showed `m = 64` skinny direct-`B` is only a modest improvement
over the old full-pack path, with examples around 365-373 GF/s median for
`64x2112x7168`, about 384 GF/s for `64x4096x7168`, about 452 GF/s for
`64x7168x2048`, and about 295 GF/s for `64x32768x512`. It remains below
Accelerate on those one-shot cases, but avoids some full-pack overhead. The
`m = 96/128` chunked path was later improved by `40454ae`; see the follow-up
note below.

### 2026-05-05: skinny chunk packer follow-up

Commit `40454ae` tuned the skinny one-shot chunk packer added after the earlier
dispatch change. Packing panels within each chunk was faster than the initial
row-wise chunk packing order, so the chunked skinny path now keeps the
panel-wise setup shape. The chunk width is also conditional: `m = 128` with
`k >= 1024` uses 512-column chunks, while the other chunked skinny shapes stay
at 256 columns.

Result: validation passed with `make test` across 34 shapes, and
`git diff --check` passed. Focused skinny samples showed about 733 and
839 GF/s medians for `96x4096x1024` and `96x8192x512`, plus about 944, 1007,
and 1008 GF/s medians for `128x4096x1024`, `128x2048x2048`, and
`128x8192x512`. These are still below the packed-`B` path, but closer to
Accelerate than the first chunked one-shot version.

### 2026-05-05: rejected skinny AMX chunk experiments

Two skinny AMX chunk-pack experiments were tried after the `40454ae` follow-up
and reverted. A temporary compile-time chunk-width sweep tested 384, 512, 768,
and 1024 columns against the default 96=>256, 128/k>=1024=>512, otherwise 256
policy. Broad variants were mostly worse; the isolated `M128_K1024=768` case
was mixed and noisy rather than a reliable win, and `M96=1024` was unstable.

The compute loop in `cob_sgemm_rowmajor_amx_skinny_pack_b_chunks` was also
temporarily changed from chunk->panel->ap to chunk->ap->panel. It passed
`make test`, but did not improve enough and regressed large skinny cases:
`128x32768x512` measured around 475 GF/s median versus about 530 GF/s in the
prior clean run. No code from either experiment was kept.

### 2026-05-05: rejected skinny SME packed-B one-shot route

A temporary `COB_USE_APPLE_SME` path packed `A` once, packed `B` in 256-column
chunks, and called the existing `16x64` SME packed-`B` kernel for skinny
one-shot shapes. The experiment passed `make test` while present, but was
reverted because it was not a reliable win.

The wide `m = 64..128` gate was rejected because `m = 64` became much slower
than the AMX direct-`B` fallback. Same-session medians included the SME route at
about 361 GF/s versus AMX/no-SME about 707 GF/s for `64x2112x7168`, about
272 versus 477 GF/s for `64x24576x1536`, and about 248 versus 380 GF/s for
`64x32768x512`. Narrowing to `m = 128, k >= 1024` was still inconclusive or
regressive: duplicate `128x4096x1024` medians with the SME route were about
801 and 855 GF/s versus AMX/no-SME about 889 and 892 GF/s, while
`128x2048x2048` was mixed.

A temporary `build-nosme` directory was used for A/B runs against an SME-disabled
build. `.gitignore` now ignores `build-*` directories so those local comparison
builds stay out of status.

### 2026-05-05: m=64 k=512 large-N SME direct-B route

The existing skinny SME direct-`B` fallback was narrowed into a useful large-`N`
case: it now allows `m = 64` only when `n >= 32768` and `k = 512`. This catches
`64x32768x512` after the AMX chunked path rejects `m = 64`.

Wider gates were rejected. Allowing the route from `n >= 1024` hurt smaller
`m = 64, k = 512` cases such as `64x2112x512` and `64x4096x512`. A threshold
at `n >= 8192` was still mixed and noisy, with weak results around `24576`.
The final narrow route improved duplicate 15-repeat `64x32768x512` one-shot
medians from the AMX/no-SME control around 252-256 GF/s to about 316-317 GF/s.
Accelerate was still faster on that case at roughly 410-415 GF/s.

Validation passed with `make test` across 35 shapes after adding
`64x32768x512` direct-vs-packed correctness coverage.

### 2026-05-05: rejected small-square SME direct-B route

A temporary dispatch experiment lowered the SME direct-`B` medium route to also
try square `192` and `320`, and moved that route ahead of AMX direct-`B` so the
small cases could reach it.

Result: the experiment passed `make test`, but was rejected. `192` regressed
badly in one duplicate 15-repeat run, with medians around 1387-1619 GF/s versus
the AMX direct-`B` baseline around 1696 GF/s. `320` remained below Accelerate,
and moving the medium SME route earlier also regressed routed medium sizes such
as `1088`, `1152`, and `1216`. The code change was reverted.

### 2026-05-05: rejected AMX loop-control experiments

Two more low-risk control-flow experiments were tested and rejected, then
re-evaluated with the paired A/B harness. The AMX fixed-width strided-`B`
panel-loop unroll in `/private/tmp/cob_amx_unroll_exp` passed `make test`, but
paired B/A results for `960`, `1088`, `1152`, and `1216` showed no convincing
win: medians were `0.9954x`, `0.9992x`, `0.9993x`, and `0.9928x`, with
negative trends at `960` and `1216`.

The skinny AMX single-allocation scratch experiment in
`/private/tmp/cob_skinny_single_alloc_exp` also passed `make test`, but paired
skinny-case results were mixed and noisy: medians ranged from `0.9828x` to
`1.0070x`, with one negative trend at `128x8192x512`. No code from either
experiment was kept.

### 2026-05-05: paired benchmark diagnostics tightened

Commit `d36a41c` tightened the paired A/B benchmark harness. It added paired
speedup CV output, lets auto-repeat stop once the paired ratio is stable, prints
a two-tailed sign-test p-value, and exits nonzero on checksum divergence. The
default no-argument shape set was expanded, side-specific compiler flags
`COB_AB_A_FLAGS` and `COB_AB_B_FLAGS` were added, and the design rules were
updated.

The commit also added neutral `COB_SGEMM_SME_DIRECT_MAX_N` compile-time routing
bound for route A/B tests without changing default dispatch.

Result: validation before commit passed with `make all`, `make test` across
37 shapes, `sh -n tools/paired_ab_bench.sh`, `git diff --check`, a paired
self-smoke at `128`, and a side-flag smoke forcing
`COB_SGEMM_AMX_STRIDED_B_MAX_N=128` at `192`.

### 2026-05-05: rejected m=64 skinny KC probe

A compile-flag probe tested `COB_SGEMM_SKINNY_SME_KC=768` against the current
default on `64x2112x7168`, `64x4096x7168`, `64x8192x1024`, and
`64x7168x2048`.

Result: the probe was mixed and noisy. Paired results were `0.9969x` median,
mean-log `1.0046x`, CI `[0.9962,1.0131]` for `64x2112x7168`; `0.9913x`
median, mean-log `0.9890x`, CI `[0.9795,0.9986]` for `64x4096x7168`;
`1.0058x` median, CI `[0.9960,1.0083]` for `64x8192x1024`; and `1.0017x`
median, CI `[0.9952,1.0085]` for `64x7168x2048`. The probe was rejected, with
no source behavior change.

### 2026-05-05: rejected skinny direct-B bypass

The AMX skinny chunked path was temporarily disabled for `m = 96/128` below
`n = 32768`, forcing those shapes onto the AMX direct-`B` path instead. This
tested whether avoiding one-shot chunk packing would help when `B` packing
dominates.

Result: the bypass helped one huge `96x24576x1536` sample, improving the
one-shot median from roughly 380 GF/s to about 472 GF/s, but it badly regressed
common skinny shapes such as `96x4096x1024`, `96x8192x512`, `128x4096x1024`,
`128x2048x2048`, and `128x8192x512`. Large `k = 512` cases still lost to
Accelerate. The experiment was reverted.

### 2026-05-05: K-blocked m=64 SME skinny route

MpGEMM's live FP32 row-SGEMM benchmark was rebuilt from the current checkout
and remains the stronger open-source one-shot skinny blocker. The latest local
baseline measured about 1285 GF/s on `64x2112x7168`, 1071 GF/s on
`64x4096x7168`, 833 GF/s on `64x32768x512`, 1045 GF/s on
`64x24576x1536`, 1099 GF/s on `64x7168x2048`, and 1056 GF/s on
`64x7168x16384`.

COB added a clean-room K-blocked SME direct-`B` route for selected `m = 64`
skinny cases. The route uses `KC = 512`, packs only the current `A` K block,
streams `B`, and accumulates later K blocks into `C`. The dispatch is narrow:
`n = 1024..4096, k >= 7168`, plus the existing long-`N` `k = 512` case.
Wider routing was rejected because it hurt shapes such as `64x24576x1536` and
`64x7168x2048`.

Result: validation passed with `make test` across 37 shapes after adding
`64x2112x7168` and `64x4096x7168` direct-vs-packed coverage. After removing
the failed 64x16 experiment below, the latest clean working-tree `make test`
also passed all 37 shapes.

Latest focused 15-repeat skinny medians:

| Shape | COB one-shot | Accelerate | COB packed-B |
| --- | ---: | ---: | ---: |
| `64x2112x7168` | 1118.16 GF/s | 1131.87 GF/s | 1686.48 GF/s |
| `64x4096x7168` | 785.56 GF/s | 660.13 GF/s | 1702.81 GF/s |
| `64x32768x512` | 364.54 GF/s | 433.14 GF/s | 1366.95 GF/s |
| `64x24576x1536` | 462.60 GF/s | 654.19 GF/s | 1602.07 GF/s |
| `64x7168x2048` | 480.82 GF/s | 720.77 GF/s | 1712.90 GF/s |
| `64x7168x16384` | 508.43 GF/s | 696.10 GF/s | 1688.46 GF/s |
| `64x2112x512` | 1977.31 GF/s | 1949.46 GF/s | n/a |
| `64x4096x512` | 1322.34 GF/s | 1225.73 GF/s | n/a |

### 2026-05-05: rejected 64x16 B-reuse SME experiment

A 64x16 B-reuse SME experiment was tried after the K-blocked `m = 64` route
and removed. It regressed large-`K`, moderate-`N` skinny cases, with one run
around 833 GF/s on `64x2112x7168` and 672 GF/s on `64x4096x7168`. The apparent
long-`N`, `k = 512` gain did not repeat, later measuring only about
314-316 GF/s, so no code from the experiment was kept.

### 2026-05-05: Sapphire package checked

An online search found the `sapphire-compute` PyPI package, which advertises
1.56 TF/s SGEMM on Apple Silicon. The wheel was installed into a temporary
environment and inspected locally. Its native `libsapphire.dylib` links
Accelerate and imports `cblas_sgemm`, and the Python SGEMM wrapper documents
that it uses Apple's Accelerate framework.

Result: it is not a separate open-source single-thread SGEMM blocker for COB's
scope. With `VECLIB_MAXIMUM_THREADS=1`, local `native.sgemm` medians were about
604 GF/s on `64x2112x7168`, 395 GF/s on `64x4096x7168`, 230 GF/s on
`64x32768x512`, 643 GF/s on `64x24576x1536`, 401 GF/s on `64x7168x2048`, and
684 GF/s on `64x7168x16384`.

### 2026-05-05: m=64 SME B-pack reuse route

COB added a clean-room SME fused B-pack/reuse route for
`m = 64, n >= 32768, k = 512`, then generalized it into one `m = 64` SME
B-reuse helper with two narrow gates: `n >= 32768 && k == 512`, and
`n == 4096 && k >= 7168`. It packs all of `A` once, streams `B` and packs a
64-column panel while computing the first 16 rows, then reuses that packed `B`
for the other three 16-row panels. This was inspired by a high-level read of
MpGEMM's row-kernel behavior, but no assembly was copied.

The large-`K` gate is intentionally limited to `n == 4096`; `64x2112x7168`
stays on the existing K-blocked direct SME route because B-pack reuse regressed
that shape.

Result: `make test` passed all 37 shapes, and `git diff --check` passed. After
rebuilding `build/cob_gemm_bench`, a final focused repeat-25 benchmark measured
`64x2112x7168` at 1124.00 GF/s COB one-shot, 1705.78 GF/s packed-`B`, and
1139.19 GF/s Accelerate; `64x4096x7168` at 941.88, 1714.46, and 694.27 GF/s;
`64x4096x8192` at 943.74, 1629.35, and 695.88 GF/s; and `64x32768x512` at
918.91, 1421.23, and 435.15 GF/s.

Follow-up: the B-reuse helper was extended with a third narrow-ish wide gate,
`m = 64, n >= 7168, k >= 1024`, while leaving `64x2112x7168` on the existing
direct K-blocked SME route because B-reuse still regressed that shape. After
that gate, `make test` passed all 37 shapes and `git diff --check` passed. A
focused repeat-25 rebuild measured COB one-shot / packed-`B` / Accelerate at
`64x24576x1536` 924.93 / 1605.80 / 660.99 GF/s, `64x7168x2048` 980.71 /
1739.86 / 721.60 GF/s, `64x7168x16384` 945.91 / 1716.61 / 694.46 GF/s, and
`64x8192x1024` 985.08 / 1701.65 / 618.87 GF/s; the same run had
`64x2112x7168` at 1134.52 GF/s and `64x4096x7168` at 957.96 GF/s. This beats
Accelerate across those wide one-shot cases and narrows the MpGEMM gap, but
still does not universally overtake MpGEMM.

Tuning follow-up, not a new route: after the wide gate, the reuse K chunk was
tuned separately so `m = 64, n >= 7168, k >= 1536` now uses `KC = 768`; the
`n = 4096` large-`K` gate and `k = 1024` wide cases stay at `KC = 512`.
Validation again passed with all 37 `make test` shapes and `git diff --check`.
After rebuild, repeat-25 measured COB one-shot / packed-`B` / Accelerate at
`64x24576x1536` 932.25 / 1592.04 / 668.03 GF/s, `64x7168x2048` 971.08 /
1632.54 / 695.69 GF/s, `64x7168x16384` 951.84 / 1699.92 / 695.85 GF/s,
`64x8192x1024` 1011.06 / 1698.96 / 697.23 GF/s, `64x4096x7168` 959.43 /
1708.23 / 688.80 GF/s, and `64x32768x512` 906.11 / 1205.10 / 403.28 GF/s.

This still does not close the MpGEMM gap on `64x2112x7168` or
`64x4096x7168`, but it narrows `64x4096x7168` and beats the earlier local
MpGEMM `64x32768x512` baseline of about 833 GF/s.

Rejected AMX follow-up: from main commit `5744026`, a temporary experiment in
`/private/tmp/cob_amx_m64_chunk_exp` changed
`cob_sgemm_rowmajor_amx_skinny_pack_b_chunks` to allow
`m == 64, n >= 7168, k >= 1024` with `skinny_nc = 256`. The experiment passed
`/private/tmp/cob_amx_m64_chunk_exp/build/cob_gemm_test` across all 37 shapes,
but repeat-25 one-shot benchmarks were much worse than the existing SME
B-reuse route: `64x24576x1536` measured 271.24 GF/s,
`64x7168x2048` 506.07 GF/s, `64x7168x16384` 238.27 GF/s, and
`64x8192x1024` 474.27 GF/s, versus roughly 930-1010 GF/s for SME B-reuse on
those cases. The better-looking `64x4096x7168` at 946.39 GF/s and
`64x32768x512` at 924.84 GF/s were not useful enough to justify broad AMX
routing, so the AMX chunk route was rejected.

### 2026-05-05: exact 768 one-shot SME direct route

Commit `931daf0` (`Route square 768 through SME direct`) added one narrow
one-shot dispatch exception: exact `768x768x768` now bypasses the AMX
strided-`B` route and uses the existing SME medium direct-`B` path.

Result: full `make test` passed. Focused repeat-25 single-shape runs measured
one-shot medians around 1956.7 and 1961.0 GF/s, versus about 1915.4 GF/s in an
unchanged baseline worktree. Packed-`B` for the same shape stayed around
1969-1974 GF/s.

Rejected follow-ups after `a80d7d4`: Arm's BSD-licensed April 2026 SME2
learning-path sample was built and timed with a throwaway driver, but it was
educational rather than competitive: about 1552 GF/s median at `512`, 1713 at
`1024`, and about 500 on `m = 64` wide shapes. An AMX-only build with
`COB_DISABLE_APPLE_SME=1` confirmed the current SME dispatch is needed for
`m = 64` one-shot and is not a better global square route. Revisited
one-shot `1024` SME-from-packed routing passed tests but was equal or worse
than the current path. A low-threshold SME-before-AMX experiment hurt `192` and
`512` and was mixed at `256` and `384`, so only exact `768` was isolated and
kept. `m = 64` reuse `NC = 256` and `NC = 1024` probes were much slower, and a
`K = 1024` chunk-size probe stayed noisy; none were committed.

Commit `1f5bb7e` (`Route square 384 through SME direct`) added the matching
narrow exception for exact `384x384x384`: it now bypasses AMX strided-`B` and
uses the existing SME medium direct path. `make test` passed. Focused repeat-31
A/B measured main/baseline one-shot around 1696-1705 GF/s median, while the
direct-SME route measured around 1762-1766 GF/s median; the final focused rerun
after the patch measured about 1765 GF/s median at `384` and kept `768` around
1957 GF/s median.

Rejected follow-ups after `1f5bb7e`: a one-shot `1024` direct-SME exception
passed tests but measured around 1750 GF/s median, worse than current
one-shot/Accelerate in that run. A B64 one-shot SME layout for exact `512` and
`1024` passed tests but was not useful: `512` one-shot dropped to about
1583 GF/s median and `1024` was about 1760 GF/s median. Packed-`B` SME A-block
retests with 512-row and 256-row blocks both passed tests but reduced packed-`B`
medians at `768` and `1024` versus the current 384-row block. None were
committed.

Additional post-`1c1e827` rejected probes: exact `256` direct-SME routing was
A/B tested and rejected after the repeat-31 broad low-threshold worktree route
measured about 1491 GF/s median versus current AMX around 1633 GF/s. Medium
exact `1088` direct-SME also failed to hold up after the `384`/`768` wins:
repeat-31 current one-shot was around 1808-1868 GF/s median versus direct-SME
around 1687-1784 GF/s. For `m = 64`, `COB_SGEMM_SKINNY_SME_KC=256` was below
current, with repeat-25 medians around 914 GF/s at `64x8192x1024`, 870 at
`64x4096x7168`, and 775 at `64x32768x512`. Row-major full-panel `B` packing
traversal preserved the packed layout and passed tests, but pack setup fell to
about 19-20 GB/s and one-shot `512`/`1024` collapsed. A fixed NEON 128-byte
copy for `B` panel packing passed tests but did not beat current `memcpy`, whose
main-branch pack setup bandwidth was higher under the same harness. Finally,
the B-panel-first loop order for the SME packed-`B` kernel passed tests but
reduced packed-`B` medians at `768`, `1024`, and `1280`. None were committed.

### 2026-05-05: KleidiAI FP32 SME2 scan

ARM-software/kleidiai was cloned to `/private/tmp/kleidiai_latest` at commit
`7a7da26`. The checkout is Apache-2.0 and reports version `v1.24.0`. It
contains relevant FP32 SME/SME2 matmul microkernels, not just quantized
kernels.

The benchmark was built with `KLEIDIAI_BUILD_BENCHMARK=ON` and
`KLEIDIAI_INTERNAL_EXTRA_ARCH=+sme`. The stock benchmark binary exited before
printing results on Apple, so a temporary direct driver was written at
`/private/tmp/kleidiai_fp32_driver.c`.

Direct elastic FP32 SME2 path median results:

| Shape | Compute-only | One-shot |
| --- | ---: | ---: |
| `384^3` | 1986.77 GF/s | 1715.85 GF/s |
| `768^3` | 2063.71 GF/s | 1911.33 GF/s |
| `1024^3` | 2078.88 GF/s | 1791.06 GF/s |
| `64x4096x7168` | 1341.22 GF/s | 685.78 GF/s |
| `64x8192x1024` | 1317.47 GF/s | 630.13 GF/s |
| `64x32768x512` | 1005.38 GF/s | 535.80 GF/s |

Conclusion: KleidiAI is not a faster one-shot replacement for this repo. Its
compute-only path is an interesting upper-bound/reference, but it requires both
LHS and RHS prepacking, so it is not directly comparable to COB's public
one-shot SGEMM path.

### 2026-05-05: skinny AP-first loop order rejected

An AP-first skinny AMX chunk-loop experiment in
`/private/tmp/cob_skinny_apfirst_exp` swapped the packed-`B` chunk loop from
panel-first to `A`-panel-first. It passed `make all` and `make test`, but the
paired A/B signal was mixed and noisy.

Result: repeat-31, 8-iteration validation showed `96x4096x1024` median
`0.9992x` CI `[0.9909,1.0209]`, `96x8192x512` `1.0012x`
`[0.9934,1.0092]`, `128x4096x1024` `1.0038x` with mean-log `0.9928x`
`[0.9691,1.0101]`, `128x2048x2048` `1.0067x` `[1.0017,1.0126]`, and
`128x8192x512` `1.0077x` `[0.9997,1.0145]`. Smaller-N validation was also
noisy: `96x1024x512` `0.9981x` `[0.9926,1.0087]`, `96x2048x512` `0.9981x`
`[0.9753,1.0367]`, `128x1024x512` `1.0096x` `[0.9993,1.0574]`, and
`128x2048x512` `1.0174x` `[0.9730,1.1025]`. There was no broad
no-regression signal, so the kernel change was not committed.

### 2026-05-05: paired A/B repeat extension

Commit `edfc327` added optional CV-based automatic repeat extension to the
paired A/B benchmark harness. The new knobs are `COB_AB_MAX_REPEATS`,
`COB_AB_REPEAT_BATCH`, and `COB_AB_CV_TARGET`.

Result: default behavior remains fixed-repeat. The harness only extends beyond
`COB_AB_REPEATS` when `COB_AB_MAX_REPEATS` is set above it.

### 2026-05-05: benchmark-boundary infrastructure

Commit `f024172 Add CSV benchmark output` added `COB_BENCH_CSV=1` to
`bench/bench_gemm.c`, emitting
`kind,implementation,m,n,k,best_throughput,median_throughput,unit,best_seconds,median_seconds,checksum`.
README documents the env var. Validation passed `make all`, `make test` across
37 shapes, `git diff --check`, a default text smoke with
`COB_BENCH_REPEATS=1 ./build/cob_gemm_bench 64`, and a CSV plus pack-setup
smoke with `COB_BENCH_REPEATS=1 COB_BENCH_CSV=1 COB_BENCH_PACK_SETUP=1
./build/cob_gemm_bench 64`.

Commit `732453f Add benchmark grid helper` added `tools/bench_grid.sh`, a CSV
wrapper for explicit shape arguments or env Cartesian grids via `COB_GRID_M`,
`COB_GRID_N`, and `COB_GRID_K`, plus `COB_GRID_SIZES`, `COB_GRID_BATCH`, and
`COB_GRID_BENCH`. README documents examples. Validation passed
`sh -n tools/bench_grid.sh`, an explicit-shape smoke with `64 128`, and a grid
smoke with `COB_GRID_M="64 96" COB_GRID_N="1024" COB_GRID_K="512"`.

Result: forcing `832` off current AMX direct-B by compiling with
`COB_SGEMM_AMX_STRIDED_B_MAX_N=768` was rejected. Paired A/B for
`832x832x832` repeat-61 showed candidate median `0.9931x`, mean-log `0.9983x`,
CI `[0.9911,1.0070]`, B-faster `15/61`, sign-p `8.84e-05`; the candidate was
consistently slower despite noisy CV. `896x896x896` stayed neutral with median
`1.0000x`, mean-log `0.9998x`, CI `[0.9900,1.0080]`, sign-p `1`. Keep the
current `832` AMX route; no source behavior change. `/usr/bin/xctrace` was
unusable because the active developer directory is CommandLineTools, so
hardware-counter profiling was deferred.

### 2026-05-05: m=64 k=512 SME reuse threshold lowered

Commit `e5f758d` lowered the default
`COB_SGEMM_M64_SME_LONG_N_K512_MIN_N` from the old implicit `32768` to `4096`,
so the `m = 64, k = 512` SME `B`-reuse route applies from `n >= 4096`. It also
added direct-vs-packed correctness coverage for `64x4096x512` and
`64x8192x512`, raising coverage to 39 shapes.

Before the source change, the grid showed weak old one-shot medians around
561.58 GF/s at `64x6144x512`, 484.54 GF/s at `64x8192x512`, and 259.11 GF/s
at `64x16384x512`, while `64x32768x512` jumped to 809.45 GF/s under the
existing `B`-reuse route. A paired old-threshold versus new-threshold
comparison, with the old threshold forced by
`COB_AB_A_FLAGS=-DCOB_SGEMM_M64_SME_LONG_N_K512_MIN_N=32768`, was neutral at
`64x3072x512` and `64x32768x512` but clearly positive from `n = 4096` through
`16384`: median ratios were `1.0164x` CI `[0.9991,1.0330]` at `3072`,
`1.2932x` CI `[1.2646,1.3234]` at `4096` with B faster `61/61` and sign-p
`8.67e-19`, `1.5324x` CI `[1.4818,1.5686]` at `6144`, `2.2419x` CI
`[2.2258,2.3454]` at `8192`, `2.5033x` CI `[2.4697,2.6239]` at `16384`, and
`0.9942x` CI `[0.9860,1.0059]` at `32768`.

Result: accepted. Final validation passed with `make all`, `make test` across
39 shapes, and `git diff --check`. A focused 15-repeat run after the source
commit measured COB one-shot medians of 1390.86 GF/s versus Accelerate 1109.24
at `64x4096x512`, 1290.55 versus 1021.96 at `64x6144x512`, 1182.53 versus
769.16 at `64x8192x512`, 1017.76 versus 599.86 at `64x16384x512`, and 942.71
versus 621.92 at `64x32768x512`. This closes a clear `m = 64, k = 512`
one-shot gap against Accelerate for `n >= 4096`; the universal fastest claim
is still not fully proven because other MpGEMM and square gaps may remain.

### 2026-05-05: rejected m=96/128 skinny direct-B fallback probe

A temporary compile-time `COB_SGEMM_AMX_SKINNY_CHUNK_MAX_N` guard was tested
with `=4096`, so `m = 96/128, n > 4096, k = 512` fell back to AMX direct-`B`
instead of the chunked skinny route.

Result: rejected, and the temporary guard was removed. Paired results were
neutral at `4096`, but clearly worse for larger `n`: `96x4096x512` median
`1.0000x` CI `[0.9980,1.0164]`, sign-p `0.576`; `96x8192x512` `0.8108x` CI
`[0.7956,0.8247]`, sign-p `8.88e-16`; `96x16384x512` `0.7677x` CI
`[0.7563,0.7781]`, sign-p `8.88e-16`; `128x4096x512` `1.0000x` CI
`[0.9934,1.0067]`, sign-p `0.78`; `128x8192x512` `0.7026x` CI
`[0.6959,0.7086]`, sign-p `8.88e-16`; and `128x16384x512` `0.6396x` CI
`[0.5851,0.6274]`, sign-p `8.88e-16`. Keep the chunked skinny route for these
shapes; no source behavior change came from the probe.

### 2026-05-05: wider skinny k512 SME B-reuse route

Commit `85e7afb` generalized the internal SME B-pack/reuse helper from a
hard-coded `m = 64` / four-16-row-panel shape into a loop over 16-row panels.
The existing `m = 64` routes stayed in place, and the same reuse path is now
enabled for `m = 96/128, k = 512, n >= 4096`. It also added direct-vs-packed
tests for `96x4096x512`, `96x8192x512`, `128x4096x512`, and `128x8192x512`,
raising `make test` coverage to 43 shapes.

Before the change, the grid still showed long-`N` one-shot gaps for
`m = 96/128, k = 512`: examples included `96x8192x512` around 483.96 GF/s COB
median versus 665.54 Accelerate, `96x16384x512` 269.33 versus 657.39,
`128x8192x512` 809.15 versus 924.84, and `128x16384x512` 639.32 versus
765.32. A compile-flag probe with
`COB_SGEMM_M96_128_SME_REUSE_K512_MIN_N=8192` was strongly positive at
`8192` and `16384`, and neutral at `4096`.

Result: accepted with threshold `4096`. Paired threshold validation was neutral
at `2048`, then strongly positive: `96x4096x512` median `1.6720x` CI
`[1.6374,1.6768]` with B faster `61/61`, `96x8192x512` `1.8655x` CI
`[1.8486,1.8806]`, `128x4096x512` `1.5446x` CI `[1.5343,1.5572]`, and
`128x8192x512` `1.6819x` CI `[1.6780,1.6983]`. HEAD-vs-candidate paired runs
checked existing `m = 64` route risk: `64x4096x512`, `64x8192x512`,
`64x4096x7168`, and `64x8192x1024` were neutral/noisy, while the new shapes
won strongly: `96x4096x512` `1.7104x`, `96x8192x512` `1.6743x`,
`128x4096x512` `1.5913x`, and `128x8192x512` `1.5505x`, all B faster
`61/61`.

Validation before commit passed with `make all`, `make test` across 43 shapes,
and `git diff --check`. A focused post-change benchmark measured COB one-shot
median versus Accelerate at `96x4096x512` 1443.20 versus 1011.69 GF/s,
`96x8192x512` 1167.11 versus 837.99, `96x16384x512` 1089.72 versus 672.21,
`128x4096x512` 1542.73 versus 1195.70, `128x8192x512` 1439.33 versus 975.24,
and `128x16384x512` 1138.04 versus 388.83. `96x2048x512` is outside the new
route; one outlier row collapsed, but an immediate duplicate rerun returned to
about 1032-1038 GF/s, matching the paired neutral result. This closes another
skinny `k = 512` one-shot gap for `n >= 4096`; the universal fastest claim is
still not fully proven.

### 2026-05-05: wider skinny k1024 SME B-reuse route

Commit `ff0332b` extended the generalized SME B-reuse helper from `85e7afb`
to `m = 96/128, k >= 1024, n >= 4096`, using
`COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N=4096`. It added direct-vs-packed
tests for `96x8192x1024`, `128x4096x1024`, and `128x8192x1024`, raising
`make test` coverage to 46 shapes.

After the k512 route, the grid still showed k1024 gaps: `96x4096x1024`
measured about 679.58 GF/s COB median versus 891.81 Accelerate,
`96x8192x1024` 580.19 versus 801.70, `96x16384x1024` 456.85 versus 694.98,
`128x4096x1024` 756.69 versus 992.37, `128x8192x1024` 680.88 versus 911.88,
and `128x16384x1024` 549.09 versus 854.21. A compile-flag probe was neutral at
`2048` and strongly positive at `4096+`: `96x4096x1024` median `1.6868x` CI
`[1.6560,1.6986]`, `96x8192x1024` `1.9255x` CI `[1.8973,1.9606]`,
`96x16384x1024` `2.2737x` CI `[2.1768,2.2854]`, `128x4096x1024` `1.6486x` CI
`[1.6269,1.6692]`, `128x8192x1024` `1.8273x` CI `[1.7853,1.8423]`, and
`128x16384x1024` `2.1552x` CI `[2.1216,2.1888]`, with `51/51` B-faster on the
winners.

HEAD-vs-candidate paired validation against the pre-`ff0332b` baseline was
neutral at `96x2048x1024` and `128x2048x1024`, while the routed shapes won:
`96x4096x1024` median `1.7352x` CI `[1.6622,1.7840]`, `96x8192x1024`
`2.0858x` CI `[1.9582,2.0931]`, `128x4096x1024` `1.6755x` CI
`[1.6364,1.7290]`, and `128x8192x1024` `1.9331x` CI `[1.8644,1.9405]`.
Existing k512 shapes also remained strong.

Result: accepted. Validation before commit passed with `make all`,
`make test` across 46 shapes, and `git diff --check`. Focused post-change
COB one-shot medians versus Accelerate were mostly strong wins:
`96x8192x1024` 1141.47 versus 826.80 GF/s, `96x16384x1024` 1147.98 versus
767.51, `128x4096x1024` 1350.62 versus 975.24, `128x8192x1024` 1336.33
versus 981.93, and `128x16384x1024` 1256.21 versus 803.85. The first
`96x4096x1024` median was noisy, but duplicate reruns measured COB one-shot
1066.63 and 1214.64 GF/s versus Accelerate 731.43 and 759.01. This closes
another skinny one-shot k1024 gap for `n >= 4096`; the universal fastest claim
is still not fully proven.

Follow-up commit `aa1811e` covered higher-`k` samples for the same route. A
grid check over `m = 96/128`, `n = 4096/8192`, and `k = 1024/2048/4096` kept
the skinny SME reuse path faster than Accelerate in those samples. It added
direct-vs-packed correctness tests for `96x4096x2048` and `128x8192x4096`,
raising `make test` coverage to 48 shapes. Validation passed with `make all`,
all 48 test shapes, `git diff --check`, and a pushed commit.

### 2026-05-05: four-panel B-pack setup rejected

After `9b7ef5f`, a temporary `COB_SGEMM_PACK_B32_GROUP4=1` experiment packed
four adjacent 32-column `B` panels row-by-row. The intent was to improve large
one-shot B-pack setup/locality while preserving the packed layout. It built and
`make test` passed all 48 shapes while present, but performance did not hold.

Result: paired one-shot A/B with the candidate flag showed only weak/noisy
positive movement at `1280` with median `1.0134x`, CI `[0.9888,1.0489]`, and
sign-p `0.262`. It regressed `1408` at `0.9545x`, CI `[0.9057,0.9566]`,
sign-p `2.42e-10`, and `1536` at `0.9625x`, CI `[0.9445,0.9705]`, sign-p
`1.83e-08`. `2048` was inconclusive/noisy at median `0.9852x`, CI
`[0.9930,1.0943]`, sign-p `0.161`, and skinny routed checks
`96x4096x1024` / `128x4096x1024` were neutral/noisy. Reject; the temporary code
was removed, `make test` again passed all 48 shapes, `git diff --check` passed,
and there is no source behavior change.

### 2026-05-05: route-aware gap sweep follow-up

A route-aware sweep was used to inspect medium-square and `m = 64` skinny gaps.
Forcing `1024` square onto the AMX strided-`B` route with
`COB_SGEMM_AMX_STRIDED_B_EXTRA_N2=1024` was rejected: 101 repeats measured
about `0.934x` median, bootstrap95 `[0.8828x,0.9414x]`, with B faster in
`18/101` samples.

Forcing `COB_SGEMM_M64_SME_REUSE_NC=256` was also rejected/inconclusive.
`64x4096x7168` was neutral with holdout around `1.000x`, and the other wide
shapes did not show a commit-worthy win.

Tooling note: `tools/bench_gap_report.py` was added to rank route-labelled CSV
benchmark gaps reproducibly. Validation covered `python3 -m py_compile`, a live
CSV smoke through the helper, and `git diff --check`.

### 2026-05-05: SME medium direct cap tightened

The route-aware gap report highlighted medium-square gaps. A no-SME paired A/B
rejected disabling SME for `384`, `768`, and `896`: candidate medians were
roughly `0.9625x`, `0.9858x`, and `0.8788x`. The same sweep showed the `1280`
SME medium direct route had become harmful.

An isolated gate probe with `COB_SGEMM_SME_DIRECT_MAX_N=1216` improved `1280`
by `1.2640x` at 101 repeats, bootstrap95 `[1.2428x, 1.2813x]`, with B faster
in `100/101` samples. `1216` and `1152` were neutral/noisy, so the accepted
change capped SME medium direct at `1216`.

Post-change A/B against the old `1280` default still showed `1280` at
`1.0545x`, bootstrap95 `[1.0534x, 1.0870x]`, while `1216` stayed neutral.
Validation passed with `make test` across 49 shapes, `make all`, and
`git diff --check`.

### 2026-05-05: packed-B large-square AMX row block

The route-aware gap report showed large-square packed and one-shot areas were
still worth checking. The AMX `MC` row block was made compile-time tunable and
`MC = 256` / `512` were probed across one-shot and public packed-`B` routes.

Result: one-shot large-square `MC = 256` and `512` were rejected/neutral, so
one-shot keeps `MC = 384`. Packed-`B` `MC = 256` was rejected because it
regressed `1536` and was neutral elsewhere. Packed-`B` `MC = 512` was accepted
for `n >= 2048`: `2048` packed-`B` improved about `1.007-1.010x`, with an
agreeing holdout, while `1152`, `1216`, and `1792` were neutral.

The accepted source change uses packed-`B`-only
`COB_SGEMM_AMX_PACKED_LARGE_MC=512` for `n >= 2048`, leaving one-shot at
`MC = 384`. Validation passed with `make test` across 50 shapes, `make all`,
and `git diff --check`. A route smoke in that session showed `2048` packed-`B`
median above Accelerate.

### 2026-05-05: B-pack prefetch distance increase

Commit `7fae628` increased the default
`COB_SGEMM_PACK_B_PREFETCH_DISTANCE` from `16` to `64`. An earlier sweep after
the default-16 baseline tested candidate distances `32`, `64`, and `128`
against `16`; `64` was best overall.

Focused post-change A/B compared the new default against the old distance with:
`env COB_AB_REPEATS=101 COB_AB_MAX_REPEATS=101 COB_AB_REPEAT_BATCH=10
COB_AB_CV_TARGET=2 COB_AB_A_FLAGS=-DCOB_SGEMM_PACK_B_PREFETCH_DISTANCE=16 sh
tools/paired_ab_bench.sh src/gemm.c src/gemm.c 1280 1536 2048`.

Result: `1280` median `1.0074x`, bootstrap95 `[0.9979,1.0275]`, B faster
`76/101`, sign-p `3.72e-07`, holdout median `1.0069x`
`[1.0030,1.0139]`; `1536` median `1.0109x`, `[1.0009,1.0120]`, B faster
`86/101`, sign-p `2.83e-13`, holdout `1.0109x` `[1.0023,1.0143]`; and `2048`
median `1.0198x`, `[1.0146,1.0454]`, B faster `95/101`, sign-p `1.07e-21`,
holdout `1.0193x` `[1.0044,1.0478]`.

Checksums matched. Validation passed with `make all`, `make test` across 50
shapes, and `git diff --check`. A broader post-change run showed `512` as tiny
positive/noisy and `1024` as neutral/noisy, so the accepted win is for larger
one-shot packed-`B` route shapes.

### 2026-05-05: wide m64 SME K chunk increase

Commit `ea7c963` increased `COB_SGEMM_M64_SME_WIDE_KC` from `768` to `1024`
for the wide `m = 64` SME `B`-reuse route when `k >= 1536`.

Rejected probes stayed out of source. Adding `192x192x192` to the existing SME
medium direct route passed `make test`, but paired A/B measured only `0.8329x`
median, bootstrap95 `[0.8263,0.8469]`, B-faster `1/101`, and holdout
`0.8305x`, so it was reverted. `m = 64` reuse probes also failed to hold:
`REUSE_NC=1024` regressed, `REUSE_NC=256` was weak/noisy, `REUSE_NC=768`
regressed guard shapes, and `SKINNY_SME_KC=768` / `384` regressed or did not
help `64x2112x7168` and `64x4096x7168`.

The accepted `WIDE_KC=1024` change was positive in broad A/B against old `768`:
`64x7168x1536` median `1.0031x` CI `[1.0017,1.0116]`,
`64x7168x2048` `1.0187x` CI `[1.0161,1.0317]` with holdout `1.0166x`,
`64x7168x4096` `1.0139x` CI `[1.0055,1.0157]` with holdout `1.0139x`,
`64x7168x16384` `1.0089x` CI `[1.0031,1.0136]`, and `64x24576x1536`
`1.0077x` CI `[1.0066,1.0244]`. Guard shapes `64x8192x1024`,
`64x4096x7168`, and `64x32768x512` were neutral/noisy.

Post-change repeat-101 validation comparing old `768` against the new default
confirmed the target gains: `64x7168x2048` median `1.0181x` CI
`[1.0136,1.0203]`, B-faster `86/101`; `64x7168x4096` `1.0175x` CI
`[1.0096,1.0412]`, B-faster `83/101`; `64x7168x16384` `1.0121x` with noisy CI
`[0.9958,1.0416]`, B-faster `79/101`; and `64x24576x1536` `1.0095x` with
noisy CI `[0.9902,1.0186]`, B-faster `68/101`. Guard shapes stayed neutral.

Validation passed with `make test` across 50 shapes, `make all`, and
`git diff --check`. The MpGEMM O0-driver refresh after the change was noisy but
showed targeted COB wide-shape best/median uplift. The fastest claim is still
not complete because MpGEMM remains ahead on several `m = 64` shapes.

### 2026-05-05: exact k1536 wide m64 full chunk

Commit `1b23220` (`Use full K chunk for wide m64 k1536`) added
`COB_SGEMM_M64_SME_WIDE_K1536_KC=1536` and uses the full K chunk only for the
wide `m = 64` SME reuse route when `k == 1536`. Other wide `k >= 1536` cases
keep `COB_SGEMM_M64_SME_WIDE_KC=1024`.

A broader replacement making `WIDE_KC=1536` the global default was rejected: it
helped exact `k = 1536`, but was weak or worse at `k >= 2048`, including
`64x7168x2048` median `0.9906x` in one sweep, and was noisy elsewhere.

The accepted narrow candidate was compared against the old exact-`k1536` value
with `COB_AB_A_FLAGS=-DCOB_SGEMM_M64_SME_WIDE_K1536_KC=1024` and
`repeats=101`. Results: `64x7168x1536` median `1.0246x`, bootstrap95
`[1.0046,1.0396]`, B-faster `92/101`, holdout `1.0258x`; and
`64x24576x1536` median `1.0395x`, bootstrap95 `[1.0364,1.0521]`, B-faster
`98/101`, holdout `1.0389x`. Guard shapes were neutral/noisy:
`64x7168x2048` median `0.9983x`, `64x7168x4096` `1.0011x`, and
`64x8192x1024` `0.9971x`.

Rejected probes from the same pass stayed out of source. Forcing `192` and
`256` off AMX strided-`B` with `COB_SGEMM_AMX_STRIDED_B_MAX_N=128` was a hard
regression, with medians `0.7040x` and `0.7809x`. SME skinny `B` prefetch
distance `16` was also a hard regression on `m = 64` skinny shapes, including
`64x7168x2048` median `0.7693x` and `64x8192x1024` `0.8273x`.

Validation passed with `make test` across 50 shapes, `make all`, and
`git diff --check`.

### 2026-05-05: wide m64 SME large-K chunk

Commit `0a20d56` (`Tune wide m64 large-K chunk`) added
`COB_SGEMM_M64_SME_WIDE_LARGE_KC=1280` and uses it only for the wide `m = 64`
SME reuse route when `k >= 8192`. Exact `k == 1536` keeps the previous full-K
chunk, while `k = 2048` and `k = 4096` keep `WIDE_KC=1024`.

Rejected wider defaults stayed out of source. `WIDE_KC=2048` regressed
`k >= 2048` shapes in the A/B sweep, including `64x7168x2048` median
`0.9848x`, `64x7168x4096` `0.9843x`, and `64x7168x16384` `0.9871x`.
`WIDE_KC=1280` as a global non-1536 replacement was mixed: neutral/worse at
`k = 2048` and `k = 4096`, but positive at `64x7168x16384` median `1.0061x`,
CI `[1.0024,1.0182]`.

Focused large-K validation for the guarded `k >= 8192` rule compared against
old `1024`. The first sweep showed `64x7168x8192` median `1.0057x` with
holdout `1.0067x`, `64x7168x16384` median `1.0074x` CI
`[1.0051,1.0385]`, and `64x24576x8192` median `1.0073x` with holdout
`1.0067x`. Post-change repeat-101 validation was weaker but acceptable:
`64x7168x8192` median `1.0026x` with positive holdout `1.0048x`,
`64x7168x16384` median `1.0040x` with sign-p `0.00265` but weak holdout, and
`64x24576x8192` median `1.0064x` CI `[1.0002,1.0189]` with holdout `1.0061x`.
Guard shapes stayed neutral: `64x7168x2048` median `1.0006x`,
`64x7168x4096` `1.0018x`, and `64x8192x1024` `0.9970x`.

Other rejected probes from the same pass: `-Ofast` and `-mcpu=native` were not
broadly useful; disabling m64 reuse with `REUSE_NC=0` caused hard regressions;
forcing small `192` / `256` off AMX strided-`B` also hard-regressed; and SME
skinny `B` prefetch distance `16` hard-regressed m64 skinny shapes.

Validation passed with `make test` across 50 shapes, `make all`, and
`git diff --check`.

### 2026-05-05 local-uncommitted: n1280-1408 SME direct route

The route-aware gap sweep pointed at extra medium widths where the SME direct
path was still useful, but broad caps were too blunt. The local candidate routes
rectangular `m = 832..960, n = 1280..1472, k = 832..1152` and same-M/N
`n = 1280..1408, k = 832..960`. Same-M/N `1472` and `k = 1280` stay on AMX
packed after paired validation failed to hold up cleanly.

Paired validation against `/private/tmp/cob_gemm_baseline_before_strided192_block.c`
with `COB_AB_ITERS=3` showed accepted-shape wins:

- `832x1280x832` median `1.1276x`, B-faster `60/61`, holdout `1.1263x`.
- `960x1280x960` median `1.1169x`, B-faster `60/61`, holdout `1.1135x`.
- `1280x1280x832` median `1.0881x`, B-faster `59/61`, holdout `1.0877x`.
- `832x1344x832` median `1.0968x`, B-faster `61/61`, holdout `1.0983x`.
- `960x1344x960` median `1.0971x`, B-faster `60/61`, holdout `1.0933x`.
- `1344x1344x832` median `1.0600x`, B-faster `52/61`, holdout `1.0605x`.
- `832x1408x832` median `1.1281x`, B-faster `61/61`, holdout `1.1281x`.
- `960x1408x960` median `1.1131x`, B-faster `61/61`, holdout `1.1109x`.
- `1408x1408x832` median `1.0847x`, B-faster `61/61`, holdout `1.0845x`.
- Rectangular `k = 1152` extension wins included `832x1280x1152` median
  `1.1328x`, `960x1280x1152` `1.1257x`, `832x1344x1152` `1.1098x`,
  `960x1344x1152` `1.0950x`, `832x1408x1152` `1.1205x`,
  `960x1408x1152` `1.0824x`, `832x1472x1152` `1.1076x`, and
  `960x1472x1152` `1.0928x`, all with positive holdouts.

Focused repeat-101 validation with `COB_AB_ITERS=5` kept the same-M/N `k = 960`
cases: `1280x1280x960` median `1.0901x`, `1344x1344x960` `1.0698x`, and
`1408x1408x960` `1.0881x`, all with positive holdouts and sign-p below
`1e-18`.

Guard shapes stayed neutral or were rejected: `1280x1280x1152` median
`0.9987x`, `1344x1344x1152` `1.0001x`, `1408x1408x1152` `1.0011x`, and
same-M/N `1472x1472x832` stayed off the route. Validation passed with
`make test` across 60 shapes, `make all`, route-label smoke, and
`git diff --check`.

### 2026-05-05 local-uncommitted: cache-fit wide m64 blocking rejected

Apple M5 Max reports `hw.l2cachesize = 8388608` and `hw.cachelinesize = 128`.
That makes larger `m = 64` `NC/KC` combinations look plausible under the
`mc*kc + kc*nc + mc*nc < L2` capacity model, but the local paired probe did not
validate the idea.

Using `COB_AB_B_FLAGS='-DCOB_SGEMM_M64_SME_REUSE_NC=768
-DCOB_SGEMM_M64_SME_LONG_WIDE_NC=512 -DCOB_SGEMM_M64_SME_WIDE_KC=1536
-DCOB_SGEMM_M64_SME_WIDE_LARGE_KC=2048'` against current source regressed key
wide shapes: `64x7168x2048` median `0.9741x`, `64x7168x4096` `0.9722x`, and
`64x7168x8192` `0.9426x`. `64x4096x7168`, `64x24576x1536`,
`64x24576x8192`, and `64x8192x1024` were neutral/noisy. Do not promote
cache-fit blocking parameters without paired validation on this hardware.

### 2026-05-05 local-uncommitted: route-specific cache-fit probes rejected

The route-by-route cache-derived blocking pass was bounded to compile-flag A/B
probes. Local hardware context: M5 Max, 64 KB L1d, 8 MB L2 per P-cluster,
16 KB pages, and 128 B cache lines. This makes direct porting of M4 Pro blocking
constants invalid.

- AMX one-shot large-block interaction: `COB_SGEMM_AMX_MC=512` was neutral/noisy
  across `512x1152x512`, `768x1152x768`, `1024x1152x1024`,
  `1152x1152x512`, and `1216x1216x768`; `COB_SGEMM_AMX_MC=256` slightly
  regressed `768x1152x768` and was otherwise neutral. Keep `384`.
- AMX public packed-B large block: `COB_SGEMM_AMX_PACKED_LARGE_MC=384`
  regressed `2048` and `2048x3072x1024`; `768` regressed `2048` and was noisy
  elsewhere. Keep `512`.
- SME public packed-B A-blocking: shared `COB_SGEMM_AMX_MC=512` was neutral to
  slightly worse at `1152`; `256` was neutral/noisy with weak regressions. Keep
  `384` for the shared SME packed-B A block.
- SME B-reuse: the combined larger `NC/KC` probe in the previous entry regressed
  key wide `m = 64` shapes. Keep the empirically tuned current constants.

Conclusion: the current empirically tuned route constants are a better local
default than simple cache-capacity-derived replacements on this M5 Max.

### 2026-05-05 local-uncommitted: m64 k1536/k2048/k4096 SME skinny threshold accepted

After the large-`K` streaming-B prefetch gate, a route-aware gap sweep showed
`m = 64, k = 1536/2048/4096, n <= 4096` still trailing badly because those
shapes used AMX strided-B. A temp source lowered the SME skinny direct route
from `k >= 7168` to `k >= 2048`; a follow-up accepted a narrower `k = 1536`
gate from `n >= 1536`. The existing prefetch gate applies on non-512 `n`
widths, while 512-multiple widths use the unprefetched SME direct kernel.

Paired validation versus the pre-change source was strongly positive:

- `64x1088x2048` median `1.1000x`, bootstrap95 `[1.1029,1.1416]`,
  B-faster `61/61`; holdout median `1.1000x`.
- `64x1024x2048` repeat-101 median `1.2267x`, bootstrap95
  `[1.2042,1.2772]`, B-faster `100/101`; holdout median `1.2256x`.
- `64x1856x2048` median `1.3531x`, bootstrap95 `[1.3307,1.3781]`,
  B-faster `61/61`; holdout median `1.3675x`.
- `64x2048x2048` median `1.8954x`, bootstrap95 `[1.8057,1.9081]`,
  B-faster `60/61`; holdout median `1.8899x`.
- `64x4032x2048` median `2.4147x`, bootstrap95 `[2.3907,2.4454]`,
  B-faster `61/61`; holdout median `2.4147x`.
- `64x1088x4096` median `1.4040x`, bootstrap95 `[1.3874,1.4312]`,
  B-faster `61/61`; holdout median `1.3803x`.
- `64x1024x4096` repeat-101 median `1.9355x`, bootstrap95
  `[1.8937,1.9584]`, B-faster `101/101`; holdout median `1.9539x`.
- `64x2048x4096` median `2.0111x`, bootstrap95 `[2.0104,2.0342]`,
  B-faster `61/61`; holdout median `2.0111x`.
- `64x4032x4096` median `2.4947x`, bootstrap95 `[2.4742,2.5060]`,
  B-faster `31/31`; holdout median `2.4858x`.

The `k = 1536` follow-up is narrower because low widths regressed:
`64x1088x1536` median `0.9464x`, `64x1152x1536` `0.9580x`, and
`64x1280x1536` `0.9706x`. From `n >= 1536`, the route was positive:
`64x1536x1536` repeat-101 median `1.1390x`, `64x1856x1536` `1.0890x`,
`64x2048x1536` `1.3593x`, `64x2560x1536` `1.8034x`,
`64x3328x1536` `1.9378x`, `64x4032x1536` `2.1315x`, and
`64x4096x1536` `1.5214x`, all with positive holdouts. The paired harness
checksum guard fired at `64x2560x1536` because the sampled checksum was near
zero; a direct-vs-packed max-diff check on that shape measured `0.000118`,
inside the existing `0.002` correctness tolerance, so the narrow gate was
accepted with added correctness coverage.

Follow-up rejected: compiling the candidate side with `-funroll-loops` did not
improve the m64 SME tuple/reuse routes. Paired checks were neutral or slightly
negative on `64x7168x1024`, `64x7168x2048`, `64x7168x4096`,
`64x8192x1024`, `64x8192x2048`, `64x16384x2048`, `64x24576x1536`,
`64x4096x7168`, and `64x2112x7168`; guard shapes `96x4096x1024` and
`832x1472x1152` were also neutral. This does not rule out a hand-written
targeted K-unroll, but a broad compiler-unroll flag is not useful.

Follow-up rejected: hand-written K-unrolls in the prefetched m64 SME
streaming-B kernel did not beat the current single-iteration loop. A K=4
variant in `/private/tmp/cob_m64_unroll_exp` passed all 63 correctness shapes,
but paired A/B regressed the remaining high-`K` gaps: `64x4032x8192` median
`0.8264x`, `64x3712x12288` `0.9220x`, `64x3328x12288` `0.9383x`, and
`64x2112x7168` `0.8741x`. A lower-pressure K=2 variant in
`/private/tmp/cob_m64_unroll2_exp` also passed correctness but stayed
noise-grade: `64x4032x8192` median `0.9987x`, `64x3712x12288` `1.0019x`,
`64x3328x12288` `1.0026x`, `64x4096x7168` `1.0006x`,
`64x2048x2048` `0.9933x`, and `64x2560x1536` `0.9976x`. The only positive
line was `64x2112x7168` median `1.0061x`, bootstrap95 `[1.0047,1.0125]`,
B-faster `65/93`, holdout median `1.0038x`, which is too small and isolated to
justify extra inner-loop complexity. Keep the compiler's current K loop until
counter evidence or an assembly kernel shows a different schedule is needed.

Follow-up rejected: retuning `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE` away from
the default `32` did not produce a broad improvement. Distance `16` regressed
`64x2112x7168` (`0.9859x`) and was neutral/mixed on the remaining high-`K`
checks. Distance `64` improved `64x4032x8192` (`1.0310x`) and slightly helped
`64x2112x7168` (`1.0105x`), but was neutral or negative on neighboring
`64x3712x12288`, `64x3328x12288`, `64x4096x8192`, and `64x3968x8192`.
Keep distance `32`.

Hardware-counter follow-up remains blocked in this local environment. `mperf`
is not installed, Homebrew search does not expose an `mperf`/`kperf` formula,
`/Applications/Xcode.app` is absent, and `/usr/bin/xctrace` still exits with
`tool 'xctrace' requires Xcode` because the active developer directory is
`/Library/Developer/CommandLineTools`. Skip counter-driven tuning until full
Xcode or a working local counter tool is available.

Follow-up accepted: a cache-blocking probe found a narrow `m = 64, k = 2048`
band where `COB_SGEMM_SKINNY_SME_LARGE_KC=1024` beats the default `KC=512`.
The source gate now keeps the existing `n = 1024` large-KC case and adds only
`n = 1088..1280` plus exact `n = 1600`; nearby uncertain or negative widths
stay on `KC=512`.

The accepted source gate passed `make test` with 67 shapes, `make all`, and
`git diff --check`. Paired source validation against the pre-change source:
`64x1088x2048` repeat-101 median `1.0526x`, bootstrap95 `[1.0221,1.0728]`,
B-faster `95/101`; `64x1152x2048` `1.0373x`, `[1.0163,1.0605]`,
B-faster `82/101`; `64x1280x2048` `1.0429x`, `[1.0337,1.0595]`,
B-faster `75/101`; and `64x1600x2048` `1.0446x`, `[1.0343,1.0638]`,
B-faster `90/101`. Guard shapes not selected by the new gate stayed neutral
within noise: `64x1408x2048`, `64x1472x2048`, `64x1536x2048`,
`64x1664x2048`, `64x1792x2048`, `64x2048x2048`, `64x2560x2048`,
`64x1600x1536`, `64x4032x8192`, and `96x4096x1024`.

Rejected cache-blocking probes: global `COB_SGEMM_SKINNY_SME_KC=384`, `768`,
or `1024` was not viable. `KC=384` regressed high-`K` shapes such as
`64x4032x8192` (`0.9764x`), `64x3328x12288` (`0.9811x`), `64x4096x7168`
(`0.9822x`), and `64x2112x7168` (`0.9638x`). Global `KC=768` regressed
`64x4032x8192` (`0.9554x`), `64x3712x12288` (`0.9837x`),
`64x2048x2048` (`0.9081x`), and other guards. Global `KC=1024` exposed the
accepted low-width `k = 2048` band, but regressed `64x4032x8192` (`0.8297x`),
`64x3712x12288` (`0.8510x`), `64x3328x12288` (`0.8856x`), `64x2048x2048`
(`0.9630x`), and `64x2560x1536` (`0.9231x` in the boundary sweep). The
`k = 1536` and high-`K` low-width positives were too noisy or discontinuous to
ship.

Follow-up rejected: forcing the m64 SME skinny direct route to use full-K
chunks, as a low-effort proxy for a single-store epilogue, was not viable.
The candidate in `/private/tmp/cob_m64_fullkc_exp` passed correctness, but
paired A/B was catastrophic beyond a tiny low-width pocket: `64x2048x2048`
`0.5336x`, `64x2112x7168` `0.6881x`, `64x4032x8192` `0.6074x`, and
`64x3712x12288` `0.5541x`. A narrower sweep showed only isolated low-width
`k = 2048` positives, mainly `64x1088x2048` `1.0138x` and `64x1152x2048`
`1.0129x`, while `64x1024x2048`, `64x1216x2048`, `64x1280x2048`, and all
tested `k = 4096` low-width shapes regressed. Do not replace the current
K-blocking policy with full-K chunks; a real single-store epilogue would need a
more careful structure than simply growing `KC` to `k`.

Follow-up accepted: the `m = 64, k = 1536` SME skinny route was lowered from
`n >= 1536` to `n >= 1408`, with the streaming-B prefetch subroute lowered to
the same boundary. This targets the previous `64x1408x1536` and
`64x1472x1536` front-door gaps without reopening the rejected `1088..1280`
region. The accepted source behavior passed `make test` with 69 shapes,
`make all`, and `git diff --check`.

Paired source validation against the pre-change source was strong on the new
sizes: `64x1408x1536` repeat-101 median `1.2831x`, bootstrap95
`[1.2460,1.3062]`, B-faster `100/101`, holdout median `1.2919x`; and
`64x1472x1536` median `1.2395x`, bootstrap95 `[1.2382,1.3011]`, B-faster
`100/101`, holdout median `1.2312x`. Guard shapes stayed neutral within noise:
`64x1152x1536`, `64x1280x1536`, `64x1536x1536`, `64x1600x1536`,
`64x1728x1536`, and `64x2048x1536`.

Follow-up accepted: the public packed-B dispatcher now keeps high-`K` packed-B
shapes on AMX instead of the current SME packed-B kernel when `k >= 4096`, and
also keeps `n = 1152, k >= 2048` on AMX. The clean gate in
`/private/tmp/cob_packed_amx_gate_exp` passed all correctness tests and a
focused packed-B paired A/B rerun showed the suspicious non-selected neighbor
`512x1024x2048` was neutral (`1.0148x`, CI `[0.9836,1.0254]`, sign-p `0.32`,
holdout `1.0086x`). Selected shapes were strong: `512x1152x2048` `1.1760x`
with B-faster `97/101`, `1024x1152x2048` `1.1044x` with B-faster `101/101`,
`512x1024x4096` `1.6663x`, `1024x1024x4096` `1.6446x`,
`512x1152x4096` `1.7688x`, and `1024x1152x4096` `1.7397x`, all four
`k = 4096` selected shapes with B-faster `101/101`. The change adds packed-B
correctness coverage for `512x1152x2048`, `512x1024x4096`, and
`1024x1152x4096`.

Follow-up accepted: the one-shot AMX medium dispatcher now avoids strided-B for
the high-`K` medium shapes where packing B wins even after one-shot pack cost.
The gate is intentionally narrower than the first probe: `m >= 512, k >= 4096`
uses the packed path; `n = 1152, k >= 3072` uses the packed path; and
`n = 1152, k = 2048` only uses the packed path at `m >= 1024`.

The broad `k >= 4096` validation was strong across neighboring widths:
`512x832x4096` `1.2144x`, `768x960x4096` `1.2667x`,
`1024x1088x4096` `1.5542x`, and `1024x1216x4096` `1.6352x`, with selected
checks showing B-faster counts of at least `80/81`. The refined boundary kept
`512x1152x2048` and `768x1152x2048` neutral, while still improving
`1024x1152x2048` (`1.0553x`, CI `[1.0485,1.0659]`, B-faster `99/101`),
`512x1152x3072` (`1.2808x`, B-faster `101/101`), and
`1024x1152x3072` (`1.4104x`, B-faster `101/101`). The `n = 768, k = 3072`
guard stayed neutral, so that boundary remains on strided-B. Correctness
coverage added `1024x1152x2048`, `512x1152x3072`, and `512x960x4096`.

Follow-up accepted: the one-shot `n = 512` conflict path now uses packed AMX
instead of the SME packed-B kernel at `k >= 2048`, while keeping SME at
`k <= 1024`. The first probe that switched from `k >= 1024` was rejected because
`512x512x1024`, `768x512x1024`, and `1024x512x1024` regressed or stayed noisy.
The refined `k >= 2048` gate was positive: `512x512x2048` median `1.0211x`,
CI `[1.0145,1.0292]`, B-faster `71/101`; `768x512x2048` `1.0228x`,
B-faster `82/101`; `1024x512x2048` `1.0206x`, B-faster `73/101`; and
`512/768/1024 x 512 x 3072` at `1.0272x`, `1.0456x`, and `1.0327x` with
holdout agreement. Correctness coverage added `512x512x2048` and
`768x512x3072`.

Follow-up accepted: the public packed-B AMX large-block threshold now keeps
exact `m = 384, n >= 2048, k >= 1024` on a 384-row block instead of requiring
the generic 512-row block threshold. The source change was found as an
uncommitted helper extraction and then validated against a clean `HEAD`
worktree before keeping it. Packed-B paired A/B showed `384x2048x1024`
`1.0383x`, `384x2048x2048` `1.2262x`, `384x2048x4096` `1.1716x`,
`384x4096x1024` `1.1510x`, `384x4096x2048` `1.1962x`, and
`384x4096x4096` `1.1313x`. Guard shapes stayed neutral:
`512x2048x2048` `1.0033x`, `512x4096x2048` `1.0009x`, and
`768x2048x2048` `0.9991x`. Correctness coverage added `384x2048x2048` and
`384x4096x1024`.

Follow-up rejected: forcing `n = 768, k >= 2048` one-shot medium shapes from
AMX strided-B to packed AMX was a hard regression at `k = 2048` and `3072`,
including `512x768x2048` `0.8764x`, `768x768x2048` `0.9201x`,
`1024x768x2048` `0.9313x`, `768x768x3072` `0.8937x`, and
`1024x768x3072` `0.9366x`. Keep that width on strided-B below the existing
`k >= 4096` packed-path gate.

Follow-up rejected: extending the SME medium direct path to exact
`768x768x2048` was neutral/noisy, not an improvement. The candidate passed
correctness, but paired A/B measured `768x768x2048` `0.9952x` with CI
`[0.9912,1.0363]` and holdout `0.9984x`. Neighboring guards were also neutral,
so leave the current AMX strided-B route.

Follow-up rejected: broad `COB_SGEMM_AMX_MC=512` cache blocking was not viable
for the current high-`K` medium packed routes. It was neutral at some
`n = 1024` shapes, but regressed the important `n = 1152, k = 4096` one-shot
band: `512x1152x4096` `0.9548x`, `768x1152x4096` `0.9552x`, and
`1024x1152x4096` `0.9313x`. Keep the current route-specific 384/512 policy.

Follow-up rejected: changing `COB_SGEMM_PACK_B_PREFETCH_DISTANCE` from the
current `64` to `128` was neutral/noisy, and lowering it to `32` regressed
several pack-heavy one-shot cases such as `512x768x4096` `0.9921x`,
`768x512x4096` `0.9876x`, `512x1152x4096` `0.9756x`, and
`512x1024x2048` `0.9906x`. Keep the current distance `64`.

Follow-up accepted: the one-shot packed-path gate was extended to the
low-height `m = 384` high-`K` medium band. Direct repeat-9 benchmarking showed
`384x1152x3072` and `384x1152x4096` still used AMX strided-B and trailed the
packed path badly even after pack setup. A paired source gate in
`/private/tmp/cob_m384_highk_pack_exp` passed correctness and validated the
narrow rule: `m = 384, n >= 1152, k >= 4096`, plus exact
`m = 384, n = 1152, k >= 3072`. Wins were large on the target band:
`384x1152x3072` `1.2427x`, CI `[1.2339,1.2572]`, B-faster `101/101`;
`384x1152x4096` `1.5692x`, CI `[1.5609,1.5875]`, B-faster `101/101`; and
`384x1216x4096` `1.5077x`, CI `[1.4862,1.5163]`, B-faster `101/101`.
Existing `m >= 512, n = 1152, k = 3072` guards stayed neutral. Lower `m = 384`
width guards were neutral/noisy, so the source rule does not broaden those:
`384x512x4096` `0.9960x`, `384x768x4096` `1.0119x` with neutral holdout, and
`384x1024x4096` `0.9966x`. Correctness coverage added `384x1152x3072` and
`384x1216x4096`.

Follow-up accepted: exact `n = 1216` one-shot medium shapes now use the packed
path from `k >= 3072`. The `k = 2048` neighbor was neutral/noisy, but the
`k = 3072` band was a large and consistent win against the previous AMX
strided-B route: `384x1216x3072` `1.2450x`, CI `[1.2492,1.2679]`, B-faster
`101/101`; `512x1216x3072` `1.3160x`, B-faster `99/101`;
`768x1216x3072` `1.4334x`, B-faster `99/101`; and `1024x1216x3072`
`1.5064x`, B-faster `101/101`. `k = 4096` was already selected by the existing
high-`K` gate and stayed neutral. Correctness coverage added `384x1216x3072`
and `1024x1216x3072`.

Follow-up rejected: a chunked one-shot packed-B path for exact `m = 384` was
not a clean replacement for the current full packed-B scratch path. The
`NC=512` candidate in `/private/tmp/cob_m384_chunked_oneshot_exp` passed
correctness but was mostly neutral and regressed `384x4096x4096` (`0.9934x`,
holdout `0.9931x`). `NC=1024` improved `384x2048x4096` (`1.0178x`) and was
weakly positive at `384x4096x2048`, but still regressed `384x4096x4096`
(`0.9857x`) and did not help the `1152/1216 x 3072` band. Do not revisit
chunked one-shot B packing for `m = 384` without a smoother rule or a different
packing schedule.

Follow-up accepted: public packed-B `n = 1024, k >= 3072` now stays on AMX
instead of the current SME packed-B kernel. The `k = 3072` packed-B A/B was
strong across the tested heights: `512x1024x3072` `1.3090x`,
`768x1024x3072` `1.3784x`, and `1024x1024x3072` `1.4153x`, all with
B-faster `101/101` or `99/101` and agreeing holdouts. The `k = 2048` neighbor
was mixed (`512/768` negative, `1024` positive), so the source gate does not
broaden below `3072`; `k = 4096` was already AMX and stayed neutral.
Correctness coverage added `512x1024x3072` and `1024x1024x3072`.

Follow-up rejected: routing `64x1408x2048` and `64x1472x2048` back to the old
AMX path did not close their remaining small gaps. The AMX fallback candidate
in `/private/tmp/cob_k2048_1408_amx_exp` passed correctness but paired A/B
regressed `64x1408x2048` to median `0.8164x`, bootstrap95 `[0.8014,0.8381]`,
and `64x1472x2048` to `0.7508x`, `[0.7363,0.7548]`. Keep those shapes on the
SME skinny route.

Follow-up rejected: revisiting `COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE=64`
showed real high-`K` wins but no smooth enough boundary for a runtime gate.
Positive lines included `64x2112x7168` `1.0102x`, `64x2112x12288`
`1.0079x`, `64x2240x7168` `1.0157x`, `64x2240x8192` `1.0151x`,
`64x3904x8192` `1.0330x`, and `64x4032x7168/8192/12288` at
`1.0383x`/`1.0342x`/`1.0334x`. Neighboring checks were neutral or negative:
`64x1984x7168` `0.9912x`, `64x2176x7168` `0.9747x`, `64x2176x8192`
`0.9786x`, `64x3712x12288` `0.9989x`, and `64x3328x12288` `0.9998x`.
Keep the default distance `32`; do not add exact-width distance gates without
counter evidence or a smoother rule.

### 2026-05-05 local-uncommitted: m64 large-K SME B-prefetch accepted

A temp source added two-cacheline software prefetching for the SME streaming-B
load path and tested it with the paired harness. Broad prefetching was rejected:
medium SME direct shapes were neutral/noisy, `m = 96/128` reuse shapes regressed
or stayed noisy, and wide `n >= 7168` B-reuse mostly regressed. The useful
region was the `m = 64`, large-`K` skinny path.

Accepted source behavior is narrowly gated:

- Skinny direct `m = 64`, `k >= 2048`, `1088 <= n <= 4096`, and `n % 512 != 0`
  uses a prefetched SME strided-B kernel. At `k = 1536`, the prefetched
  subroute starts from `n >= 1408`.
- The existing exact `m = 64, n = 4096, k >= 7168` B-reuse path uses a
  prefetched tuple pack helper.
- Multiples of 512 such as `1536`, `2048`, `2560`, and `3584` stay on the old
  unprefetched path.

Key paired validation against the pre-change source:

- `64x1088x7168` repeat-61 median `1.2233x`, bootstrap95
  `[1.2154,1.2490]`, B-faster `61/61`; holdout median `1.2233x`.
- `64x1856x7168` repeat-31 median `1.1748x`, bootstrap95
  `[1.1669,1.1745]`, B-faster `31/31`.
- `64x3712x7168` repeat-101 median `1.1172x`, bootstrap95
  `[1.1138,1.1210]`, B-faster `101/101`; holdout median `1.1172x`.
- `64x4032x8192` repeat-101 median `1.2469x`, bootstrap95
  `[1.2361,1.2596]`, B-faster `101/101`; holdout median `1.2475x`.
- `64x4096x7168` repeat-101 median `1.0695x`, bootstrap95
  `[1.0584,1.0728]`, B-faster `98/101`; holdout median `1.0718x`.
- `64x4096x12288` repeat-101 median `1.0609x`, bootstrap95
  `[1.0410,1.0548]`, B-faster `89/101`; holdout median `1.0409x`.

Guard validation stayed neutral: `64x1536x7168`, `64x2048x7168`,
`64x2560x7168`, `64x3584x7168`, `64x4096x1536`, `64x7168x2048`,
`832x1472x1152`, `96x4096x1024`, and `128x8192x1024` did not show a
commit-worthy regression.

### 2026-05-05 local-uncommitted: n1152 SME-before-strided rejected

A temp source moved the medium SME direct route ahead of AMX strided-`B` for
`n = 1152` candidates. The first paired runs looked promising for some
`k = 832/960` shapes, but validation against the committed baseline did not
hold. Repeat-81 with `COB_AB_ITERS=5` showed `832x1152x832` neutral
(`0.9991x`), `832x1152x960` neutral/negative (`0.9928x`), `832x1152x1152`
regressive (`0.8835x`), and `1152x1152x832` regressive (`0.9945x`, negative
CI). Only isolated `1024x1152x832` and `1088x1152x832` stayed positive, which
is not a smooth enough boundary to ship. Keep AMX strided-`B` priority for
`n = 1152`.

### 2026-05-05 local-uncommitted: mperf counter setup blocked

`mperf` was found via <https://lambdafoo.com/posts/2026-03-25-mperf-hardware-counters-macos.html>
and cloned from <https://github.com/tmcgilchrist/mperf> into `/private/tmp`.
It built cleanly with `make`, producing `mperf-stat`. Running without root
reported `Root privileges required`; running with `sudo` failed because this
session cannot provide an interactive password (`sudo: a terminal is required
to read the password`). Counter profiling remains a local setup task rather
than a blocker for the current source changes.

### 2026-05-05 local-uncommitted: long-k512 tuple helper rejected

A temp source copy forced the long-`n`, `k = 512` reuse path to use the tuple
`svld1_x4` / `svst1` helper instead of the non-tuple helper. This rechecked
whether wider four-Z load/store adoption had become useful after the later
skinny route changes.

Paired validation rejected the change. It regressed `64x4096x512` median
`0.9442x` and `64x32768x512` `0.9286x`. `64x8192x512`, `96x4096x512`,
`96x8192x512`, `128x4096x512`, and `128x8192x512` were neutral/noisy with weak
or reversed holdout. Keep the current split: non-tuple for long-`n`, `k = 512`,
tuple helper for the other B-reuse paths.

### 2026-05-05 local-uncommitted: public-source scan refresh

A fresh public scan did not identify a new licensed open-source single-thread
FP32 row-major CPU GEMM blocker for the current Apple Silicon target. The tested
licensed/open-source baseline set remains BLIS, OpenBLAS, BLASFEO, Eigen, Rust
`matrixmultiply`, `coral-aarch64`, LIBXSMM, `tract-linalg`, and KleidiAI's
comparable public one-shot path. The notable scan hits were:

- <https://github.com/mmperf/mmperf>, a single-core matmul benchmark collection,
  useful for future comparison ideas but not a directly tested Apple AMX/SME
  library in this pass.
- <https://pypi.org/project/sapphire-compute/>, which advertises 1.56 TFLOPS
  SGEMM and MIT metadata on PyPI, but the release has no source distribution and
  describes its SGEMM stack as `cblas -> AMX`, so it remains a wrapper/packaging
  item rather than a separate open-source kernel blocker.
- <https://github.com/usstq/mm_amx>, an AMX matmul repo focused on Intel
  AMX/BF16 rather than this Apple FP32 row-major target.
- <https://github.com/corsix/amx>, an MIT Apple AMX instruction reference, not a
  competing GEMM implementation.

README wording was tightened to say COB beats the tested licensed/open-source
baselines on the routed shape ranges in the repo's narrow benchmarked scope,
while Accelerate, source-available, and non-inspectable projects remain separate
calibration/comparison targets.

### 2026-05-06 faf7106+local: public packed-B n=768 k=2048/3072 AMX fallback

Route-aware packed-B sweeps left one medium-width high-`K` band on the current
SME packed-B kernel. A paired packed-B A/B probe forced exact `n = 768,
k = 3072` through AMX while leaving the neighboring `k = 2048` and `k = 4096`
guards on their existing paths.

The target shapes were positive across the routed `m = 512..1024` band:
`512x768x3072` median `1.0833x`, bootstrap95 `[1.1139,1.1601]`, B-faster
`95/101`, sign-p `1.07e-21`, holdout median `1.0426x`; `768x768x3072` median
`1.0453x`, bootstrap95 `[1.0414,1.0549]`, B-faster `95/101`, sign-p
`1.07e-21`, holdout median `1.0517x`; and `1024x768x3072` median `1.0583x`,
bootstrap95 `[1.0548,1.0668]`, B-faster `100/101`, sign-p `8.05e-29`,
holdout median `1.0634x`.

The first guard run showed a suspicious `768x768x2048` regression even though
the route was unchanged. A focused repeat-181 rerun reduced it to neutral:
median `1.0023x`, bootstrap95 `[0.9924,1.0023]`, B-faster `96/181`, sign-p
`0.457`, holdout median `1.0008x`. `512x768x2048` was also neutral at median
`1.0023x`, bootstrap95 `[0.9935,1.0040]`, sign-p `0.842`. The `k = 4096`
guards were already AMX in the baseline; the paired ratios were mixed/noisy and
not used to broaden the rule.

The first accepted patch kept the rule exact: public packed-B `n = 768,
k = 3072` used AMX, while `k = 2048` remained on SME packed-B and `k >= 4096`
remained on the existing AMX high-`K` path. Correctness coverage added
`512x768x3072` and `1024x768x3072`.

A follow-up exact `k = 2048` probe initially looked positive but weak, so it was
rerun with repeat-201 focused on the target shapes. The rerun validated AMX for
the full `m = 512..1024` band: `512x768x2048` median `1.0372x`, bootstrap95
`[1.0201,1.0324]`, B-faster `131/201`, sign-p `2.02e-05`, holdout median
`1.0417x`; `768x768x2048` median `1.0262x`, bootstrap95 `[1.0163,1.0322]`,
B-faster `128/201`, sign-p `0.000128`, holdout median `1.0197x`; and
`1024x768x2048` median `1.0300x`, bootstrap95 `[1.0285,1.0480]`, B-faster
`146/201`, sign-p `1.05e-10`, holdout median `1.0310x`. The accepted rule now
uses AMX for public packed-B `n = 768` at exact `k = 2048` and `k = 3072`, with
`k = 1536` still treated as a neutral/noisy guard and `k >= 4096` covered by
the existing high-`K` AMX path. Correctness coverage adds `512x768x2048`,
`768x768x2048`, and `1024x768x2048`.

### 2026-05-06 5c852b2+local: public packed-B n=1152 k=1536 AMX fallback

The route-aware medium grid still showed `n = 1152, k = 1536` on the SME
packed-B path, while `n = 1152, k >= 2048` was already routed to AMX. A broad
threshold probe down to `k >= 1536` improved the target band but produced a
tiny same-route guard dip at `1024x1152x2048`. The candidate was narrowed to an
exact `k = 1536` sibling and ordered after the existing `k >= 2048` condition,
so the already-routed high-`K` branch short-circuits before the new check.

The final focused repeat-201 packed-B A/B validated the exact sibling:
`512x1152x1536` median `1.0372x`, bootstrap95 `[1.0537,1.0790]`, B-faster
`161/201`, sign-p `2.11e-18`, holdout median `1.0112x`; `768x1152x1536`
median `1.0341x`, bootstrap95 `[1.0397,1.0599]`, B-faster `147/201`, sign-p
`3.86e-11`, holdout median `1.0142x`; and `1024x1152x1536` median `1.0352x`,
bootstrap95 `[1.0452,1.0719]`, B-faster `151/201`, sign-p `5.57e-13`,
holdout median `1.0348x`. The reordered `1024x1152x2048` guard rerun was
neutral at median `1.0012x`, bootstrap95 `[0.9954,1.0086]`, B-faster `74/141`,
sign-p `0.614`, holdout median `1.0016x`.

The accepted rule keeps public packed-B `n = 1152` on AMX at exact `k = 1536`
and at the existing `k >= 2048` band. Correctness coverage adds
`512x1152x1536`, `768x1152x1536`, and `1024x1152x1536`.

### 2026-05-06 7916d8e+local: public packed-B n=512 k=3072 high-m AMX fallback

After the `n = 1152` update, the medium route grid showed `1024x512x3072`
still on `packed_sme` and trailing Accelerate, while `512x512x3072` and
`768x512x3072` were not gaps. A candidate excluded the SME packed-B route only
for `m >= 1024, n = 512, k = 3072`.

Paired packed-B A/B validated the high-`m` exact gate: `1024x512x3072` median
`1.0323x`, bootstrap95 `[1.0286,1.0558]`, B-faster `130/141`, sign-p
`5.75e-27`, holdout median `1.0288x`; and `1280x512x3072` median `1.0347x`,
bootstrap95 `[1.0391,1.0580]`, B-faster `136/141`, sign-p `3.22e-34`,
holdout median `1.0390x`. The lower-`m` neighbors stayed neutral/noisy:
`512x512x3072` median `1.0021x`, sign-p `0.238`, holdout median `1.0000x`;
and `768x512x3072` median `0.9979x`, sign-p `0.4`, holdout median `1.0000x`.
The `k = 2048` guards were also neutral, so the rule is limited to exact
`k = 3072` and `m >= 1024`. Correctness coverage adds `1024x512x3072` and
`1280x512x3072`.

### 2026-05-06 a911501+local: lower packed-B AMX large-block threshold for high-K n>=768

After the SME-vs-AMX dispatch gaps narrowed, the remaining public packed-B
medium losses were mostly already on AMX. A structural probe lowered the AMX
packed-B large-block schedule from `n >= 1152` to `n >= 768`. The broad version
was strongly positive at `k >= 3072`, but it touched `n = 768, k = 2048` and
produced a behavior-identical/noisy guard mix, including one repeat with a
`1024x768x2048` regression. The accepted rule keeps the old `n >= 1152` path
and only uses the lower `n >= 768` threshold when `k >= 3072`.

The narrowed paired packed-B A/B remained strongly positive on the target
routes: `512x768x3072` median `1.1135x`, bootstrap95 `[1.1038,1.1191]`,
B-faster `100/101`, sign-p `8.05e-29`, holdout median `1.1161x`;
`768x768x3072` median `1.0428x`, bootstrap95 `[1.0167,1.0395]`, B-faster
`81/101`, sign-p `6.93e-10`, holdout median `1.0517x`; `1024x768x3072`
median `1.0539x`, bootstrap95 `[1.0544,1.0687]`, B-faster `98/101`, sign-p
`1.36e-25`, holdout median `1.0615x`; `512x768x4096` median `1.1061x`,
bootstrap95 `[1.0758,1.1017]`, B-faster `92/101`, sign-p `1.82e-18`, holdout
median `1.1231x`; `768x768x4096` median `1.1153x`, bootstrap95
`[1.0493,1.0934]`, B-faster `72/101`, sign-p `2.24e-05`, holdout median
`1.1226x`; `1024x768x4096` median `1.1089x`, bootstrap95 `[1.0761,1.1026]`,
B-faster `91/101`, sign-p `1.7e-17`, holdout median `1.1081x`;
`1024x1024x3072` median `1.1714x`, bootstrap95 `[1.1386,1.1745]`, B-faster
`98/101`, sign-p `1.36e-25`, holdout median `1.1709x`; and
`1024x1024x4096` median `1.1006x`, bootstrap95 `[1.0967,1.1068]`, B-faster
`61/61`, sign-p `8.67e-19`, holdout median `1.1000x`.

Correctness coverage adds `512x768x4096`, `768x768x4096`, and
`1024x1024x4096`.

### 2026-05-06 226827d+local: lower one-shot AMX large-block threshold for high-K n>=768

The public packed-B large-block win changed the one-shot tradeoff too. The
one-shot AMX path still used the old `n >= 1152` large-block threshold, which
left `n = 768/1024, k >= 3072` either on strided-B or on the old one-panel
packed-A schedule. A paired one-shot A/B applied the same `n >= 768,
k >= 3072` large-block rule to the one-shot path.

The target rows were consistently positive: `512x768x3072` median `1.0441x`,
bootstrap95 `[1.0314,1.0568]`, B-faster `73/101`, sign-p `8.64e-06`, holdout
median `1.0407x`; `768x768x3072` median `1.0488x`, bootstrap95
`[1.0403,1.0629]`, B-faster `87/101`, sign-p `4.8e-14`, holdout median
`1.0526x`; `1024x768x3072` median `1.1261x`, bootstrap95 `[1.1392,1.1791]`,
B-faster `101/101`, sign-p `7.89e-31`, holdout median `1.1318x`;
`512x768x4096` median `1.0282x`, bootstrap95 `[1.0175,1.0432]`, B-faster
`66/101`, sign-p `0.00265`, holdout median `1.0337x`; `768x768x4096` median
`1.0327x`, bootstrap95 `[1.0231,1.0538]`, B-faster `80/101`, sign-p
`2.73e-09`, holdout median `1.0294x`; `1024x768x4096` median `1.0353x`,
bootstrap95 `[1.0249,1.0377]`, B-faster `82/101`, sign-p `1.66e-10`, holdout
median `1.0395x`; `1024x1024x3072` median `1.0985x`, bootstrap95
`[1.0918,1.1317]`, B-faster `101/101`, sign-p `7.89e-31`, holdout median
`1.0780x`; and `1024x1024x4096` median `1.0976x`, bootstrap95
`[1.0812,1.1000]`, B-faster `98/101`, sign-p `1.36e-25`, holdout median
`1.1061x`.

The `n = 768, k = 2048` rows were behavior-identical/noisy and did not justify
lowering the threshold below `k = 3072`. No new correctness shapes were needed;
the affected one-shot and packed-B shapes are already covered by the 98-shape
suite.

### 2026-05-06 0a86b71+local: n=512 high-m AMX large-block for k>=4096

A focused repeated benchmark showed `1024x512x4096` remained a coherent medium
gap after the `n >= 768` large-block changes. A narrow large-block probe added
only `n = 512, m >= 1024, k >= 4096` to both the public packed-B AMX path and
the one-shot packed path.

Public packed-B A/B validated the high-row target: `1024x512x4096` median
`1.0228x`, bootstrap95 `[1.0176,1.0405]`, B-faster `86/101`, sign-p
`2.83e-13`, holdout median `1.0224x`; and `1280x512x4096` median `1.0178x`,
bootstrap95 `[1.0147,1.0204]`, B-faster `56/61`, sign-p `5.65e-12`, holdout
median `1.0191x`. Lower rows were not admitted: `768x512x4096` regressed in
the behavior-identical guard, and `512x512x4096` was only a noisy small
positive.

One-shot A/B was stronger on the same high-row target: `1024x512x4096` median
`1.0500x`, bootstrap95 `[1.0276,1.0508]`, B-faster `95/101`, sign-p
`1.07e-21`, holdout median `1.0500x`; and `1280x512x4096` median `1.0469x`,
bootstrap95 `[1.0245,1.0573]`, B-faster `88/101`, sign-p `7.52e-15`, holdout
median `1.0498x`. The `1024/1280 x 512 x 3072` guards were neutral/noisy, so
the rule starts only at `k >= 4096`. Correctness coverage adds
`1024x512x4096` and `1280x512x4096`.

### 2026-05-06 5a6cc0a+local: public packed-B n=1024 k=2048 AMX fallback

The public packed-B `n = 1024` rule previously started at `k >= 3072`, leaving
`k = 2048` on the SME packed-B path. A focused exact-sibling probe kept the old
`k >= 3072` condition first and added only exact `k = 2048`.

The repeat-201 packed-B rerun was positive on the target band:
`512x1024x2048` median `1.0199x`, bootstrap95 `[1.0119,1.0283]`, B-faster
`119/201`, sign-p `0.0109`, holdout median `1.0146x`; `768x1024x2048` median
`1.0253x`, bootstrap95 `[1.0245,1.0393]`, B-faster `151/201`, sign-p
`5.57e-13`, holdout median `1.0220x`; and `1024x1024x2048` median `1.0436x`,
bootstrap95 `[1.0705,1.1142]`, B-faster `185/201`, sign-p `1.25e-37`,
holdout median `1.0727x`. The `512` row had a weaker holdout sign count
(`59/101`, sign-p `0.111`), but its median and bootstrap holdout stayed
positive and no target row reversed.

The `512x1024x1536` guard was behavior-identical/noisy, and the
`1024x1024x3072` guard stayed neutral at median `0.9986x`, bootstrap95
`[0.9974,1.0004]`, sign-p `0.467`, holdout median `0.9974x`. The accepted rule
uses AMX for public packed-B `n = 1024` at exact `k = 2048` and at the existing
`k >= 3072` band. Correctness coverage adds `512x1024x2048`,
`768x1024x2048`, and `1024x1024x2048`.

### 2026-05-06 743ecac+local: claim audit bundle tool

After the low-`k` exact packed-B probes stopped producing clean wins, broad
threshold work was paused in favor of audit hygiene. `tools/claim_audit.sh`
now writes a timestamped bundle containing `context.txt`, route-aware benchmark
CSVs, one-shot and packed-B gap reports, and `summary.md`. The default suites
cover square, medium, and sparse skinny route families, with environment
variables for cold reruns of narrower families.

The smoke run used `COB_AUDIT_SUITES=square COB_AUDIT_SQUARE_SIZES="64"
COB_AUDIT_REPEATS=1 COB_AUDIT_OUT_DIR=/tmp/cob-claim-audit-smoke
sh tools/claim_audit.sh`, and produced the expected context, CSV, gap, and
summary files. `docs/claims.md` now points the audit recipe at this wrapper so
future full-claim checks do not rely on ad hoc terminal output.

Follow-up tooling tweak: the audit bundle also emits one-shot-vs-best gap
reports that include COB packed-B as an internal baseline. This makes pack/setup
overhead visible separately from external-baseline misses.

Follow-up sanity tweak: `tools/bench_sanity_report.py` flags benchmark rows
where median throughput falls far below the best sample, and
`tools/claim_audit.sh` now writes per-suite `.sanity.csv` files plus summary
sections. This gives noisy/collapsed audit runs an explicit warning before
their gap reports are used for route tuning.

### 2026-05-06 8ccd77b+local: one-shot n=1216 k=2048 packed path for m>=768

The medium audit's internal one-shot-vs-best report highlighted
`1024x1216x2048`: one-shot stayed on AMX strided-B while COB packed-B was much
faster after `B` was already packed. A broad exact `n = 1216, k = 2048` packed
path rerun showed strong wins at `m = 768/1024`, but only weak/noisy behavior at
`m = 512`. The accepted candidate therefore keeps `m = 512` on the existing
strided-B path and uses the packed path only when `m >= 768`.

The focused repeat-201 rerun validated the narrowed gate:
`768x1216x2048` median `1.0252x`, bootstrap95 `[1.0535,1.0992]`, B-faster
`139/201`, sign-p `5.77e-08`, holdout median `1.0313x`; and
`1024x1216x2048` median `1.0936x`, bootstrap95 `[1.1083,1.1563]`, B-faster
`189/201`, sign-p `4.31e-42`, holdout median `1.0971x`. The excluded
`512x1216x2048` row stayed neutral/noisy at median `0.9986x`, bootstrap95
`[0.9750,1.0181]`, sign-p `0.778`. The `k = 1536` and `k = 3072` guards were
behavior-identical/noisy or neutral, so the rule is exact to `k = 2048` for
`m >= 768`; the existing `k >= 3072` path remains unchanged. Correctness
coverage adds `768x1216x2048` and `1024x1216x2048`.

## Current Conclusion

COB is very competitive in its exact current scope. The qualified claim now is:
fastest among tested licensed/open-source baselines on the routed shape ranges
for single-thread FP32 row-major GEMM. MpGEMM remains a source-available
calibration target with unclear licensing and still wins some one-shot `m = 64`
shapes. Accelerate also remains ahead on a few small/medium cases, especially
where one-shot pack overhead dominates.

The current working loop is: route-aware grid sweep, gap ranking, paired A/B
candidate probe, cold/holdout validation, then either a narrow source change or
a rejection note. That loop has produced the public packed-AB repeated-use path,
skinny SME B-reuse generalization for `m = 64/96/128`, the `m = 64, k = 512`
threshold and mid-width extensions, m64 high-K SME N-chunking for
`3584 <= n < 4096`, B-pack prefetching, packed-B large-square and `m = 384`
blocking, small-A packed-B B-panel traversal, medium SME direct routes in the
`n = 1280..1472` band, high-K AMX packed-path fallbacks, and several public
packed-B AMX fallbacks for shapes where SME lost.

The latest C-level attempts to close the remaining m64 large-K calibration gap
did not hold up: B-panel-first SME direct traversal, prefetch locality changes,
epilogue branch hoisting, broad compiler unrolling, and `-mcpu=native` were all
neutral, noisy, or regressive. The remaining gap is therefore still best treated
as an SME kernel scheduling problem, likely requiring a dedicated fixed-shape
kernel or assembly rather than more dispatch gates. Current correctness
coverage is 371 GEMM shapes.

Historical post-`5e6da0a` rejected/probed follow-ups:

- Successful tuple-B load/store follow-up: ACLE tuple `svld1_x4` / `svst1` was
  tested in `/private/tmp/cob_tuple_b_exp` and passed all 37 tests. The full
  tuple experiment improved several blockers, with repeat-25 medians of
  `64x24576x1536` 975.14 GF/s, `64x7168x2048` 1029.62 GF/s,
  `64x7168x16384` 980.65 GF/s, `64x4096x7168` 982.00 GF/s, and
  `64x2112x7168` 1214.90 GF/s, but regressed `64x32768x512` to
  767.23 GF/s. The main selective patch keeps tuple direct strided and tuple
  `m = 64` B-reuse except long-`n`, `k = 512`, which stays on the old helper.
  Selective validation passed `make test` across 37 shapes and
  `git diff --check`. Focused repeat-25 selective medians were
  `64x24576x1536` 971.42 GF/s, `64x7168x2048` 1017.35 GF/s,
  `64x7168x16384` 988.00 GF/s, `64x8192x1024` 999.76 GF/s,
  `64x4096x7168` 989.23 GF/s, and `64x2112x7168` 1210.35 GF/s;
  `64x32768x512` was noisy, but A/B repeat-31 measured default 896.28 GF/s
  versus selective 916.16 GF/s.
- Direct-SME-wide worktree `/private/tmp/cob_sme_direct_wide_exp` disabled wide
  B-reuse and let direct strided SME cover `n >= 7168, k >= 1024`. It passed
  all 37 GEMM tests, but repeat-25 one-shot results were much worse on wide
  shapes: `64x24576x1536` median 437.31 GF/s, `64x7168x2048` 637.61 GF/s,
  `64x7168x16384` 610.06 GF/s, and `64x8192x1024` 564.24 GF/s. Reject.
- Compile-time `NC` probes for `m = 64` B-reuse did not produce a commit-worthy
  default. `NC = 256` improved some medians in one run but hurt or was noisy
  for `64x32768x512`; `NC = 512` looked modestly better in repeat-25, including
  `64x24576x1536` median 950.59 GF/s, `64x7168x2048` 997.37 GF/s, and
  `64x8192x1024` 1024.56 GF/s, but repeat-31 was inconclusive/noisy.
  `NC = 768` was not better. No source commit yet.
- `703de33` (`Tighten m64 SME reuse chunks`) revisited the `m = 64` B-reuse
  chunk size after tuple B loads/stores landed in `10849f3` and were documented
  in `3cb1ee2`. Retesting `COB_SGEMM_M64_SME_REUSE_NC=512` showed it had become
  a consistent win over the previous `1024` chunk size for the tuple-era
  B-reuse path. Before the source commit, `make test` passed all 37 shapes,
  `make all` rebuilt, and `git diff --check` passed. Focused repeat-25 after
  the source change measured `64x4096x7168` 1029.05 GF/s, `64x7168x16384`
  1029.47 GF/s, `64x24576x1536` 936.58 GF/s, `64x7168x2048` 1013.51 GF/s,
  `64x32768x512` 935.72 GF/s, `64x2112x7168` 1235.82 GF/s, and
  `64x8192x1024` 1053.72 GF/s. This improves the tuple-era B-reuse route, but
  MpGEMM still remains ahead on some shapes, especially `64x4096x7168`,
  `64x7168x16384`, and `64x8192x1024`.
- `47e5e7a` (`Tune long wide m64 SME chunks`) followed up after `703de33` made
  `NC = 512` the default. A narrow long-wide gate was tested and committed:
  use `COB_SGEMM_M64_SME_LONG_WIDE_NC=256` only when `m = 64`, wide B-reuse
  applies, `n >= 24576`, and `k == 1536`; otherwise keep `NC = 512`.
  Validation before the source commit passed `make test` across 37 shapes,
  `make all`, and `git diff --check`. Focused repeat-25 after the source change
  measured `64x24576x1536` median 1020.02 GF/s, `64x32768x512` median
  937.77 GF/s, `64x7168x16384` median 1025.75 GF/s, and `64x4096x7168`
  median 1029.62 GF/s. This targets the `64x24576x1536` MpGEMM comparison:
  MpGEMM's `O0` timing driver measured best 1042.92 GF/s and median
  853.23 GF/s in a noisy but usable separate-process run, while COB now
  measured best 1031.78 GF/s and median 1020.02 GF/s in the focused bench.
- `COB_SGEMM_M64_SME_REUSE_NC=384` was rejected after a broad paired run looked
  only weakly positive and focused validation did not hold up. Broad paired
  ratios were `64x4096x7168` median 1.0046x, mean-log 1.0080x, CI
  [0.9987,1.0223], sign-p 0.349; `64x7168x2048` median 1.0016x, CI
  [0.9960,1.0127], sign-p 0.755; `64x7168x16384` median 1.0034x, CI
  [0.9958,1.0089], sign-p 0.0596; and `64x8192x1024` median 1.0049x, CI
  [1.0006,1.0121], sign-p 0.0784. Focused validation rejected it:
  `64x7168x16384` repeat-81 median 1.0002x, mean-log 1.0002x, CI
  [0.9911,1.0077], sign-p 1; `64x8192x1024` repeat-51 median 0.9990x,
  mean-log 1.0011x, CI [0.9958,1.0068], sign-p 1. No source behavior change.
- `KC` probes also stayed uncommitted. `NC512 + KC1024` and targeted
  `k == 2048` / `k == 1024` variants were not clean enough to commit; the
  `k == 1024`-only repeat-31 gave `64x8192x1024` median 970.83 GF/s and
  `64x7168x1024` 952.86 GF/s, so reject.
- The 64x16 SME strided kernel temp worktree
  `/private/tmp/cob_sme_64x16_exp` passed all 37 tests but was much slower:
  `64x24576x1536` median 564.27 GF/s, `64x7168x2048` 708.54 GF/s,
  `64x4096x7168` 667.99 GF/s, and `64x2112x7168` 887.26 GF/s. Reject.
- Post-`030448f` direct-SME-wide route in
  `/private/tmp/cob_sme_direct_wide_exp` disabled wide B-reuse and let direct
  strided SME cover `n >= 7168, k >= 1024`. It was correct, but much slower on
  wide `m = 64` shapes, so reject.
- Post-`030448f` tuple chunk sweeps were rejected. Global `NC = 256` helped
  `64x24576x1536` but hurt broader targets, and `KC = 640`, `896`, and `1024`
  variants were worse than the current `KC = 768`.
- The `n = 2112` B-reuse gate was correct but regressed to about 995 GF/s
  median. Direct tuple SME remains better, so reject.
- The 64x16 SME strided kernel was correct but much slower. Reject.
- Panel-immediate reuse and fused-four helper variants were correct but slower.
  Reject.
- Tuple pack-only and tuple reuse-only split experiments were correct but worse
  or noisy. Reject.
- One-shot SME packed `n = 1024` plus the medium gate, low gate, and
  packed-384 experiments were correct but did not close the `768` / `1024` /
  `1280` gaps. Reject.
- Packed-B micro-experiments were rejected. `x2` loads and K-loop unroll were
  correct but slower; the first `vg4` column-store failed correctness, and the
  corrected `vg4` row-group store passed tests but lost packed-B median at
  `768`, `1024`, and `1280`.
- MpGEMM checkout `/private/tmp/mpgemm_latest` was refreshed and was already up
  to date at `8d83011`.

Historical note: this older conclusion treated MpGEMM as an in-scope blocker and
predates the newer paired-harness methodology, skinny SME generalization, and
medium-width SME direct routes. Keep it as context for the old experiments, but
use the `Current Conclusion` section above and `docs/claims.md` for the current
claim boundary.
