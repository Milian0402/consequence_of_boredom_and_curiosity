"""Instrument the pinned Linux experiment sources in a disposable snapshot."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
mode = sys.argv[2]
smoke = len(sys.argv) == 4 and sys.argv[3] == 'smoke'
src = root / 'src'


def one(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source anchor for {old[:90]!r}')
    return text.replace(old, new)


if smoke:
    path = src / 'cob_hk15.inc'
    hk = path.read_text().replace('768', '128').replace('1280', '224')
    assert '960*area' in hk and 'ldc, 960)' in hk
    path.write_text(hk.replace('960*area', '144*area').replace('ldc, 960)', 'ldc, 144)'))
    path = src / 'cob_fused_strassen.inc'
    text = one(path.read_text(), '#define COB_FUSED_STRASSEN_MIN_HARMONIC_DIM 1024',
               '#define COB_FUSED_STRASSEN_MIN_HARMONIC_DIM 192')
    path.write_text(text)
if mode == 'control':
    sys.exit(0)
if mode != 'profile':
    raise ValueError(mode)


def timed(category, code):
    return f'HK_TIMED({category}, {code});'


def calls(text, function, category, count=1):
    pattern = rf'(?m)^(\s*)({function}\([^;]*?\));'
    result, found = re.subn(pattern, lambda m: m[1] + timed(category, m[2]), text)
    if found != count:
        raise ValueError(f'Expected {count} calls to {function}, found {found}')
    return result


# Time each code-generated parent loop once, leaving all inner element loops unchanged.
schedule = root / 'tools/world-rank/finite/hk233.inc'
groups = re.split(r'(?=    /\* Product \d+ of 15\. \*/)', schedule.read_text())
if len(groups) != 16:
    raise ValueError('Unexpected number of certified products')
for i, group in enumerate(groups[1:], start=1):
    header, body = group.split('\n', 1)
    before, leaf, after = re.split(r'(?m)^(    cob_sgemm_rowmajor\([^;\n]+;)$', body)
    fragments = [header]
    if before.strip():
        fragments += [timed('HK_FORMS', '{\n' + before + '}')]
    fragments += [timed('HK_CHILD', leaf.strip()[:-1])]
    if after.strip():
        fragments += [timed('HK_OUTPUT', '{\n' + after + '}')]
    groups[i] = '\n'.join(fragments) + '\n'
schedule.write_text(''.join(groups))

path = src / 'cob_fused_strassen.inc'
text = path.read_text()
text = calls(text, 'cob_fused_pack_b', 'FUSED_PACK_B')
text = calls(text, 'cob_fused_pack_a', 'FUSED_PACK_A')
for function in ['cob_blocked_apply_panel', 'cob_final_panel']:
    text = calls(text, function, '(kb <= 64 ? FUSED_APPLY_SHORT : FUSED_APPLY_LONG)')
text = calls(text, 'cob_strassen_merge_m1', 'FUSED_MERGE')
text = calls(text, 'cob_strassen_merge_m5', 'FUSED_MERGE')
text = calls(text, 'cob_strassen_merge_product', 'FUSED_MERGE', 2)
anchor = '    if (!work) return 0;'
text = one(text, anchor, anchor + '\n    const uint64_t fused_start = hk_phase_clock();')
end = '    cob_fused_finish_edges(m, n, k, a, lda, b, ldb, c, ldc);\n    return 1;'
text = one(text, end, '    ' + timed('FUSED_EDGES', end.split(';')[0].strip()) +
           '\n    hk_phase_end(FUSED_BODY, fused_start);\n    return 1;')
path.write_text(text)

path = src / 'cob_blocked_leaf.inc'
text, body = path.read_text().split('static int cob_sgemm_blocked_impl(', 1)
body = calls(body, 'cob_blocked_pack_b', 'BASE_PACK_B')
# One optional first-tile fused-packing branch is disabled by the frozen build flags.
body = calls(body, 'cob_blocked_pack_a', 'BASE_PACK_A', 2)
body = calls(body, 'cob_blocked_apply_panel', '(kb <= 64 ? BASE_APPLY_SHORT : BASE_APPLY_LONG)')
anchor = '    if (!pack_b) return 0;'
body = one(body, anchor, anchor + '\n    const uint64_t base_start = hk_phase_clock();')
body = one(body, '    free(pack_b);\n    return 1;', '    free(pack_b);\n    hk_phase_end(BASE_BODY, base_start);\n    return 1;')
path.write_text(text + 'static int cob_sgemm_blocked_impl(' + body)
