#include "amx_compute.h"
#include <stddef.h>

#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#if defined(_tile_dpbf16ps) && defined(_tile_loadd) && defined(_tile_stored)

struct tile_config {
    uint8_t palette, start_row;
    uint8_t reserved0[14];
    uint16_t column_bytes[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};
_Static_assert(sizeof(struct tile_config) == 64 &&
               offsetof(struct tile_config, column_bytes) == 16 &&
               offsetof(struct tile_config, rows) == 48, "AMX tile configuration layout");
static _Alignas(64) const struct tile_config config = {
    .palette = 1,
    .column_bytes = {64,64,64,64,64,64,64,64},
    .rows = {16,16,16,16,16,16,16,16}
};

int cob_amx_six_compiled(void) { return 1; }

#if defined(COB_AMX_TRACE) && COB_AMX_TRACE
static _Thread_local uint64_t observed_tiles, observed_kblocks, observed_dpbf16;
uint64_t cob_amx_six_trace(int index)
{
    switch (index) {
    case 0: return observed_tiles;
    case 1: return observed_kblocks;
    case 2: return observed_dpbf16;
    default: return 0;
    }
}
#else
uint64_t cob_amx_six_trace(int index) { (void)index; return 0; }
#endif

__attribute__((target("amx-tile,amx-bf16")))
int cob_amx_six_tiles(int m, int n, int k, const uint16_t *packed_a,
                      const uint16_t *packed_b, float *c, int ldc)
{
    if (m < 1 || n < 1 || k < 1 || ldc < n || !packed_a || !packed_b || !c)
        return 1;
    const size_t mt = ((size_t)m + 15) / 16;
    const size_t nt = ((size_t)n + 15) / 16;
    const size_t kt = ((size_t)k + 31) / 32;
    enum { panel_elements = 16*32, row_bytes = 64 };
    if (mt > (size_t)PTRDIFF_MAX / (3 * panel_elements * sizeof(uint16_t)) / kt ||
        nt > (size_t)PTRDIFF_MAX / (3 * panel_elements * sizeof(uint16_t)) / kt ||
        (size_t)m > (size_t)PTRDIFF_MAX / sizeof(float) / (size_t)ldc)
        return 1;
    const size_t a_plane = mt * kt * panel_elements;
    const size_t b_plane = nt * kt * panel_elements;
    _Alignas(64) float d0[16][16], d1[16][16], d2[16][16];
    /* GCC 13's _tile_loadconfig prints a QWORD operand in Intel syntax;
       the instruction takes the full 64-byte record with an implicit size. */
    __asm__ volatile ("ldtilecfg {(%0)|[%0]}" : : "r"(&config), "m"(config) : "memory");
    /* Two adjacent B panels serve up to eight row tiles before advancing N. */
    for (size_t i0 = 0; i0 < mt; i0 += 8) {
        const size_t i_end = mt-i0 < 8 ? mt : i0+8;
        for (size_t j0 = 0; j0 < nt; j0 += 2) {
            const size_t j_end = nt-j0 < 2 ? nt : j0+2;
            for (size_t i = i0; i < i_end; ++i) {
                const size_t row = i * 16;
                const size_t nr = (size_t)m - row < 16 ? (size_t)m - row : 16;
                for (size_t j = j0; j < j_end; ++j) {
                    const size_t col = j * 16;
                    const size_t nc = (size_t)n - col < 16 ? (size_t)n - col : 16;
                    const uint16_t *a = packed_a + i * kt * panel_elements;
                    const uint16_t *b = packed_b + j * kt * panel_elements;
                    _tile_zero(0);
                    _tile_zero(1);
                    _tile_zero(2);
                    for (size_t p = 0; p < kt; ++p, a += panel_elements, b += panel_elements) {
                        _tile_loadd(3, a, row_bytes);
                        _tile_loadd(4, b, row_bytes);
                        _tile_dpbf16ps(0, 3, 4);    /* hA × hB */
                        _tile_loadd(5, a + a_plane, row_bytes);
                        _tile_loadd(6, b + b_plane, row_bytes);
                        _tile_dpbf16ps(1, 3, 6);    /* hA × mB */
                        _tile_dpbf16ps(1, 5, 4);    /* mA × hB */
                        _tile_loadd(7, b + 2*b_plane, row_bytes);
                        _tile_dpbf16ps(2, 3, 7);    /* hA × lB */
                        _tile_dpbf16ps(2, 5, 6);    /* mA × mB */
                        _tile_loadd(7, a + 2*a_plane, row_bytes);
                        _tile_dpbf16ps(2, 7, 4);    /* lA × hB */
#if defined(COB_AMX_TRACE) && COB_AMX_TRACE
                        ++observed_kblocks;
                        observed_dpbf16 += 6;
#endif
                    }
                    _tile_stored(0, d0, row_bytes);
                    _tile_stored(1, d1, row_bytes);
                    _tile_stored(2, d2, row_bytes);
#if defined(COB_AMX_TRACE) && COB_AMX_TRACE
                    ++observed_tiles;
#endif
                    for (size_t r = 0; r < nr; ++r) {
                        float *out = c + (row+r)*(size_t)ldc + col;
                        for (size_t s = 0; s < nc; ++s)
                            out[s] = (d2[r][s] + d1[r][s]) + d0[r][s];
                    }
                }
            }
        }
    }
    _tile_release();
    return 0;
}
#else
int cob_amx_six_compiled(void) { return 0; }
uint64_t cob_amx_six_trace(int index) { (void)index; return 0; }
int cob_amx_six_tiles(int m, int n, int k, const uint16_t *packed_a,
                      const uint16_t *packed_b, float *c, int ldc)
{
    (void)m; (void)n; (void)k; (void)packed_a; (void)packed_b; (void)c; (void)ldc;
    return 1;
}
#endif
