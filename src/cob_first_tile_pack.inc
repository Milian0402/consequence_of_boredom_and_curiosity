/* Compute the first output row tile while copying unpadded B columns to
   the same packed layout consumed by the remaining conventional tiles. */

__attribute__((target("avx2,fma")))
static void cob_first_pack_6x16(int k, const float* a, const float* b, int ldb,
    float* packed, float* c, int ldc, int resume)
{
    __m256 low[6], high[6];
    for (int r = 0; r < 6; ++r) {
        const float* row = c + (size_t)r * ldc;
        low[r] = resume ? _mm256_loadu_ps(row) : _mm256_setzero_ps();
        high[r] = resume ? _mm256_loadu_ps(row + 8) : _mm256_setzero_ps();
    }
    for (int p = 0; p < k; ++p) {
        const float* source = b + (size_t)p * ldb;
        const __m256 first = _mm256_loadu_ps(source);
        const __m256 second = _mm256_loadu_ps(source + 8);
        _mm256_store_ps(packed, first);
        _mm256_store_ps(packed + 8, second);
#pragma GCC unroll 6
        for (int r = 0; r < 6; ++r) {
            const __m256 left = _mm256_set1_ps(a[r]);
            low[r] = _mm256_fmadd_ps(left, first, low[r]);
            high[r] = _mm256_fmadd_ps(left, second, high[r]);
        }
        a += 6;
        packed += 16;
    }
    for (int r = 0; r < 6; ++r) {
        float* row = c + (size_t)r * ldc;
        _mm256_storeu_ps(row, low[r]);
        _mm256_storeu_ps(row + 8, high[r]);
    }
}

__attribute__((target("avx512f,fma")))
static void cob_first_pack_12x32(int k, const float* a, const float* b, int ldb,
    float* packed, float* c, int ldc, int resume)
{
    __m512 low[12], high[12];
    for (int r = 0; r < 12; ++r) {
        const float* row = c + (size_t)r * ldc;
        low[r] = resume ? _mm512_loadu_ps(row) : _mm512_setzero_ps();
        high[r] = resume ? _mm512_loadu_ps(row + 16) : _mm512_setzero_ps();
    }
    for (int p = 0; p < k; ++p) {
        const float* source = b + (size_t)p * ldb;
        const __m512 first = _mm512_loadu_ps(source);
        const __m512 second = _mm512_loadu_ps(source + 16);
        _mm512_store_ps(packed, first);
        _mm512_store_ps(packed + 16, second);
#pragma GCC unroll 12
        for (int r = 0; r < 12; ++r) {
            const __m512 left = _mm512_set1_ps(a[r]);
            low[r] = _mm512_fmadd_ps(left, first, low[r]);
            high[r] = _mm512_fmadd_ps(left, second, high[r]);
        }
        a += 12;
        packed += 32;
    }
    for (int r = 0; r < 12; ++r) {
        float* row = c + (size_t)r * ldc;
        _mm512_storeu_ps(row, low[r]);
        _mm512_storeu_ps(row + 16, high[r]);
    }
}

static void cob_first_pack_panel(cob_blocked_kernel kernel, int n, int k,
    const float* a, const float* b, int ldb, float* packed, float* c, int ldc, int resume)
{
    const int nr = kernel.nr;
    const int complete = n / nr * nr;
    for (int j = 0; j < complete; j += nr) {
        float* panel = packed + (size_t)j * k;
        if (kernel.mr == 6) {
            cob_first_pack_6x16(k, a, b + j, ldb, panel, c + j, ldc, resume);
        } else {
            cob_first_pack_12x32(k, a, b + j, ldb, panel, c + j, ldc, resume);
        }
    }
    if (complete < n) {
        float* panel = packed + (size_t)complete * k;
        const int cols = n - complete;
        cob_blocked_pack_b(panel, b + complete, ldb, k, cols, nr);
        cob_blocked_edge(kernel.tile, k, a, panel, c + complete, ldc,
            kernel.mr, cols, nr, resume);
    }
}
