/* Packed panels are private to a single multiply. Increasing the column
   slab gives each transposed A panel more output columns over which to
   amortize its packing, independently of the caller's leading dimensions. */
typedef void (*cob_blocked_tile_fn)(int, const float*, const float*, float*, int, int);

#if defined(COB_USE_X86_SIMD)
__attribute__((target("avx512f,fma")))
static void cob_blocked_tile_12x32(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
#define COB_BLOCKED_INIT_ROW(r) \
    __m512 lo##r = resume ? _mm512_loadu_ps(c + (size_t)(r)*ldc) : _mm512_setzero_ps(); \
    __m512 hi##r = resume ? _mm512_loadu_ps(c + (size_t)(r)*ldc + 16) : _mm512_setzero_ps()
    COB_BLOCKED_INIT_ROW(0);
    COB_BLOCKED_INIT_ROW(1);
    COB_BLOCKED_INIT_ROW(2);
    COB_BLOCKED_INIT_ROW(3);
    COB_BLOCKED_INIT_ROW(4);
    COB_BLOCKED_INIT_ROW(5);
    COB_BLOCKED_INIT_ROW(6);
    COB_BLOCKED_INIT_ROW(7);
    COB_BLOCKED_INIT_ROW(8);
    COB_BLOCKED_INIT_ROW(9);
    COB_BLOCKED_INIT_ROW(10);
    COB_BLOCKED_INIT_ROW(11);
#undef COB_BLOCKED_INIT_ROW
    for (int p = 0; p < k; ++p) {
        const __m512 left = _mm512_load_ps(b), right = _mm512_load_ps(b + 16);
#define COB_BLOCKED_FMA_ROW(r) do { \
        const __m512 av = _mm512_set1_ps(a[r]); \
        lo##r = _mm512_fmadd_ps(av, left, lo##r); \
        hi##r = _mm512_fmadd_ps(av, right, hi##r); \
    } while (0)
        COB_BLOCKED_FMA_ROW(0);
        COB_BLOCKED_FMA_ROW(1);
        COB_BLOCKED_FMA_ROW(2);
        COB_BLOCKED_FMA_ROW(3);
        COB_BLOCKED_FMA_ROW(4);
        COB_BLOCKED_FMA_ROW(5);
        COB_BLOCKED_FMA_ROW(6);
        COB_BLOCKED_FMA_ROW(7);
        COB_BLOCKED_FMA_ROW(8);
        COB_BLOCKED_FMA_ROW(9);
        COB_BLOCKED_FMA_ROW(10);
        COB_BLOCKED_FMA_ROW(11);
#undef COB_BLOCKED_FMA_ROW
        a += 12;
        b += 32;
    }
#define COB_BLOCKED_STORE_ROW(r) do { \
    _mm512_storeu_ps(c + (size_t)(r)*ldc, lo##r); \
    _mm512_storeu_ps(c + (size_t)(r)*ldc + 16, hi##r); \
} while (0)
    COB_BLOCKED_STORE_ROW(0);
    COB_BLOCKED_STORE_ROW(1);
    COB_BLOCKED_STORE_ROW(2);
    COB_BLOCKED_STORE_ROW(3);
    COB_BLOCKED_STORE_ROW(4);
    COB_BLOCKED_STORE_ROW(5);
    COB_BLOCKED_STORE_ROW(6);
    COB_BLOCKED_STORE_ROW(7);
    COB_BLOCKED_STORE_ROW(8);
    COB_BLOCKED_STORE_ROW(9);
    COB_BLOCKED_STORE_ROW(10);
    COB_BLOCKED_STORE_ROW(11);
#undef COB_BLOCKED_STORE_ROW
}
#elif defined(COB_USE_NEON)
static void cob_blocked_tile_8x12(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
#define COB_BLOCKED_INIT_ROW(r) \
    float32x4_t c##r##0 = resume ? vld1q_f32(c + (size_t)(r)*ldc) : vdupq_n_f32(0); \
    float32x4_t c##r##1 = resume ? vld1q_f32(c + (size_t)(r)*ldc + 4) : vdupq_n_f32(0); \
    float32x4_t c##r##2 = resume ? vld1q_f32(c + (size_t)(r)*ldc + 8) : vdupq_n_f32(0)
    COB_BLOCKED_INIT_ROW(0);
    COB_BLOCKED_INIT_ROW(1);
    COB_BLOCKED_INIT_ROW(2);
    COB_BLOCKED_INIT_ROW(3);
    COB_BLOCKED_INIT_ROW(4);
    COB_BLOCKED_INIT_ROW(5);
    COB_BLOCKED_INIT_ROW(6);
    COB_BLOCKED_INIT_ROW(7);
#undef COB_BLOCKED_INIT_ROW
    for (int p = 0; p < k; ++p) {
        const float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
        const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
#define COB_BLOCKED_FMA_ROW(row, av, lane) do { \
        c##row##0 = vfmaq_laneq_f32(c##row##0, b0, av, lane); \
        c##row##1 = vfmaq_laneq_f32(c##row##1, b1, av, lane); \
        c##row##2 = vfmaq_laneq_f32(c##row##2, b2, av, lane); \
    } while (0)
        COB_BLOCKED_FMA_ROW(0, a0, 0);
        COB_BLOCKED_FMA_ROW(1, a0, 1);
        COB_BLOCKED_FMA_ROW(2, a0, 2);
        COB_BLOCKED_FMA_ROW(3, a0, 3);
        COB_BLOCKED_FMA_ROW(4, a1, 0);
        COB_BLOCKED_FMA_ROW(5, a1, 1);
        COB_BLOCKED_FMA_ROW(6, a1, 2);
        COB_BLOCKED_FMA_ROW(7, a1, 3);
#undef COB_BLOCKED_FMA_ROW
        a += 8;
        b += 12;
    }
#define COB_BLOCKED_STORE_ROW(r) do { \
    vst1q_f32(c + (size_t)(r)*ldc, c##r##0); \
    vst1q_f32(c + (size_t)(r)*ldc + 4, c##r##1); \
    vst1q_f32(c + (size_t)(r)*ldc + 8, c##r##2); \
} while (0)
    COB_BLOCKED_STORE_ROW(0);
    COB_BLOCKED_STORE_ROW(1);
    COB_BLOCKED_STORE_ROW(2);
    COB_BLOCKED_STORE_ROW(3);
    COB_BLOCKED_STORE_ROW(4);
    COB_BLOCKED_STORE_ROW(5);
    COB_BLOCKED_STORE_ROW(6);
    COB_BLOCKED_STORE_ROW(7);
#undef COB_BLOCKED_STORE_ROW
}
#endif

#if defined(COB_USE_X86_SIMD)
typedef __m128 cob_blocked_vec4;
#define cob_blocked_load4 _mm_loadu_ps
#define cob_blocked_add4 _mm_add_ps
#define cob_blocked_sub4 _mm_sub_ps
#else
typedef float32x4_t cob_blocked_vec4;
#define cob_blocked_load4 vld1q_f32
#define cob_blocked_add4 vaddq_f32
#define cob_blocked_sub4 vsubq_f32
#endif

static inline void cob_blocked_store_transpose4(cob_blocked_vec4 a0, cob_blocked_vec4 a1,
    cob_blocked_vec4 a2, cob_blocked_vec4 a3, float* dst, int mr)
{
#if defined(COB_USE_X86_SIMD)
    _MM_TRANSPOSE4_PS(a0, a1, a2, a3);
    _mm_storeu_ps(dst, a0);
    _mm_storeu_ps(dst + mr, a1);
    _mm_storeu_ps(dst + 2*mr, a2);
    _mm_storeu_ps(dst + 3*mr, a3);
#else
    const float32x4_t lo0 = vzip1q_f32(a0, a1), lo1 = vzip1q_f32(a2, a3);
    const float32x4_t hi0 = vzip2q_f32(a0, a1), hi1 = vzip2q_f32(a2, a3);
    vst1q_f32(dst, vcombine_f32(vget_low_f32(lo0), vget_low_f32(lo1)));
    vst1q_f32(dst + mr, vcombine_f32(vget_high_f32(lo0), vget_high_f32(lo1)));
    vst1q_f32(dst + 2*mr, vcombine_f32(vget_low_f32(hi0), vget_low_f32(hi1)));
    vst1q_f32(dst + 3*mr, vcombine_f32(vget_high_f32(hi0), vget_high_f32(hi1)));
#endif
}

static inline void cob_blocked_transpose4(const float* a, int lda, float* dst, int mr)
{
    cob_blocked_store_transpose4(cob_blocked_load4(a), cob_blocked_load4(a + (size_t)lda),
        cob_blocked_load4(a + (size_t)2*lda), cob_blocked_load4(a + (size_t)3*lda), dst, mr);
}

static void cob_blocked_pack_a(float* packed, const float* a, int lda, int m, int k, int mr)
{
    for (int i = 0; i < m; i += mr) {
        const float* first = a + (size_t)i * lda;
        const int rows = cob_min_i32(mr, m - i);
        int p = 0;
        if (rows == mr) {
            for (; k - p >= 4; p += 4) {
                float* dst = packed + (size_t)p * mr;
                int r = 0;
                for (; r + 4 <= mr; r += 4)
                    cob_blocked_transpose4(first + (size_t)r*lda + p, lda, dst + r, mr);
                for (; r < mr; ++r)
                    for (int q = 0; q < 4; ++q)
                        dst[q*mr+r] = first[(size_t)r*lda+p+q];
            }
        }
        for (; p < k; ++p) {
            float* dst = packed + (size_t)p * mr;
            for (int r = 0; r < rows; ++r) dst[r] = first[(size_t)r*lda+p];
            for (int r = rows; r < mr; ++r) dst[r] = 0;
        }
        packed += (size_t)mr * k;
    }
}

static void cob_blocked_pack_b(float* packed, const float* b, int ldb, int k, int n, int nr)
{
    /* Visit a small group of source rows before moving to the next one.
       Each output panel remains K-major for the microkernel. */
    for (int p0 = 0; p0 < k; p0 += 16) {
        const int count = cob_min_i32(16, k - p0);
        for (int j = 0; j < n; j += nr) {
            float* dst = packed + (size_t)j*k + (size_t)p0*nr;
            const float* src = b + (size_t)p0*ldb + j;
            const int cols = cob_min_i32(nr, n - j);
            for (int p = 0; p < count; ++p) {
                if (cols == 32) memcpy(dst, src, 32*sizeof(float));
                else if (cols == 16) memcpy(dst, src, 16*sizeof(float));
                else if (cols == 12) memcpy(dst, src, 12*sizeof(float));
                else memcpy(dst, src, (size_t)cols*sizeof(float));
                if (cols < nr) memset(dst + cols, 0, (size_t)(nr-cols)*sizeof(float));
                dst += nr;
                if (p + 1 < count) src += ldb;
            }
        }
    }
}

static void cob_blocked_edge(cob_blocked_tile_fn tile, int k, const float* a, const float* b,
    float* c, int ldc, int rows, int cols, int nr, int resume)
{
    _Alignas(64) float edge[12 * 32] = {0};
    if (resume)
        for (int i = 0; i < rows; ++i)
            memcpy(edge + i*nr, c + (size_t)i*ldc, (size_t)cols*sizeof(float));
    tile(k, a, b, edge, nr, resume);
    for (int i = 0; i < rows; ++i)
        memcpy(c + (size_t)i*ldc, edge + i*nr, (size_t)cols*sizeof(float));
}

typedef struct {
    int mr, nr, mc;
    cob_blocked_tile_fn tile;
} cob_blocked_kernel;

static int cob_blocked_select_kernel(cob_blocked_kernel* kernel)
{
#if defined(COB_USE_X86_SIMD)
    if (!__builtin_cpu_supports("fma")) return 0;
#if !defined(COB_DISABLE_X86_AVX512)
    if (__builtin_cpu_supports("avx512f")) {
        *kernel = (cob_blocked_kernel){12, 32, 120, cob_blocked_tile_12x32};
        return 1;
    }
#endif
    if (!__builtin_cpu_supports("avx2")) return 0;
    *kernel = (cob_blocked_kernel){6, 16, 120, cob_sgemm_tile_avx2};
#else
    *kernel = (cob_blocked_kernel){8, 12, 128, cob_blocked_tile_8x12};
#endif
    return 1;
}

static void cob_blocked_apply_panel(cob_blocked_kernel kernel, int m, int n, int k,
    const float* pack_a, const float* pack_b, float* c, int ldc, int resume)
{
    for (int j = 0; j < n; j += kernel.nr) {
        const float* bp = pack_b + (size_t)j*k;
        const int cols = cob_min_i32(kernel.nr, n-j);
        for (int i = 0; i < m; i += kernel.mr) {
            const float* ap = pack_a + (size_t)i*k;
            float* cp = c + (size_t)i*ldc + j;
            const int rows = cob_min_i32(kernel.mr, m-i);
            if (rows == kernel.mr && cols == kernel.nr) kernel.tile(k, ap, bp, cp, ldc, resume);
            else cob_blocked_edge(kernel.tile, k, ap, bp, cp, ldc, rows, cols, kernel.nr, resume);
        }
    }
}

static int cob_sgemm_one_shot_blocked(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if (m < 12 || n < 24 || k < 32) return 0;
    size_t c_count = (size_t)(m-1)*ldc+n;
    if (cob_f32_ranges_overlap(c, c_count, a, (size_t)(m-1)*lda+k) ||
        cob_f32_ranges_overlap(c, c_count, b, (size_t)(k-1)*ldb+n)) return 0;
    cob_blocked_kernel kernel;
    if (!cob_blocked_select_kernel(&kernel)) return 0;
    const int mc = kernel.mc, mr = kernel.mr, nr = kernel.nr;
    const int kc = cob_min_i32(256, k), nc = cob_min_i32(768, n);
    const int packed_n = (nc + nr - 1) / nr * nr;
    const int packed_m = (cob_min_i32(m, mc) + mr - 1) / mr * mr;
    float* pack_b = cob_aligned_alloc(64, (size_t)(packed_n + packed_m)*kc*sizeof(float));
    if (!pack_b) return 0;
    float* pack_a = pack_b + (size_t)packed_n*kc;
    for (int jc = 0; jc < n;) {
        int nb = cob_min_i32(nc, n-jc);
        for (int pc = 0; pc < k;) {
            int kb = cob_min_i32(kc, k-pc);
            cob_blocked_pack_b(pack_b, b + (size_t)pc*ldb + jc, ldb, kb, nb, nr);
            for (int ic = 0; ic < m;) {
                int mb = cob_min_i32(mc, m-ic);
                cob_blocked_pack_a(pack_a, a + (size_t)ic*lda + pc, lda, mb, kb, mr);
                cob_blocked_apply_panel(kernel, mb, nb, kb, pack_a, pack_b,
                    c + (size_t)ic*ldc + jc, ldc, pc != 0);
                ic += mb;
            }
            pc += kb;
        }
        jc += nb;
    }
    free(pack_b);
    return 1;
}
