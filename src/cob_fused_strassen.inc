/* One-level Strassen with sums in the input packers. Each output slab is
   finished before moving on, so scratch is bounded independently of M,N,K. */
#ifndef COB_FUSED_STRASSEN_MIN_HARMONIC_DIM
#define COB_FUSED_STRASSEN_MIN_HARMONIC_DIM 1024
#endif

#ifndef COB_FUSED_MC
#define COB_FUSED_MC 2048
#endif
#ifndef COB_FUSED_NC
#define COB_FUSED_NC 768
#endif
enum { COB_FUSED_KC = 256 };
#if defined(COB_USE_X86_SIMD) && defined(COB_ENABLE_HK15_PARENT_960) && COB_ENABLE_HK15_PARENT_960 && defined(COB_USE_HK15)
#if defined(COB_ENABLE_HK_X86_REGISTER) && COB_ENABLE_HK_X86_REGISTER
#define COB_USE_HK_REGISTER 1
#include "cob_hk_x86_register.inc"
#endif
#if defined(COB_USE_HK_REGISTER) || (defined(COB_ENABLE_HK_X86_K288) && COB_ENABLE_HK_X86_K288)
#define COB_USE_HK_X86_LEAF 1
#if defined(COB_HK_X86_LEAF_TRACE) && COB_HK_X86_LEAF_TRACE
#define COB_TRACE_HK_X86_LEAF 1
static _Thread_local int cob_hk_x86_children, cob_hk_x86_products, cob_hk_x86_nr;
int rank_hk_x86_leaf_count(int id) {
    return id == 0 ? cob_hk_x86_children : id == 1 ? cob_hk_x86_products : id == 2 ? cob_hk_x86_nr : -1;
}
#endif
#endif
#endif
#if defined(COB_USE_HK_X86_LEAF) && defined(COB_ENABLE_HK_X86_K288) && COB_ENABLE_HK_X86_K288
#define COB_USE_HK_X86_K288 1
/* A 513..576-deep product usually needs three 256-deep panels. Equalize
   two panels here, bounding the extra packing workspace to 32 depth rows. */
static int cob_hk_x86_k288_limit(cob_blocked_kernel kernel, int depth)
{
    if (depth >= 513 && depth <= 576 &&
        ((kernel.mr == 6 && kernel.nr == 16) || (kernel.mr == 12 && kernel.nr == 32)))
        return (depth + 1) / 2;
    return COB_FUSED_KC;
}
#endif
#if COB_FUSED_MC < 1 || COB_FUSED_MC > 2048 || COB_FUSED_NC < 1 || COB_FUSED_NC > 768
#error "Fused slab dimensions exceed workspace bounds"
#endif

#if defined(COB_ENABLE_HK_PACK_B_FORMS) && COB_ENABLE_HK_PACK_B_FORMS && \
    defined(COB_USE_HK15) && defined(COB_ENABLE_HK15_PARENT_960) && COB_ENABLE_HK15_PARENT_960 && \
    !defined(COB_USE_HK_X86_LEAF) && !(defined(COB_ENABLE_OPENBLAS_LEAF) && COB_ENABLE_OPENBLAS_LEAF) && !defined(COB_USE_APPLE_AMX)
#define COB_USE_HK_PACK_B 1
#include "cob_hk_pack_b.inc"
#endif

typedef struct {
    const float* first;
    const float* second;
    int sign;
} cob_fused_operand;

static inline cob_blocked_vec4 cob_fused_load4(const float* x, const float* y, int sign)
{
    return sign > 0 ? cob_blocked_add4(cob_blocked_load4(x), cob_blocked_load4(y)) :
                      cob_blocked_sub4(cob_blocked_load4(x), cob_blocked_load4(y));
}

static inline float cob_fused_value(float x, float y, int sign)
{
    return sign > 0 ? x + y : x - y;
}

static void cob_fused_pack_a(float* packed, cob_fused_operand a, size_t offset,
    int lda, int m, int k, int mr)
{
    if (!a.second) {
        cob_blocked_pack_a(packed, a.first + offset, lda, m, k, mr);
        return;
    }
    for (int i = 0; i < m; i += mr) {
        const float* x = a.first + offset + (size_t)i*lda;
        const float* y = a.second + offset + (size_t)i*lda;
        const int rows = cob_min_i32(mr, m-i);
        int p = 0;
        if (rows == mr) {
            for (; k-p >= 4; p += 4) {
                float* dst = packed + (size_t)p*mr;
                int r = 0;
                for (; r+4 <= mr; r += 4) {
                    const size_t z = (size_t)r*lda + p;
                    cob_blocked_store_transpose4(
                        cob_fused_load4(x+z, y+z, a.sign),
                        cob_fused_load4(x+z+lda, y+z+lda, a.sign),
                        cob_fused_load4(x+z+(size_t)2*lda, y+z+(size_t)2*lda, a.sign),
                        cob_fused_load4(x+z+(size_t)3*lda, y+z+(size_t)3*lda, a.sign),
                        dst+r, mr);
                }
                for (; r < mr; ++r)
                    for (int q = 0; q < 4; ++q) {
                        const size_t z = (size_t)r*lda+p+q;
                        dst[q*mr+r] = cob_fused_value(x[z], y[z], a.sign);
                    }
            }
        }
        for (; p < k; ++p) {
            float* dst = packed + (size_t)p*mr;
            for (int r = 0; r < rows; ++r) {
                const size_t z = (size_t)r*lda+p;
                dst[r] = cob_fused_value(x[z], y[z], a.sign);
            }
            for (int r = rows; r < mr; ++r) dst[r] = 0;
        }
        packed += (size_t)k*mr;
    }
}

static inline void cob_fused_pack_row(float* restrict dst, const float* x,
    const float* y, int cols, int sign)
{
    if (sign > 0) {
        for (int j = 0; j < cols; ++j) dst[j] = x[j] + y[j];
    } else {
        for (int j = 0; j < cols; ++j) dst[j] = x[j] - y[j];
    }
}

static void cob_fused_pack_b(float* packed, cob_fused_operand b, size_t offset,
    int ldb, int k, int n, int nr)
{
    if (!b.second) {
        cob_blocked_pack_b(packed, b.first + offset, ldb, k, n, nr);
        return;
    }
    for (int p0 = 0; p0 < k; p0 += 16) {
        const int count = cob_min_i32(16, k-p0);
        for (int j = 0; j < n; j += nr) {
            float* dst = packed + (size_t)j*k + (size_t)p0*nr;
            const int cols = cob_min_i32(nr, n-j);
            for (int p = 0; p < count; ++p) {
                const size_t z = offset + (size_t)(p0+p)*ldb+j;
                cob_fused_pack_row(dst, b.first+z, b.second+z, cols, b.sign);
                if (cols != nr) memset(dst+cols, 0, (size_t)(nr-cols)*sizeof(float));
                dst += nr;
            }
        }
    }
}

typedef struct {
    cob_blocked_kernel kernel;
    int rows, cols, depth, lda, ldb;
    float* pack_a;
    float* pack_b;
#ifdef COB_USE_HK_PACK_B
    const float* prepacked_b;
    size_t pitch_b;
    int packed_n;
#endif
#if defined(COB_USE_HK_X86_K288)
    int kc;
#endif
#if defined(COB_USE_HK_X86_LEAF)
    int x86_child;
#endif
} cob_fused_slab;

#if defined(COB_ENABLE_FUSED_FINAL_PANEL) && COB_ENABLE_FUSED_FINAL_PANEL
#include "cob_final_product.inc"
#else
typedef struct cob_final_output cob_final_output;
#endif

#ifdef COB_USE_HK_PACK_B
#define COB_HK_B_ARG(x) , x
#else
#define COB_HK_B_ARG(x)
#endif
static void cob_fused_product(const cob_fused_slab* s, cob_fused_operand a,
    cob_fused_operand b, float* c, int ldc, const cob_final_output* output COB_HK_B_ARG(enum cob_hk_b_slot b_slot))
{
#if !defined(COB_ENABLE_FUSED_FINAL_PANEL) || !COB_ENABLE_FUSED_FINAL_PANEL
    (void)output;
#endif
#ifdef COB_TRACE_HK_X86_LEAF
    if (s->x86_child) ++cob_hk_x86_products;
#endif
#ifdef COB_USE_HK_PACK_B
    const float* prepared = s->prepacked_b ? s->prepacked_b+(size_t)b_slot*s->pitch_b : NULL;
#ifdef COB_TRACE_HK_PACK_B
    if (prepared) ++cob_hk_panel_uses;
#endif
#endif
    for (int pc = 0; pc < s->depth;) {
#if defined(COB_USE_HK_X86_K288)
        const int kb = cob_min_i32(s->kc, s->depth-pc);
#else
        const int kb = cob_min_i32(COB_FUSED_KC, s->depth-pc);
#endif
#ifdef COB_USE_HK_PACK_B
        const float* packed_b = s->pack_b;
        if (prepared)
            packed_b = prepared+(size_t)pc*s->packed_n;
        else
#else
#define packed_b (s->pack_b)
#endif
            cob_fused_pack_b(s->pack_b, b, (size_t)pc*s->ldb, s->ldb, kb, s->cols, s->kernel.nr);
        for (int ic = 0; ic < s->rows;) {
            const int mb = cob_min_i32(s->kernel.mc, s->rows-ic);
            cob_fused_pack_a(s->pack_a, a, (size_t)ic*s->lda+pc, s->lda, mb, kb, s->kernel.mr);
#if defined(COB_ENABLE_FUSED_FINAL_PANEL) && COB_ENABLE_FUSED_FINAL_PANEL
            if (output && kb == s->depth-pc)
                cob_final_panel(&s->kernel,mb,s->cols,kb,s->pack_a,packed_b,
                    c+(size_t)ic*ldc,ldc,pc!=0,output,ic);
            else
#endif
#ifdef COB_USE_HK_REGISTER
            if (s->x86_child)
                cob_hk_reg_apply(s->kernel, mb, s->cols, kb, s->pack_a, packed_b,
                    c+(size_t)ic*ldc, ldc, pc != 0);
            else
#endif
                cob_blocked_apply_panel(s->kernel, mb, s->cols, kb, s->pack_a, packed_b,
                    c+(size_t)ic*ldc, ldc, pc != 0);
            ic += mb;
        }
        pc += kb;
    }
#undef packed_b
}

/* Compute the three one-element fringes directly. In the right column,
   gathering B once per depth block makes each row's dot product contiguous. */
static void cob_fused_finish_edges(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    const int me = m-m%2, ne = n-n%2;
    if (k%2) {
        const float* right = b+(size_t)(k-1)*ldb;
        for (int i = 0; i < me; ++i) {
            const float left = a[(size_t)i*lda+k-1];
            float* row = c+(size_t)i*ldc;
            for (int j = 0; j < ne; ++j) row[j] += left*right[j];
        }
    }
    if (n%2) {
        float column[COB_FUSED_KC];
        for (int pc = 0; pc < k;) {
            const int kb = cob_min_i32(COB_FUSED_KC, k-pc);
            for (int p = 0; p < kb; ++p) column[p] = b[(size_t)(pc+p)*ldb+ne];
            for (int i = 0; i < me; ++i) {
                const float* row = a+(size_t)i*lda+pc;
                float* dst = c+(size_t)i*ldc+ne;
                float value = pc ? *dst : 0;
                for (int p = 0; p < kb; ++p) value += row[p]*column[p];
                *dst = value;
            }
            pc += kb;
        }
    }
    if (m%2) {
        float* dst = c+(size_t)me*ldc;
        memset(dst, 0, (size_t)n*sizeof(float));
        for (int p = 0; p < k; ++p) {
            const float left = a[(size_t)me*lda+p];
            const float* right = b+(size_t)p*ldb;
            for (int j = 0; j < n; ++j) dst[j] += left*right[j];
        }
    }
}

static int cob_sgemm_strassen_fused_core(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, int min_harmonic, int hk_child
#ifdef COB_USE_HK_PACK_B
    , const cob_hk_b_form *form
#endif
    )
{
#ifndef COB_USE_HK_X86_LEAF
    (void)hk_child;
#endif
    const int min_dim = cob_min_i32(m, cob_min_i32(n,k));
    const int max_dim = m > n ? (m > k ? m : k) : (n > k ? n : k);
    if (min_dim < 2 || min_dim < min_harmonic / 2 ||
        (int64_t)max_dim > (int64_t)min_dim*8 || !a || !b || !c ||
        lda < k || ldb < n || ldc < n) return 0;
    const double areas = (double)m*n + (double)m*k + (double)n*k;
    if (3.0*m*n*k < min_harmonic*areas) return 0;
    const size_t cspan = (size_t)(m-1)*ldc+n;
    if (cob_f32_ranges_overlap(c, cspan, a, (size_t)(m-1)*lda+k) ||
        cob_f32_ranges_overlap(c, cspan, b, (size_t)(k-1)*ldb+n)) return 0;

    cob_blocked_kernel kernel;
    if (!cob_blocked_select_kernel(&kernel)) return 0;
    const int mh = m/2, nh = n/2, kh = k/2;
    const int rows = cob_min_i32(mh, COB_FUSED_MC), cols = cob_min_i32(nh, COB_FUSED_NC);
#if defined(COB_USE_HK_X86_K288)
    const int kc = hk_child ? cob_hk_x86_k288_limit(kernel, kh) : COB_FUSED_KC;
    const int kb = cob_min_i32(kh, kc);
#else
    const int kb = cob_min_i32(kh, COB_FUSED_KC);
#endif
    const int packed_m = (cob_min_i32(rows,kernel.mc)+kernel.mr-1)/kernel.mr*kernel.mr;
    const int packed_n = (cols+kernel.nr-1)/kernel.nr*kernel.nr;
    const size_t pack_a_size = (size_t)packed_m*kb, pack_b_size = (size_t)packed_n*kb;
    const size_t ordinary = pack_b_size+pack_a_size+(size_t)rows*cols;
    size_t slots = ordinary;
#ifdef COB_USE_HK_PACK_B
    const int form_n = (nh+kernel.nr-1)/kernel.nr*kernel.nr;
    const size_t pitch = (size_t)kh*form_n;
    if (form) {
        if (m>1280 || n>1280 || k>1280 || rows!=mh || cols!=nh) return 0;
        slots += 7*pitch;
    }
#endif
    float* work = cob_aligned_alloc(64, slots*sizeof(float));
    if (!work) return 0;
    float* product = work+pack_b_size+pack_a_size;
    cob_fused_slab slab = {kernel, 0, 0, kh, lda, ldb, work+pack_b_size, work,
#ifdef COB_USE_HK_PACK_B
                           form ? work+ordinary : NULL, pitch, form_n,
#endif
#if defined(COB_USE_HK_X86_K288)
                           kc,
#endif
#ifdef COB_USE_HK_REGISTER
                           hk_child && ((kernel.mr == 6 && kernel.nr == 16) || (kernel.mr == 12 && kernel.nr == 32)),
#elif defined(COB_USE_HK_X86_K288)
                           hk_child && kc != COB_FUSED_KC,
#endif
    };
#ifdef COB_TRACE_HK_X86_LEAF
    if (slab.x86_child) { ++cob_hk_x86_children; cob_hk_x86_nr = kernel.nr; }
#endif
#ifdef COB_USE_HK_PACK_B
    if (form) cob_hk_pack_seven_b(*form,ldb,kh,nh,kernel.nr,form_n,COB_FUSED_KC,work+ordinary);
#endif
    for (int jc = 0; jc < nh;) {
        slab.cols = cob_min_i32(cols,nh-jc);
        const float* b11 = b+jc;
        const float* b12 = b11+nh;
        const float* b21 = b11+(size_t)kh*ldb;
        const float* b22 = b21+nh;
        for (int ic = 0; ic < mh;) {
            slab.rows = cob_min_i32(rows,mh-ic);
            const float* a11 = a+(size_t)ic*lda;
            const float* a12 = a11+kh;
            const float* a21 = a11+(size_t)mh*lda;
            const float* a22 = a21+kh;
            float* c11 = c+(size_t)ic*ldc+jc;
            float* c12 = c11+nh;
            float* c21 = c11+(size_t)mh*ldc;
            float* c22 = c21+nh;
#define COB_FUSED_ONE(x) ((cob_fused_operand){x, NULL, 0})
#define COB_FUSED_ADD(x,y) ((cob_fused_operand){x, y, 1})
#define COB_FUSED_SUB(x,y) ((cob_fused_operand){x, y, -1})
            /* M2, M3, M4 seed three destinations; the remaining four products
               reuse one local slab and preserve the existing recombination order. */
#if defined(COB_ENABLE_FUSED_FINAL_PANEL) && COB_ENABLE_FUSED_FINAL_PANEL
#define COB_FINAL_OUT(op) (&(cob_final_output){c11,c12,c21,c22,ldc,(op)})
#else
#define COB_FINAL_OUT(op) NULL
#endif
            cob_fused_product(&slab, COB_FUSED_ADD(a21,a22), COB_FUSED_ONE(b11), c21, ldc, NULL COB_HK_B_ARG(COB_B_M2));
            cob_fused_product(&slab, COB_FUSED_ONE(a11), COB_FUSED_SUB(b12,b22), c12, ldc, NULL COB_HK_B_ARG(COB_B_M3));
            cob_fused_product(&slab, COB_FUSED_ONE(a22), COB_FUSED_SUB(b21,b11), c11, ldc, NULL COB_HK_B_ARG(COB_B_M4));
            cob_fused_product(&slab, COB_FUSED_ADD(a11,a22), COB_FUSED_ADD(b11,b22), product, slab.cols, COB_FINAL_OUT(COB_FINAL_M1) COB_HK_B_ARG(COB_B_M1));
#if !defined(COB_ENABLE_FUSED_FINAL_PANEL) || !COB_ENABLE_FUSED_FINAL_PANEL
            cob_strassen_merge_m1(c11, c12, c21, c22, ldc, product, slab.rows, slab.cols);
#endif
            cob_fused_product(&slab, COB_FUSED_ADD(a11,a12), COB_FUSED_ONE(b22), product, slab.cols, COB_FINAL_OUT(COB_FINAL_M5) COB_HK_B_ARG(COB_B_M5));
#if !defined(COB_ENABLE_FUSED_FINAL_PANEL) || !COB_ENABLE_FUSED_FINAL_PANEL
            cob_strassen_merge_m5(c11, c12, ldc, product, slab.rows, slab.cols);
#endif
            cob_fused_product(&slab, COB_FUSED_SUB(a21,a11), COB_FUSED_ADD(b11,b12), product, slab.cols, COB_FINAL_OUT(COB_FINAL_M6) COB_HK_B_ARG(COB_B_M6));
#if !defined(COB_ENABLE_FUSED_FINAL_PANEL) || !COB_ENABLE_FUSED_FINAL_PANEL
            cob_strassen_merge_product(c22, ldc, product, slab.rows, slab.cols, 1);
#endif
            cob_fused_product(&slab, COB_FUSED_SUB(a12,a22), COB_FUSED_ADD(b21,b22), product, slab.cols, COB_FINAL_OUT(COB_FINAL_M7) COB_HK_B_ARG(COB_B_M7));
#if !defined(COB_ENABLE_FUSED_FINAL_PANEL) || !COB_ENABLE_FUSED_FINAL_PANEL
            cob_strassen_merge_product(c11, ldc, product, slab.rows, slab.cols, 1);
#endif
#undef COB_HK_B_ARG
#undef COB_FINAL_OUT
#undef COB_FUSED_ONE
#undef COB_FUSED_ADD
#undef COB_FUSED_SUB
            ic += slab.rows;
        }
        jc += slab.cols;
    }
    free(work);
#ifdef COB_USE_HK_PACK_B
    if (form) cob_hk_b_edges(m,n,k,a,lda,*form,ldb,c,ldc);
    else
#endif
        cob_fused_finish_edges(m, n, k, a, lda, b, ldb, c, ldc);
    return 1;
}

static int cob_sgemm_strassen_fused_impl(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, int min_harmonic, int hk_child)
{
    return cob_sgemm_strassen_fused_core(m,n,k,a,lda,b,ldb,c,ldc,min_harmonic,hk_child
#ifdef COB_USE_HK_PACK_B
        , NULL
#endif
        );
}

static int cob_sgemm_strassen_fused_at(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, int min_harmonic)
{
    return cob_sgemm_strassen_fused_impl(m,n,k,a,lda,b,ldb,c,ldc,min_harmonic,0);
}

static int cob_sgemm_strassen_fused(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    return cob_sgemm_strassen_fused_at(m,n,k,a,lda,b,ldb,c,ldc,COB_FUSED_STRASSEN_MIN_HARMONIC_DIM);
}
