/* Three B vectors per scalar A broadcast. AVX2 uses twelve accumulators
   and AVX-512 uses twenty-four, leaving registers for B and the A scalar.
   A row tile is padded by the existing packer; B is padded by column.
   Incomplete C tiles go through cob_blocked_edge. */

__attribute__((target("avx2,fma")))
static void cob_blocked_wide_4x24(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
    typedef __m256 vec;
    vec c00 = resume ? _mm256_loadu_ps(c + (size_t)0 * ldc + 0) : _mm256_setzero_ps();
    vec c01 = resume ? _mm256_loadu_ps(c + (size_t)0 * ldc + 8) : _mm256_setzero_ps();
    vec c02 = resume ? _mm256_loadu_ps(c + (size_t)0 * ldc + 16) : _mm256_setzero_ps();
    vec c10 = resume ? _mm256_loadu_ps(c + (size_t)1 * ldc + 0) : _mm256_setzero_ps();
    vec c11 = resume ? _mm256_loadu_ps(c + (size_t)1 * ldc + 8) : _mm256_setzero_ps();
    vec c12 = resume ? _mm256_loadu_ps(c + (size_t)1 * ldc + 16) : _mm256_setzero_ps();
    vec c20 = resume ? _mm256_loadu_ps(c + (size_t)2 * ldc + 0) : _mm256_setzero_ps();
    vec c21 = resume ? _mm256_loadu_ps(c + (size_t)2 * ldc + 8) : _mm256_setzero_ps();
    vec c22 = resume ? _mm256_loadu_ps(c + (size_t)2 * ldc + 16) : _mm256_setzero_ps();
    vec c30 = resume ? _mm256_loadu_ps(c + (size_t)3 * ldc + 0) : _mm256_setzero_ps();
    vec c31 = resume ? _mm256_loadu_ps(c + (size_t)3 * ldc + 8) : _mm256_setzero_ps();
    vec c32 = resume ? _mm256_loadu_ps(c + (size_t)3 * ldc + 16) : _mm256_setzero_ps();
    for (int p = 0; p < k; ++p) {
        const vec b0 = _mm256_load_ps(b);
        const vec b1 = _mm256_load_ps(b + 8);
        const vec b2 = _mm256_load_ps(b + 16);
        /* Twelve accumulators, three B registers and one broadcast use all
           sixteen AVX2 registers. Express the broadcast lifetime explicitly
           so compiler scheduling cannot spill an accumulator in the K loop. */
        vec av;
        __asm__(
            "vbroadcastss 0(%[a]), %[av]\n\t"
            "vfmadd231ps %[b0], %[av], %[c00]\n\t"
            "vfmadd231ps %[b1], %[av], %[c01]\n\t"
            "vfmadd231ps %[b2], %[av], %[c02]\n\t"
            "vbroadcastss 4(%[a]), %[av]\n\t"
            "vfmadd231ps %[b0], %[av], %[c10]\n\t"
            "vfmadd231ps %[b1], %[av], %[c11]\n\t"
            "vfmadd231ps %[b2], %[av], %[c12]\n\t"
            "vbroadcastss 8(%[a]), %[av]\n\t"
            "vfmadd231ps %[b0], %[av], %[c20]\n\t"
            "vfmadd231ps %[b1], %[av], %[c21]\n\t"
            "vfmadd231ps %[b2], %[av], %[c22]\n\t"
            "vbroadcastss 12(%[a]), %[av]\n\t"
            "vfmadd231ps %[b0], %[av], %[c30]\n\t"
            "vfmadd231ps %[b1], %[av], %[c31]\n\t"
            "vfmadd231ps %[b2], %[av], %[c32]\n\t"
            : [av] "=&x" (av),
              [c00] "+&x" (c00), [c01] "+&x" (c01), [c02] "+&x" (c02),
              [c10] "+&x" (c10), [c11] "+&x" (c11), [c12] "+&x" (c12),
              [c20] "+&x" (c20), [c21] "+&x" (c21), [c22] "+&x" (c22),
              [c30] "+&x" (c30), [c31] "+&x" (c31), [c32] "+&x" (c32)
            : [a] "r" (a), [b0] "x" (b0), [b1] "x" (b1), [b2] "x" (b2)
            : "memory");
        a += 4;
        b += 24;
    }
    _mm256_storeu_ps(c + (size_t)0 * ldc + 0, c00);
    _mm256_storeu_ps(c + (size_t)0 * ldc + 8, c01);
    _mm256_storeu_ps(c + (size_t)0 * ldc + 16, c02);
    _mm256_storeu_ps(c + (size_t)1 * ldc + 0, c10);
    _mm256_storeu_ps(c + (size_t)1 * ldc + 8, c11);
    _mm256_storeu_ps(c + (size_t)1 * ldc + 16, c12);
    _mm256_storeu_ps(c + (size_t)2 * ldc + 0, c20);
    _mm256_storeu_ps(c + (size_t)2 * ldc + 8, c21);
    _mm256_storeu_ps(c + (size_t)2 * ldc + 16, c22);
    _mm256_storeu_ps(c + (size_t)3 * ldc + 0, c30);
    _mm256_storeu_ps(c + (size_t)3 * ldc + 8, c31);
    _mm256_storeu_ps(c + (size_t)3 * ldc + 16, c32);
}

__attribute__((target("avx512f,fma")))
static void cob_blocked_wide_8x48(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
    typedef __m512 vec;
    vec c00 = resume ? _mm512_loadu_ps(c + (size_t)0 * ldc + 0) : _mm512_setzero_ps();
    vec c01 = resume ? _mm512_loadu_ps(c + (size_t)0 * ldc + 16) : _mm512_setzero_ps();
    vec c02 = resume ? _mm512_loadu_ps(c + (size_t)0 * ldc + 32) : _mm512_setzero_ps();
    vec c10 = resume ? _mm512_loadu_ps(c + (size_t)1 * ldc + 0) : _mm512_setzero_ps();
    vec c11 = resume ? _mm512_loadu_ps(c + (size_t)1 * ldc + 16) : _mm512_setzero_ps();
    vec c12 = resume ? _mm512_loadu_ps(c + (size_t)1 * ldc + 32) : _mm512_setzero_ps();
    vec c20 = resume ? _mm512_loadu_ps(c + (size_t)2 * ldc + 0) : _mm512_setzero_ps();
    vec c21 = resume ? _mm512_loadu_ps(c + (size_t)2 * ldc + 16) : _mm512_setzero_ps();
    vec c22 = resume ? _mm512_loadu_ps(c + (size_t)2 * ldc + 32) : _mm512_setzero_ps();
    vec c30 = resume ? _mm512_loadu_ps(c + (size_t)3 * ldc + 0) : _mm512_setzero_ps();
    vec c31 = resume ? _mm512_loadu_ps(c + (size_t)3 * ldc + 16) : _mm512_setzero_ps();
    vec c32 = resume ? _mm512_loadu_ps(c + (size_t)3 * ldc + 32) : _mm512_setzero_ps();
    vec c40 = resume ? _mm512_loadu_ps(c + (size_t)4 * ldc + 0) : _mm512_setzero_ps();
    vec c41 = resume ? _mm512_loadu_ps(c + (size_t)4 * ldc + 16) : _mm512_setzero_ps();
    vec c42 = resume ? _mm512_loadu_ps(c + (size_t)4 * ldc + 32) : _mm512_setzero_ps();
    vec c50 = resume ? _mm512_loadu_ps(c + (size_t)5 * ldc + 0) : _mm512_setzero_ps();
    vec c51 = resume ? _mm512_loadu_ps(c + (size_t)5 * ldc + 16) : _mm512_setzero_ps();
    vec c52 = resume ? _mm512_loadu_ps(c + (size_t)5 * ldc + 32) : _mm512_setzero_ps();
    vec c60 = resume ? _mm512_loadu_ps(c + (size_t)6 * ldc + 0) : _mm512_setzero_ps();
    vec c61 = resume ? _mm512_loadu_ps(c + (size_t)6 * ldc + 16) : _mm512_setzero_ps();
    vec c62 = resume ? _mm512_loadu_ps(c + (size_t)6 * ldc + 32) : _mm512_setzero_ps();
    vec c70 = resume ? _mm512_loadu_ps(c + (size_t)7 * ldc + 0) : _mm512_setzero_ps();
    vec c71 = resume ? _mm512_loadu_ps(c + (size_t)7 * ldc + 16) : _mm512_setzero_ps();
    vec c72 = resume ? _mm512_loadu_ps(c + (size_t)7 * ldc + 32) : _mm512_setzero_ps();
    for (int p = 0; p < k; ++p) {
        const vec b0 = _mm512_load_ps(b);
        const vec b1 = _mm512_load_ps(b + 16);
        const vec b2 = _mm512_load_ps(b + 32);
        const vec a0 = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c02 = _mm512_fmadd_ps(a0, b2, c02);
        const vec a1 = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c12 = _mm512_fmadd_ps(a1, b2, c12);
        const vec a2 = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c22 = _mm512_fmadd_ps(a2, b2, c22);
        const vec a3 = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
        c32 = _mm512_fmadd_ps(a3, b2, c32);
        const vec a4 = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(a4, b0, c40);
        c41 = _mm512_fmadd_ps(a4, b1, c41);
        c42 = _mm512_fmadd_ps(a4, b2, c42);
        const vec a5 = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(a5, b0, c50);
        c51 = _mm512_fmadd_ps(a5, b1, c51);
        c52 = _mm512_fmadd_ps(a5, b2, c52);
        const vec a6 = _mm512_set1_ps(a[6]);
        c60 = _mm512_fmadd_ps(a6, b0, c60);
        c61 = _mm512_fmadd_ps(a6, b1, c61);
        c62 = _mm512_fmadd_ps(a6, b2, c62);
        const vec a7 = _mm512_set1_ps(a[7]);
        c70 = _mm512_fmadd_ps(a7, b0, c70);
        c71 = _mm512_fmadd_ps(a7, b1, c71);
        c72 = _mm512_fmadd_ps(a7, b2, c72);
        a += 8;
        b += 48;
    }
    _mm512_storeu_ps(c + (size_t)0 * ldc + 0, c00);
    _mm512_storeu_ps(c + (size_t)0 * ldc + 16, c01);
    _mm512_storeu_ps(c + (size_t)0 * ldc + 32, c02);
    _mm512_storeu_ps(c + (size_t)1 * ldc + 0, c10);
    _mm512_storeu_ps(c + (size_t)1 * ldc + 16, c11);
    _mm512_storeu_ps(c + (size_t)1 * ldc + 32, c12);
    _mm512_storeu_ps(c + (size_t)2 * ldc + 0, c20);
    _mm512_storeu_ps(c + (size_t)2 * ldc + 16, c21);
    _mm512_storeu_ps(c + (size_t)2 * ldc + 32, c22);
    _mm512_storeu_ps(c + (size_t)3 * ldc + 0, c30);
    _mm512_storeu_ps(c + (size_t)3 * ldc + 16, c31);
    _mm512_storeu_ps(c + (size_t)3 * ldc + 32, c32);
    _mm512_storeu_ps(c + (size_t)4 * ldc + 0, c40);
    _mm512_storeu_ps(c + (size_t)4 * ldc + 16, c41);
    _mm512_storeu_ps(c + (size_t)4 * ldc + 32, c42);
    _mm512_storeu_ps(c + (size_t)5 * ldc + 0, c50);
    _mm512_storeu_ps(c + (size_t)5 * ldc + 16, c51);
    _mm512_storeu_ps(c + (size_t)5 * ldc + 32, c52);
    _mm512_storeu_ps(c + (size_t)6 * ldc + 0, c60);
    _mm512_storeu_ps(c + (size_t)6 * ldc + 16, c61);
    _mm512_storeu_ps(c + (size_t)6 * ldc + 32, c62);
    _mm512_storeu_ps(c + (size_t)7 * ldc + 0, c70);
    _mm512_storeu_ps(c + (size_t)7 * ldc + 16, c71);
    _mm512_storeu_ps(c + (size_t)7 * ldc + 32, c72);
}
