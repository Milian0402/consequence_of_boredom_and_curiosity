/* Generated from Sandia's BSD-2-licensed Hopcroft-Kerr table.
 * Copyright (c) 2014, Sandia Corporation. See LICENSE.fast-matmul. */
    /* Product 1 of 15. */
#ifdef COB_HK_PACK_RIGHT
    if (!COB_HK_PACK_RIGHT(a[3], b[0], b[2], b[3], 1, 0, c[0], ldc))
#endif
    {
        for (int i = 0; i < kh; ++i) {
            const float *b0 = b[0] + (size_t)i * ldb;
            const float *b2 = b[2] + (size_t)i * ldb;
            const float *b3 = b[3] + (size_t)i * ldb;
            float *dst = y + (size_t)i * nh;
            for (int j = 0; j < nh; ++j) dst[j] = b0[j] + b2[j] - b3[j];
        }
        cob_sgemm_rowmajor(mh, nh, kh, a[3], lda, y, nh, c[0], ldc);
    }
    for (int i = 0; i < mh; ++i) {
        const float *src = c[0] + (size_t)i * ldc;
        float *c3 = c[3] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c3[j] = src[j];
        }
    }
    /* Product 2 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a3 = a[3] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] - a3[j];
    }
    for (int i = 0; i < kh; ++i) {
        const float *b0 = b[0] + (size_t)i * ldb;
        const float *b1 = b[1] + (size_t)i * ldb;
        const float *b6 = b[6] + (size_t)i * ldb;
        float *dst = y + (size_t)i * nh;
        for (int j = 0; j < nh; ++j) dst[j] = b0[j] - b1[j] + b6[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c0 = c[0] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c0[j] += src[j];
        }
    }
    /* Product 3 of 15. */
#ifdef COB_HK_PACK_RIGHT
    if (!COB_HK_PACK_RIGHT(a[1], b[1], b[4], b[5], 0, 0, product, nh))
#endif
    {
        for (int i = 0; i < kh; ++i) {
            const float *b1 = b[1] + (size_t)i * ldb;
            const float *b4 = b[4] + (size_t)i * ldb;
            const float *b5 = b[5] + (size_t)i * ldb;
            float *dst = y + (size_t)i * nh;
            for (int j = 0; j < nh; ++j) dst[j] = b1[j] - b4[j] - b5[j];
        }
        cob_sgemm_rowmajor(mh, nh, kh, a[1], lda, y, nh, product, nh);
    }
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c1 = c[1] + (size_t)i * ldc;
        float *c4 = c[4] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c1[j] = - src[j];
            c4[j] = - src[j];
        }
    }
    /* Product 4 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a1 = a[1] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] + a1[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[1], ldb, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c0 = c[0] + (size_t)i * ldc;
        float *c1 = c[1] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c0[j] += src[j];
            c1[j] += src[j];
        }
    }
    /* Product 5 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a1 = a[1] + (size_t)i * lda;
        const float *a3 = a[3] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a1[j] + a3[j];
    }
    for (int i = 0; i < kh; ++i) {
        const float *b1 = b[1] + (size_t)i * ldb;
        const float *b3 = b[3] + (size_t)i * ldb;
        float *dst = y + (size_t)i * nh;
        for (int j = 0; j < nh; ++j) dst[j] = b1[j] - b3[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c0 = c[0] + (size_t)i * ldc;
        float *c4 = c[4] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c0[j] -= src[j];
            c4[j] += src[j];
        }
    }
    /* Product 6 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a2 = a[2] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] - a2[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[2], ldb, c[2], ldc);
    for (int i = 0; i < mh; ++i) {
        const float *src = c[2] + (size_t)i * ldc;
        float *c0 = c[0] + (size_t)i * ldc;
        float *c3 = c[3] + (size_t)i * ldc;
        float *c5 = c[5] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c0[j] -= src[j];
            c3[j] -= src[j];
            c5[j] = src[j];
        }
    }
    /* Product 7 of 15. */
#ifdef COB_HK_PACK_RIGHT
    if (!COB_HK_PACK_RIGHT(a[2], b[2], b[7], b[8], 1, 1, product, nh))
#endif
    {
        for (int i = 0; i < kh; ++i) {
            const float *b2 = b[2] + (size_t)i * ldb;
            const float *b7 = b[7] + (size_t)i * ldb;
            const float *b8 = b[8] + (size_t)i * ldb;
            float *dst = y + (size_t)i * nh;
            for (int j = 0; j < nh; ++j) dst[j] = b2[j] + b7[j] + b8[j];
        }
        cob_sgemm_rowmajor(mh, nh, kh, a[2], lda, y, nh, product, nh);
    }
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c2 = c[2] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c2[j] += src[j];
        }
    }
    /* Product 8 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a2 = a[2] + (size_t)i * lda;
        const float *a3 = a[3] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] - a2[j] - a3[j];
    }
    for (int i = 0; i < kh; ++i) {
        const float *b2 = b[2] + (size_t)i * ldb;
        const float *b6 = b[6] + (size_t)i * ldb;
        float *dst = y + (size_t)i * nh;
        for (int j = 0; j < nh; ++j) dst[j] = b2[j] - b6[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c0 = c[0] + (size_t)i * ldc;
        float *c3 = c[3] + (size_t)i * ldc;
        float *c5 = c[5] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c0[j] += src[j];
            c3[j] += src[j];
            c5[j] -= src[j];
        }
    }
    /* Product 9 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a1 = a[1] + (size_t)i * lda;
        const float *a4 = a[4] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a1[j] - a4[j];
    }
    for (int i = 0; i < kh; ++i) {
        const float *b3 = b[3] + (size_t)i * ldb;
        const float *b4 = b[4] + (size_t)i * ldb;
        const float *b7 = b[7] + (size_t)i * ldb;
        float *dst = y + (size_t)i * nh;
        for (int j = 0; j < nh; ++j) dst[j] = b3[j] - b4[j] - b7[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c4 = c[4] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c4[j] += src[j];
        }
    }
    /* Product 10 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a3 = a[3] + (size_t)i * lda;
        const float *a4 = a[4] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a3[j] + a4[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[3], ldb, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c3 = c[3] + (size_t)i * ldc;
        float *c4 = c[4] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c3[j] += src[j];
            c4[j] += src[j];
        }
    }
    /* Product 11 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a4 = a[4] + (size_t)i * lda;
        const float *a5 = a[5] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a4[j] - a5[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[5], ldb, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c1 = c[1] + (size_t)i * ldc;
        float *c2 = c[2] + (size_t)i * ldc;
        float *c4 = c[4] + (size_t)i * ldc;
        float *c5 = c[5] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c1[j] -= src[j];
            c2[j] += src[j];
            c4[j] -= src[j];
            c5[j] += src[j];
        }
    }
    /* Product 12 of 15. */
#ifdef COB_HK_PACK_RIGHT
    if (!COB_HK_PACK_RIGHT(a[5], b[5], b[6], b[8], 1, 1, product, nh))
#endif
    {
        for (int i = 0; i < kh; ++i) {
            const float *b5 = b[5] + (size_t)i * ldb;
            const float *b6 = b[6] + (size_t)i * ldb;
            const float *b8 = b[8] + (size_t)i * ldb;
            float *dst = y + (size_t)i * nh;
            for (int j = 0; j < nh; ++j) dst[j] = b5[j] + b6[j] + b8[j];
        }
        cob_sgemm_rowmajor(mh, nh, kh, a[5], lda, y, nh, product, nh);
    }
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c5 = c[5] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c5[j] += src[j];
        }
    }
    /* Product 13 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a1 = a[1] + (size_t)i * lda;
        const float *a2 = a[2] + (size_t)i * lda;
        const float *a4 = a[4] + (size_t)i * lda;
        const float *a5 = a[5] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a1[j] - a2[j] - a4[j] + a5[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[7], ldb, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c1 = c[1] + (size_t)i * ldc;
        float *c2 = c[2] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c1[j] -= src[j];
            c2[j] += src[j];
        }
    }
    /* Product 14 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a1 = a[1] + (size_t)i * lda;
        const float *a4 = a[4] + (size_t)i * lda;
        const float *a5 = a[5] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a1[j] - a4[j] + a5[j];
    }
    for (int i = 0; i < kh; ++i) {
        const float *b5 = b[5] + (size_t)i * ldb;
        const float *b7 = b[7] + (size_t)i * ldb;
        float *dst = y + (size_t)i * nh;
        for (int j = 0; j < nh; ++j) dst[j] = b5[j] - b7[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c1 = c[1] + (size_t)i * ldc;
        float *c2 = c[2] + (size_t)i * ldc;
        float *c4 = c[4] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c1[j] -= src[j];
            c2[j] += src[j];
            c4[j] -= src[j];
        }
    }
    /* Product 15 of 15. */
    for (int i = 0; i < mh; ++i) {
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a2 = a[2] + (size_t)i * lda;
        const float *a3 = a[3] + (size_t)i * lda;
        const float *a5 = a[5] + (size_t)i * lda;
        float *dst = x + (size_t)i * kh;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] - a2[j] - a3[j] + a5[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[6], ldb, product, nh);
    for (int i = 0; i < mh; ++i) {
        const float *src = product + (size_t)i * nh;
        float *c3 = c[3] + (size_t)i * ldc;
        float *c5 = c[5] + (size_t)i * ldc;
        for (int j = 0; j < nh; ++j) {
            c3[j] += src[j];
            c5[j] -= src[j];
        }
    }
