/* Generated from Sandia's BSD-2-licensed Hopcroft-Kerr table.
 * Copyright (c) 2014, Sandia Corporation. See LICENSE.fast-matmul. */
    /* Product 1. */
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b0 = b[0] + (size_t)i * ldb;
        const float *b2 = b[2] + (size_t)i * ldb;
        const float *b3 = b[3] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = b0[j] + b2[j] - b3[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, a[3], lda, y, nh, c[3], ldc);
    /* Product 2. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a3 = a[3] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] - a3[j];
    }
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b0 = b[0] + (size_t)i * ldb;
        const float *b1 = b[1] + (size_t)i * ldb;
        const float *b6 = b[6] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = b0[j] - b1[j] + b6[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, c[0], ldc);
    /* Product 3. */
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b1 = b[1] + (size_t)i * ldb;
        const float *b4 = b[4] + (size_t)i * ldb;
        const float *b5 = b[5] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = -b1[j] + b4[j] + b5[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, a[1], lda, y, nh, c[1], ldc);
    /* Product 4. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a1 = a[1] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = a0[j] + a1[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[1], ldb, product + 0 * pc, nh);
    /* Product 5. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a1 = a[1] + (size_t)i * lda;
        const float *a3 = a[3] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = a1[j] + a3[j];
    }
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b1 = b[1] + (size_t)i * ldb;
        const float *b3 = b[3] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = -b1[j] + b3[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product + 1 * pc, nh);
    /* Product 6. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a0 = a[0] + (size_t)i * lda;
        const float *a2 = a[2] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = -a0[j] + a2[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[2], ldb, product + 2 * pc, nh);
    /* Product 8. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a3 = a[3] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = dst[j] + a3[j];
    }
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b2 = b[2] + (size_t)i * ldb;
        const float *b6 = b[6] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = -b2[j] + b6[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product + 3 * pc, nh);
    /* Product 15. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a5 = a[5] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = dst[j] - a5[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[6], ldb, product + 4 * pc, nh);
    /* Product 7. */
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b2 = b[2] + (size_t)i * ldb;
        const float *b7 = b[7] + (size_t)i * ldb;
        const float *b8 = b[8] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = b2[j] + b7[j] + b8[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, a[2], lda, y, nh, c[2], ldc);
    /* Product 9. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a1 = a[1] + (size_t)i * lda;
        const float *a4 = a[4] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = a1[j] - a4[j];
    }
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b3 = b[3] + (size_t)i * ldb;
        const float *b4 = b[4] + (size_t)i * ldb;
        const float *b7 = b[7] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = b3[j] - b4[j] - b7[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, c[4], ldc);
    /* Product 14. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a5 = a[5] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = dst[j] + a5[j];
    }
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b5 = b[5] + (size_t)i * ldb;
        const float *b7 = b[7] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = b5[j] - b7[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, y, nh, product + 5 * pc, nh);
    /* Product 13. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a2 = a[2] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = -dst[j] + a2[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[7], ldb, product + 6 * pc, nh);
    /* Product 10. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a3 = a[3] + (size_t)i * lda;
        const float *a4 = a[4] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = a3[j] + a4[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[3], ldb, product + 7 * pc, nh);
    /* Product 11. */
    for (int i = 0; i < mh; ++i) {
        float *dst = x + (size_t)i * kh;
        const float *a4 = a[4] + (size_t)i * lda;
        const float *a5 = a[5] + (size_t)i * lda;
        for (int j = 0; j < kh; ++j) dst[j] = -a4[j] + a5[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, x, kh, b[5], ldb, product + 8 * pc, nh);
    /* Product 12. */
    for (int i = 0; i < kh; ++i) {
        float *dst = y + (size_t)i * nh;
        const float *b5 = b[5] + (size_t)i * ldb;
        const float *b6 = b[6] + (size_t)i * ldb;
        const float *b8 = b[8] + (size_t)i * ldb;
        for (int j = 0; j < nh; ++j) dst[j] = b5[j] + b6[j] + b8[j];
    }
    cob_sgemm_rowmajor(mh, nh, kh, a[5], lda, y, nh, c[5], ldc);
    hk_reduce(mh, nh, ldc, pc, product, c[0], c[1], c[2], c[3], c[4], c[5]);
