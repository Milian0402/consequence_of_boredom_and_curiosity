    for (int j = 0; j < n; j += 64) {
        const int w = n-j < 64 ? n-j : 64;
        for (int i = 0; i < m; i += 16) {
            const int h = m-i < 16 ? m-i : 16;
            svzero_za();
            if (w == 64) cob_sme_short_k_tile(k, ap + i*depth, b + j, ldb, svptrue_c32());
            else cob_sme_short_k_tile(k, ap + i*depth, b+j, ldb, svwhilelt_c32_u64(0,(unsigned)w,4));
            float *dst = c+(size_t)i*ldc+j;
            if (w == 64 && h == 16) store_za(dst, ldc);
            else cob_sme_transfer_short_c(h, w, dst, ldc, 1);
        }
    }
