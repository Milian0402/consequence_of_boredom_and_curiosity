    for (int j = 0; j < n; j += 16) {
        svbool_t cols = svwhilelt_b32_s32(j,n);
        if (m<=32) compact_strip(2,m,k,depth,ap,b+j,ldb,c+j,ldc,cols);
        else if (m<=48) compact_strip(3,m,k,depth,ap,b+j,ldb,c+j,ldc,cols);
        else compact_strip(4,m,k,depth,ap,b+j,ldb,c+j,ldc,cols);
    }
