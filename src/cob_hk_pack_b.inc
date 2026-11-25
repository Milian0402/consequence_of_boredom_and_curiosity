/* Right operands for the four HK products whose left operand is an original
   block. Combine the original blocks, round in the HK order, and write the
   seven packed operands for the subsequent one-level Strassen child. */
enum cob_hk_b_slot { COB_B_M2, COB_B_M3, COB_B_M4, COB_B_M1, COB_B_M5, COB_B_M6, COB_B_M7 };

typedef struct {
    const float *first, *second, *third;
    int plus_second, plus_third;
} cob_hk_b_form;

#if defined(COB_HK_PACK_B_TRACE) && COB_HK_PACK_B_TRACE
#define COB_TRACE_HK_PACK_B 1
static _Thread_local int cob_hk_pack_hits, cob_hk_panel_uses, cob_hk_pack_fallbacks;
int rank_hk_packed_forms(int id)
{
    switch (id) {
        case 0: return cob_hk_pack_hits;
        case 1: return cob_hk_panel_uses;
        case 2: return cob_hk_pack_fallbacks;
        default: return -1;
    }
}
#endif

static inline float cob_hk_b_value(cob_hk_b_form f, size_t z)
{
    const float pair = f.plus_second ? f.first[z]+f.second[z] : f.first[z]-f.second[z];
    return f.plus_third ? pair+f.third[z] : pair-f.third[z];
}

static void cob_hk_pack_seven_b(cob_hk_b_form form, int ldb, int depth, int cols,
    int nr, int packed_n, int kc, float *restrict packed)
{
    const size_t pitch = (size_t)depth*packed_n;
    const size_t lower = (size_t)depth*ldb;
    for (int pc = 0; pc < depth; pc += kc) {
        const int count = cob_min_i32(depth-pc,kc);
        for (int j = 0; j < cols; j += nr) {
            const int width = cob_min_i32(cols-j,nr);
            for (int p = 0; p < count; ++p) {
                const size_t z = (size_t)(pc+p)*ldb+j;
                float *dst = packed+(size_t)pc*packed_n+(size_t)j*count+(size_t)p*nr;
                for (int t = 0; t < width; ++t) {
                    const float b11 = cob_hk_b_value(form,z+t);
                    const float b12 = cob_hk_b_value(form,z+cols+t);
                    const float b21 = cob_hk_b_value(form,z+lower+t);
                    const float b22 = cob_hk_b_value(form,z+lower+cols+t);
                    /* Consumer order: M2, M3, M4, M1, M5, M6, M7. */
                    dst[COB_B_M2*pitch+t] = b11;
                    dst[COB_B_M3*pitch+t] = b12-b22;
                    dst[COB_B_M4*pitch+t] = b21-b11;
                    dst[COB_B_M1*pitch+t] = b11+b22;
                    dst[COB_B_M5*pitch+t] = b22;
                    dst[COB_B_M6*pitch+t] = b11+b12;
                    dst[COB_B_M7*pitch+t] = b21+b22;
                }
                if (width < nr)
                    for (int q = 0; q < 7; ++q)
                        memset(dst+(size_t)q*pitch+width,0,(size_t)(nr-width)*sizeof(float));
            }
        }
    }
}

/* This HK child has each dimension bounded at 1280. Only an odd fringe
   needs source entries beyond the four packed quadrants. Preserve the
   incumbent's dot-product and update ordering, including 256-deep columns. */
static void cob_hk_b_edges(int m, int n, int k, const float* a, int lda,
    cob_hk_b_form f, int ldb, float* c, int ldc)
{
    const int me=m-m%2, ne=n-n%2;
    float edge[1280];
    if (k%2) {
        const size_t last=(size_t)(k-1)*ldb;
        for (int j=0; j<ne; ++j) edge[j]=cob_hk_b_value(f,last+j);
        for (int i=0; i<me; ++i) {
            const float left=a[(size_t)i*lda+k-1];
            float *dst=c+(size_t)i*ldc;
            for (int j=0; j<ne; ++j) dst[j]+=left*edge[j];
        }
    }
    if (n%2) {
        for (int pc=0; pc<k; pc+=COB_FUSED_KC) {
            const int kb=cob_min_i32(k-pc,COB_FUSED_KC);
            for (int p=0; p<kb; ++p) edge[p]=cob_hk_b_value(f,(size_t)(pc+p)*ldb+ne);
            for (int i=0; i<me; ++i) {
                const float *left=a+(size_t)i*lda+pc;
                float *dst=c+(size_t)i*ldc+ne;
                float sum=pc ? *dst : 0;
                for (int p=0; p<kb; ++p) sum+=left[p]*edge[p];
                *dst=sum;
            }
        }
    }
    if (m%2) {
        float *dst=c+(size_t)me*ldc;
        memset(dst,0,(size_t)n*sizeof(float));
        for (int p=0; p<k; ++p) {
            const float left=a[(size_t)me*lda+p];
            for (int j=0; j<n; ++j) edge[j]=cob_hk_b_value(f,(size_t)p*ldb+j);
            for (int j=0; j<n; ++j) dst[j]+=left*edge[j];
        }
    }
}
