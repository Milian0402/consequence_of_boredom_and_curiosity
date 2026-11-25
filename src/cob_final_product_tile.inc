/* The final panel resumes the stored FP32 sum and executes the stock number
   and order of FMAs. Keep accumulator names constant even at writeback so
   GCC can keep the full tile in registers through the depth loop. */
#if COB_FINAL_MR == 4
#define COB_FINAL_ROWS(f) f(0) f(1) f(2) f(3)
#elif COB_FINAL_MR == 6
#define COB_FINAL_ROWS(f) f(0) f(1) f(2) f(3) f(4) f(5)
#elif COB_FINAL_MR == 8
#define COB_FINAL_ROWS(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7)
#elif COB_FINAL_MR == 12
#define COB_FINAL_ROWS(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8) f(9) f(10) f(11)
#else
#error "Unrecognized native register tile"
#endif
#if COB_FINAL_NR == 2 * COB_FINAL_LANES
#define COB_FINAL_GROUPS(f,r) f(r,0) f(r,1)
#elif COB_FINAL_NR == 3 * COB_FINAL_LANES
#define COB_FINAL_GROUPS(f,r) f(r,0) f(r,1) f(r,2)
#else
#error "Unrecognized native register tile"
#endif

COB_FINAL_TARGET static void COB_FINAL_NAME(int k, const float* a, const float* b,
    const float* prior, int ldp, int resume, const cob_final_output* q, size_t offset)
{
#define COB_FINAL_INIT(r,v) COB_FINAL_VEC c##r##_##v = resume ? \
    COB_FINAL_LOAD(prior + (size_t)(r)*ldp + (v)*COB_FINAL_LANES) : COB_FINAL_ZERO();
#define COB_FINAL_INIT_ROW(r) COB_FINAL_GROUPS(COB_FINAL_INIT,r)
    COB_FINAL_ROWS(COB_FINAL_INIT_ROW)
#undef COB_FINAL_INIT_ROW
#undef COB_FINAL_INIT
    for (int p=0; p<k; ++p) {
#define COB_FINAL_LOAD_B(unused,v) const COB_FINAL_VEC b##v = COB_FINAL_LOAD(b + (v)*COB_FINAL_LANES);
        COB_FINAL_GROUPS(COB_FINAL_LOAD_B,0)
#undef COB_FINAL_LOAD_B
#define COB_FINAL_FMA_GROUP(r,v) c##r##_##v = COB_FINAL_FMA(av,b##v,c##r##_##v);
#define COB_FINAL_FMA_ROW(r) do { const COB_FINAL_VEC av=COB_FINAL_SPLAT(a[r]); \
    COB_FINAL_GROUPS(COB_FINAL_FMA_GROUP,r) } while (0);
        COB_FINAL_ROWS(COB_FINAL_FMA_ROW)
#undef COB_FINAL_FMA_ROW
#undef COB_FINAL_FMA_GROUP
        a+=COB_FINAL_MR;
        b+=COB_FINAL_NR;
    }
    const int ldc=q->ldc;
    if (q->operation==COB_FINAL_M1) {
#define COB_FINAL_M1_GROUP(r,v) do { \
        const size_t z=offset+(size_t)(r)*ldc+(v)*COB_FINAL_LANES; \
        const COB_FINAL_VEC m4=COB_FINAL_LOAD(q->c11+z); \
        const COB_FINAL_VEC m3=COB_FINAL_LOAD(q->c12+z); \
        const COB_FINAL_VEC m2=COB_FINAL_LOAD(q->c21+z); \
        COB_FINAL_STORE(q->c11+z, c##r##_##v + m4); \
        COB_FINAL_STORE(q->c21+z, m2 + m4); \
        COB_FINAL_STORE(q->c22+z, (c##r##_##v - m2) + m3); \
    } while (0);
#define COB_FINAL_M1_ROW(r) COB_FINAL_GROUPS(COB_FINAL_M1_GROUP,r)
        COB_FINAL_ROWS(COB_FINAL_M1_ROW)
#undef COB_FINAL_M1_ROW
#undef COB_FINAL_M1_GROUP
    } else if (q->operation==COB_FINAL_M5) {
#define COB_FINAL_M5_GROUP(r,v) do { \
        const size_t z=offset+(size_t)(r)*ldc+(v)*COB_FINAL_LANES; \
        COB_FINAL_STORE(q->c11+z, COB_FINAL_LOAD(q->c11+z) - c##r##_##v); \
        COB_FINAL_STORE(q->c12+z, COB_FINAL_LOAD(q->c12+z) + c##r##_##v); \
    } while (0);
#define COB_FINAL_M5_ROW(r) COB_FINAL_GROUPS(COB_FINAL_M5_GROUP,r)
        COB_FINAL_ROWS(COB_FINAL_M5_ROW)
#undef COB_FINAL_M5_ROW
#undef COB_FINAL_M5_GROUP
    } else {
        float* dst=q->operation==COB_FINAL_M6 ? q->c22 : q->c11;
#define COB_FINAL_ADD_GROUP(r,v) do { \
        float* z=dst+offset+(size_t)(r)*ldc+(v)*COB_FINAL_LANES; \
        COB_FINAL_STORE(z, COB_FINAL_LOAD(z) + c##r##_##v); \
    } while (0);
#define COB_FINAL_ADD_ROW(r) COB_FINAL_GROUPS(COB_FINAL_ADD_GROUP,r)
        COB_FINAL_ROWS(COB_FINAL_ADD_ROW)
#undef COB_FINAL_ADD_ROW
#undef COB_FINAL_ADD_GROUP
    }
}
#undef COB_FINAL_ROWS
#undef COB_FINAL_GROUPS
#undef COB_FINAL_NAME
#undef COB_FINAL_MR
#undef COB_FINAL_NR
