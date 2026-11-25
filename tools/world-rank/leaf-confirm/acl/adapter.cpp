#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Scheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include <cstdio>
#include <cstdlib>
#ifdef ACL_TRACE
#include <cstdint>
#include <sys/prctl.h>
extern "C" int rank_acl_bind();
extern "C" int rank_acl_num_routes();
extern "C" const char *rank_acl_name(int);
extern "C" uint64_t rank_acl_count(int);
#endif

using namespace arm_compute;

static void view(Tensor &tensor, int rows, int columns, int leading, const float *data)
{
    const size_t row_bytes = static_cast<size_t>(leading) * sizeof(float);
    const size_t extent = static_cast<size_t>(rows - 1) * row_bytes + columns * sizeof(float);
    TensorInfo info;
    info.init(TensorShape(columns, rows), 1, DataType::F32, Strides(sizeof(float), row_bytes), 0, extent);
    tensor.allocator()->init(info);
    if (!tensor.allocator()->import_memory(const_cast<float *>(data))) {
        std::fputs("Arm Compute Library rejected the caller's tensor storage\n", stderr);
        std::abort();
    }
}

extern "C" int rank_vendor_init()
{
#ifdef ACL_TRACE
    if (rank_acl_bind() != 0) return 2;
#endif
    Scheduler::set(Scheduler::Type::ST);
    std::fprintf(stderr, "Arm Compute Library 53.3.2, ST scheduler, threads=%u, fast_math=0, F32\n", Scheduler::get().num_threads());
    return Scheduler::get().num_threads() != 1;
}

extern "C" void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                                  const float *b, int ldb, float *c, int ldc)
{
    Tensor av, bv, cv;
    view(av, m, k, lda, a);
    view(bv, k, n, ldb, b);
    view(cv, m, n, ldc, c);
    NEGEMM op;
    op.configure(&av, &bv, nullptr, &cv, 1.f, 0.f);
#ifdef ACL_TRACE
    uint64_t before[128];
    const int known = rank_acl_num_routes();
    if (known > 128) std::abort();
    for (int i = 0; i < known; ++i) before[i] = rank_acl_count(i);
#endif
    op.run();
#ifdef ACL_TRACE
    const long vl = prctl(PR_SVE_GET_VL);
    const int bits = vl < 0 ? 0 : int(vl & PR_SVE_VL_LEN_MASK) * 8;
    for (int i = 0; i < known; ++i) {
        const uint64_t calls = rank_acl_count(i) - before[i];
        if (calls)
            std::fprintf(stderr, "ARM_PEER_TRACE method=acl m=%d n=%d k=%d lda=%d ldb=%d ldc=%d kernel=%s calls=%llu sve_bits=%d\n",
                         m,n,k,lda,ldb,ldc,rank_acl_name(i),static_cast<unsigned long long>(calls),bits);
    }
#endif
}
