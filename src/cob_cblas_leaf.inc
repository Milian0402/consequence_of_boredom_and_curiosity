/* External SGEMM leaves are a compile-time choice on native Linux. */
#ifndef COB_ENABLE_OPENBLAS_LEAF
#define COB_ENABLE_OPENBLAS_LEAF 0
#endif
#ifndef COB_ENABLE_MKL_SEQUENTIAL_LEAF
#define COB_ENABLE_MKL_SEQUENTIAL_LEAF 0
#endif
#ifndef COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
#define COB_ENABLE_ARMPL_SEQUENTIAL_LEAF 0
#endif

#if COB_ENABLE_OPENBLAS_LEAF + COB_ENABLE_MKL_SEQUENTIAL_LEAF + COB_ENABLE_ARMPL_SEQUENTIAL_LEAF > 1
#error "Select one CBLAS leaf provider."
#endif

#if COB_ENABLE_OPENBLAS_LEAF
#include <cblas.h>
#define COB_USE_CBLAS_LEAF 1
static int cob_cblas_leaf_is_serial(void)
{
    return openblas_get_num_threads() == 1;
}
#elif COB_ENABLE_MKL_SEQUENTIAL_LEAF || COB_ENABLE_ARMPL_SEQUENTIAL_LEAF
#include <dlfcn.h>
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
#include <mkl.h>
_Static_assert(sizeof(MKL_INT) == sizeof(int), "COB requires the LP64 oneMKL interface.");
#define COB_LEAF_SONAME "libmkl_intel_lp64.so"
#else
#include <armpl.h>
_Static_assert(sizeof(armpl_int_t) == sizeof(int), "COB requires the LP64 ArmPL interface.");
#define COB_LEAF_SONAME "libarmpl_lp64.so"
#endif
#define COB_USE_CBLAS_LEAF 1

/* A process can load more than one BLAS. Verify that both its C and Fortran
   SGEMM bindings reach the selected interface before sending a leaf to it. */
static int cob_cblas_provider_matches(void)
{
    static atomic_int matches;
    int saved = atomic_load_explicit(&matches, memory_order_relaxed);
    if (saved != 0) return saved > 0;

    Dl_info c_info, f_info;
    /* Resolve addresses through the loader. A non-PIE function pointer may
       identify the executable's PLT trampoline instead of the shared library. */
    void* c_gemm = dlsym(RTLD_DEFAULT, "cblas_sgemm");
    void* fortran_gemm = dlsym(RTLD_DEFAULT, "sgemm_");
    int valid = c_gemm && fortran_gemm && dladdr(fortran_gemm, &f_info) &&
        dladdr(c_gemm, &c_info) && c_info.dli_fname &&
        c_info.dli_fbase == f_info.dli_fbase;
    if (valid) {
        const char* slash = strrchr(c_info.dli_fname, '/');
        const char* base = slash ? slash+1 : c_info.dli_fname;
        const size_t size = sizeof(COB_LEAF_SONAME)-1;
        valid = strncmp(base, COB_LEAF_SONAME, size) == 0 &&
            (base[size] == '\0' || base[size] == '.');
    }
    atomic_store_explicit(&matches, valid ? 1 : -1, memory_order_relaxed);
    return valid;
}

static int cob_cblas_leaf_is_serial(void)
{
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
    if (mkl_get_max_threads() != 1 || mkl_domain_get_max_threads(MKL_DOMAIN_BLAS) != 1) return 0;
#endif
    return cob_cblas_provider_matches();
}
#undef COB_LEAF_SONAME
#endif
