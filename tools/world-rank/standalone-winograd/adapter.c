/* One-shot Winograd with a total budget of two 7-product algorithm levels. */
#define _POSIX_C_SOURCE 200112L
#include "vendor.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define API __attribute__((visibility("default")))

extern void cob_winograd_full_public(int,int,int,const float*,int,const float*,int,float*,int);
extern void cob_winograd_one_public(int,int,int,const float*,int,const float*,int,float*,int);
extern void cob_winograd_zero_public(int,int,int,const float*,int,const float*,int,float*,int);
#ifndef COB_WINOGRAD_OUTER_DEPTH
#define COB_WINOGRAD_OUTER_DEPTH 2
#endif
#if COB_WINOGRAD_OUTER_DEPTH == 1
/* The existing level-2 min/2 guard now exceeds every valid int dimension. */
#define RANK_HYBRID_LEVEL2_MIN 4294967296.0
#elif COB_WINOGRAD_OUTER_DEPTH != 2
#error "COB_WINOGRAD_OUTER_DEPTH must be 1 or 2"
#endif

#ifdef COB_WINOGRAD_TRACE
struct winograd_counts {
    int m,n,k,total,core,fringe,direct,depth;
    int hk_entries,ordinary_fused,hk_fused,hk_children;
};
static _Thread_local struct winograd_counts counts;
extern int cob_winograd_full_route(int);
extern int cob_winograd_one_route(int);
static void count_leaf(int m,int n,int k,int (*route)(int))
{
    ++counts.total;
    if(m==counts.m && n==counts.n && k==counts.k) ++counts.direct;
    else if(k==counts.k) ++counts.fringe;
    else ++counts.core;
    if(route) {
        const int nested=route(0);
        counts.hk_entries+=(nested!=0);
        counts.ordinary_fused+=route(3);
        counts.hk_fused+=route(1);
        counts.hk_children+=nested;
    }
}
#define COUNTED_LEAF(NAME,CALLEE,ROUTE) \
static void NAME(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc) \
{ \
    CALLEE(m,n,k,a,lda,b,ldb,c,ldc); \
    count_leaf(m,n,k,ROUTE); \
}
COUNTED_LEAF(full_leaf,cob_winograd_full_public,cob_winograd_full_route)
COUNTED_LEAF(one_leaf,cob_winograd_one_public,cob_winograd_one_route)
COUNTED_LEAF(zero_leaf,cob_winograd_zero_public,NULL)
#undef COUNTED_LEAF
API int rank_winograd_leaf_count(int id)
{
    switch(id) {
        case 0:return counts.total;
        case 1:return counts.core;
        case 2:return counts.fringe;
        case 3:return counts.depth;
        case 4:return counts.direct;
        case 5:return counts.hk_entries;
        case 6:return counts.ordinary_fused;
        case 7:return counts.hk_fused;
        case 8:return counts.hk_children;
        default:return -1;
    }
}
#else
#define full_leaf cob_winograd_full_public
#define one_leaf cob_winograd_one_public
#define zero_leaf cob_winograd_zero_public
#endif

/* Select after scratch allocation succeeds, so fallback from two levels to
   one gets the same bounded leaf as any other one-level multiplication. */
static rank_sgemm_fn standalone_leaf_for_depth(int depth)
{
#ifdef COB_WINOGRAD_TRACE
    counts.depth=depth;
#endif
    return depth==2?zero_leaf:one_leaf;
}
static rank_sgemm_fn standalone_leaf_bridge=full_leaf;
#define rank_bridge standalone_leaf_bridge
#define cob_sgemm_rowmajor full_leaf
#define rank_hybrid_next standalone_winograd
#include "hybrid_next.c"
#undef rank_hybrid_next
#undef cob_sgemm_rowmajor
#undef rank_bridge

API int rank_vendor_init(void) { return 0; }

API void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,
    const float* b,int ldb,float* c,int ldc)
{
#ifdef COB_WINOGRAD_TRACE
    counts=(struct winograd_counts){.m=m,.n=n,.k=k};
#endif
    standalone_winograd(m,n,k,a,lda,b,ldb,c,ldc);
}
