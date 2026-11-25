/* Same cropped-view worker for every method, with a reusable serial DGEMM oracle. */
#define RANK_VIEW_REFERENCE 1
#ifndef RANK_VIEW_RECORD_LD
#define RANK_VIEW_RECORD_LD 1
#endif
#include "../views/worker.c"
