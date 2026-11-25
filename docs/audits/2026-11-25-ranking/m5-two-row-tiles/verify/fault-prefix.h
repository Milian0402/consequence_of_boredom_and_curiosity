#include <stdlib.h>
#include <stdint.h>
static unsigned reject_next;
static uint64_t reject_count;
void compact_reject_next(void) { reject_next=1; }
uint64_t compact_rejected(void) { return reject_count; }
static void *checked_aligned_alloc(size_t alignment, size_t bytes)
{
    if(reject_next){reject_next=0; ++reject_count; return NULL;}
    return aligned_alloc(alignment, bytes);
}
#define aligned_alloc checked_aligned_alloc
