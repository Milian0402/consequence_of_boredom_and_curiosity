#ifndef COB_RANK_VENDOR_H
#define COB_RANK_VENDOR_H
typedef void (*rank_sgemm_fn)(int,int,int,const float*,int,const float*,int,float*,int);
typedef void (*rank_dgemm_fn)(int,int,int,const double*,int,const double*,int,double*,int);
int rank_vendor_init(void);
void rank_vendor_sgemm(int,int,int,const float*,int,const float*,int,float*,int);
#endif
