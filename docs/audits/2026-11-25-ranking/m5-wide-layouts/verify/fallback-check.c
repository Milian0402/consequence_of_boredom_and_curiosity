#define main primary_oracle_entry
#include "compact-check.c"
#undef main

int main(int argc, char **argv)
{
    if(argc!=3){fprintf(stderr,"usage: %s BASE_DYLIB FAULT_TRACE_DYLIB\n",argv[0]);return 2;}
    gemm_fn base,fn;library(argv[1],&base);void *h=library(argv[2],&fn);
    counter_fn selected=NULL,rejected=NULL;void (*arm)(void)=NULL;void *p=lookup(h,"compact_observed");
    memcpy(&selected,&p,sizeof(selected));p=lookup(h,"compact_rejected");memcpy(&rejected,&p,sizeof(rejected));
    p=lookup(h,"compact_reject_next");memcpy(&arm,&p,sizeof(arm));
    enum {m=24,n=1025,k=97};size_t count=(size_t)k*n;
    guarded a=buffer((size_t)m*k,1),b=buffer(count,1),c=buffer((size_t)m*n,1);
    uint32_t seed=599888119u;
    for(size_t i=0;i<a.len;++i)a.data[i]=draw(&seed,DYADIC,0,1);
    for(size_t i=0;i<b.len;++i)b.data[i]=draw(&seed,DYADIC,0,0);
    readonly(a);readonly(b);
    puts("case,selected,faults,checked,passed");
    int ok=1;
    for(int replay=0;replay<2;++replay){
        for(size_t i=0;i<c.len;++i)c.data[i]=NAN;
        uint64_t old=selected(),fail=rejected();
        if(!replay)arm();
        fn(m,n,k,a.data,k,b.data,n,c.data,n);
        const uint64_t delta=selected()-old,fd=rejected()-fail;
        int good=(delta==(uint64_t)replay && fd==(uint64_t)(1-replay) && guard_ok(c));
        for(int i=0;i<m;++i)for(int j=0;j<n;++j){
            double v=0;
            for(int q=0;q<k;++q)v+=(double)a.data[(size_t)i*k+q]*b.data[(size_t)q*n+j];
            good&=v==(double)c.data[(size_t)i*n+j];
        }
        printf("%s,%"PRIu64",%"PRIu64",%zu,%d\n",replay?"normal-after-failure":"fail-next-workspace",delta,fd,c.len,good);ok&=good;
    }
    const char *invalid[]={"null-a","null-b","null-c","short-lda","short-ldb","short-ldc","zero-m","zero-n","zero-k"};
    for(size_t q=0;q<sizeof(invalid)/sizeof(*invalid);++q){
        for(size_t z=0;z<c.len;++z)c.data[z]=sentinel;
        uint64_t old=selected(),fail=rejected();
        fn(q==6?0:m,q==7?0:n,q==8?0:k,q==0?NULL:a.data,q==3?k-1:k,q==1?NULL:b.data,q==4?n-1:n,q==2?NULL:c.data,q==5?n-1:n);
        int good=selected()==old && rejected()==fail && guard_ok(c);
        for(size_t z=0;z<c.len;++z)good&=c.data[z]==sentinel;
        ok&=good;printf("%s,%"PRIu64",%"PRIu64",%zu,%d\n",invalid[q],selected()-old,rejected()-fail,c.len,good);
    }
    /* Compare incumbent behavior on overlap. The API does not promise a
       mathematical GEMM result for aliased output. Leave backing writable. */
    for(int q=0;q<4;++q){
        const int use_b=q&1,offset=q>=2?3:0;
        guarded x[2]={buffer(count+3,1),buffer(count+3,1)};
        for(size_t z=0;z<x[0].len;++z)x[0].data[z]=draw(&seed,DYADIC,0,1)*.01f;
        memcpy(x[1].data,x[0].data,x[0].len*sizeof(float));
        uint64_t old=selected(),fail=rejected();
        for(int j=0;j<2;++j){
            gemm_fn method=j?fn:base;
            method(m,n,k,use_b?a.data:x[j].data,k,use_b?x[j].data:b.data,n,x[j].data+offset,n);
        }
        int good=selected()==old && rejected()==fail && guard_ok(x[0]) && guard_ok(x[1]) &&
            memcmp(x[0].data,x[1].data,x[0].len*sizeof(float))==0;
        printf("overlap-%c-%d,%"PRIu64",%"PRIu64",%zu,%d\n",use_b?'b':'a',offset,selected()-old,rejected()-fail,x[0].len,good);
        ok&=good;release(x[0]);release(x[1]);
    }
    release(a);release(b);release(c);return ok?0:1;
}
