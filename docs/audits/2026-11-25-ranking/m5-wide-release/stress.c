#define main pair_main
#include "common-bench.c"
#undef main
static uint32_t rand32(uint32_t *s) { *s=1664525u**s+1013904223u;return *s; }
static float draw(uint32_t *s) {return (float)((int)(rand32(s)>>8)-8388608)/8388608.0f;}
int main(int argc,char **argv) {
    if(argc!=3)fail("ON.so OFF.so");gemm_fn on=load(argv[1]),off=load(argv[2]);
    int shape[][4]={{24,1024,96,0},{25,1087,97,3},{31,4033,199,7},{39,2304,160,0},{46,2983,353,13},{55,1536,384,0}};
    uint32_t seeds[]={318741563,1557601921,1987668599};
    const char *names[]={"normalized","cancellation","reciprocal","correlated4","correlated16","correlated32"};
    puts("m,n,k,pad,seed,fixture,cells,max_on,max_off,max_nearest,min_excess_on,max_excess_on,max_excess_off,over_on,over_off,over_nearest,avoidable_on,avoidable_off,newly_over_on,newly_under_on,unequal_outputs,inputs_and_guards");
    mtmm_config cfg={.num_threads=1};
    for(size_t si=0;si<sizeof(shape)/sizeof(*shape);++si)for(int q=0;q<3;++q)for(int mode=0;mode<6;++mode) {
        int m=shape[si][0],n=shape[si][1],k=shape[si][2],pad=shape[si][3],lda=k+pad,ld=n+pad;
        float *a=matrix(m,lda),*b=matrix(k,ld),*ao=matrix(m,lda),*bo=matrix(k,ld),*c=matrix(m,ld),*d=matrix(m,ld);
        uint32_t s=seeds[q];size_t cells=(size_t)m*n;
        fill(a,m,k,lda,&s);fill(b,k,n,ld,&s);
        for(int i=0;i<m;++i)for(int p=0;p<k;++p){
            float v=draw(&s);
            if(mode==1)v=(1.0f+v/32768.0f)*((p&1)?-1.0f:1.0f);
            if(mode==2)v=ldexpf(v,p%21-10);
            if(mode>=3)v*=mode==3?4.0f:mode==4?16.0f:32.0f;
            a[(size_t)i*lda+p]=v;
        }
        for(int p=0;p<k;++p)for(int j=0;j<n;++j){
            float v=draw(&s);
            if(mode==1)v=1.0f+v/32768.0f;
            if(mode==2)v=ldexpf(v,-(p%21-10));
            if(mode>=3)v=a[(size_t)(j%m)*lda+p];
            b[(size_t)p*ld+j]=v;
        }
        size_t sa=(size_t)m*lda*sizeof(float),sb=(size_t)k*ld*sizeof(float);
        memcpy(ao,a,sa);memcpy(bo,b,sb);prepare_output(c,m,n,ld);prepare_output(d,m,n,ld);
        double *r=calloc(cells,sizeof(*r));if(!r)fail("reference");
        for(int i=0;i<m;++i)for(int p=0;p<k;++p) {
            double x=a[(size_t)i*lda+p];for(int j=0;j<n;++j)r[(size_t)i*n+j]+=x*b[(size_t)p*ld+j];
        }
        on(m,n,k,a,lda,b,ld,c,ld,&cfg);
        if(memcmp(a,ao,sa)||memcmp(b,bo,sb))fail("ON changed input");
        off(m,n,k,a,lda,b,ld,d,ld,&cfg);
        if(memcmp(a,ao,sa)||memcmp(b,bo,sb))fail("OFF changed input");
        check_guard(a,(size_t)m*lda);check_guard(b,(size_t)k*ld);check_output_padding(c,m,n,ld);check_output_padding(d,m,n,ld);
        double x=0,y=0,z=0,lo=INFINITY,hi=0,ho=0;
        size_t f[8]={0};
        for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
            double v=r[(size_t)i*n+j],co=c[(size_t)i*ld+j],di=d[(size_t)i*ld+j];
            double dx=fabs(co-v),dy=fabs(di-v),dz=fabs((double)(float)v-v);
            if(!isfinite(dx)||!isfinite(dy)||!isfinite(dz))fail("non-finite");
            if(dx+1e-12<dz||dy+1e-12<dz)fail("rounding floor");
            x=fmax(x,dx);y=fmax(y,dy);z=fmax(z,dz);lo=fmin(lo,dx-dz);hi=fmax(hi,dx-dz);ho=fmax(ho,dy-dz);
            int bx=dx>.002,by=dy>.002,bz=dz>.002;
            f[0]+=bx;f[1]+=by;f[2]+=bz;f[3]+=bx&&!bz;f[4]+=by&&!bz;f[5]+=bx&&!by;f[6]+=by&&!bx;f[7]+=co!=di;
        }
        printf("%d,%d,%d,%d,%u,%s,%zu,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g",m,n,k,pad,seeds[q],names[mode],cells,x,y,z,lo,hi,ho);
        for(int i=0;i<8;++i)printf(",%zu",f[i]);puts(",1");
        free(a);free(ao);free(b);free(bo);free(c);free(d);free(r);
    }
}
