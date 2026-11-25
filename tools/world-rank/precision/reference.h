/* Share the signed FGEMM oracle, timing clocks, logical SHA256 and parent checks. */
#define generate_reference signed_reference
#define verify verify_signed
#include "../ragged-hk/golden.h"
#undef generate_reference
#undef verify

static void generate_reference(struct view a,struct view b,struct view c,void *lib)
{
    signed_reference(a,b,c,lib);
    const char *path=getenv("PRECISION_WRITE_WEIGHT");
    if(!path || !*path) fatal("missing weight oracle output");
    float *abs_a=allocate((size_t)a.rows*a.cols,sizeof(float));
    float *abs_b=allocate((size_t)b.rows*b.cols,sizeof(float));
    for(int i=0;i<a.rows;i++) for(int j=0;j<a.cols;j++)
        abs_a[(size_t)i*a.cols+j]=fabsf(a.data[(size_t)i*a.ld+j]);
    for(int i=0;i<b.rows;i++) for(int j=0;j<b.cols;j++)
        abs_b[(size_t)i*b.cols+j]=fabsf(b.data[(size_t)i*b.ld+j]);
    write_reference((shape){c.rows,c.cols,a.cols,0},abs_a,abs_b,path,lib);
    FILE *f=fopen(path,"rb");
    if(!f) fatal("weight oracle reopen");
    double worst=0;
    for(int s=0;s<23;s++) {
        int i=(int)(((size_t)123457*s+31)%c.rows),j=(int)(((size_t)456793*s+47)%c.cols);
        double sum=0,value;
        for(int p=0;p<a.cols;p++) sum+=(double)abs_a[(size_t)i*a.cols+p]*abs_b[(size_t)p*b.cols+j];
        if(fseek(f,(long)(((size_t)i*c.cols+j)*sizeof(double)),SEEK_SET) || fread(&value,sizeof(value),1,f)!=1)
            fatal("short weight spot check");
        double diff=fabs(value-sum);
        if(!isfinite(diff) || diff>1e-9) fatal("weight oracle disagrees with scalar FP64");
        if(diff>worst) worst=diff;
    }
    fclose(f); free(abs_a); free(abs_b);
    printf("WEIGHT,23,%.12g\n",worst);
}

static double verify(struct view a,struct view b,struct view c,size_t *guards)
{
    double absolute=verify_signed(a,b,c,guards);
    FILE *ref=fopen(getenv("RAGGED_REFERENCE"),"rb");
    const char *weight_path=getenv("PRECISION_WEIGHT");
    FILE *weight=weight_path?fopen(weight_path,"rb"):NULL;
    if(!ref || !weight) fatal("missing signed or magnitude oracle");
    double *p=allocate(c.cols,sizeof(double)),*s=allocate(c.cols,sizeof(double));
    size_t nonfinite=0,over_abs=0,over_gamma=0,zero_weight=0;
    double max_rho=0,gamma=a.cols/(16777216.0-a.cols);
    for(int i=0;i<c.rows;i++) {
        if(fread(p,sizeof(double),c.cols,ref)!=(size_t)c.cols || fread(s,sizeof(double),c.cols,weight)!=(size_t)c.cols)
            fatal("incomplete signed/magnitude oracle");
        for(int j=0;j<c.cols;j++) {
            double value=(double)c.data[(size_t)i*c.ld+j];
            if(!isfinite(p[j]) || !isfinite(s[j]) || s[j]<0) fatal("invalid signed/magnitude oracle contents");
            if(!isfinite(value)){++nonfinite; continue;}
            double delta=fabs(value-p[j]);
            const double rho=s[j]>0?delta/s[j]:(delta==0?0:INFINITY);
            if(s[j]==0) ++zero_weight;
            if(delta>.002) ++over_abs;
            if(delta>gamma*s[j]) ++over_gamma;
            if(rho>max_rho) max_rho=rho;
        }
    }
    if(fgetc(weight)!=EOF || ferror(weight)) fatal("weight oracle overlength or read failure");
    free(p);free(s);fclose(ref);fclose(weight);
    printf("ACCURACY,%.17g,%.17g,%zu,%zu,%zu,%zu\n",gamma,max_rho,over_abs,over_gamma,nonfinite,zero_weight);
    return absolute;
}

static void report_precision_route(void *lib,const char *stage)
{
    if(!lib) return;
    int (*route)(void)=NULL;
    void *entry=dlsym(lib,"precision_last_route");
    memcpy(&route,&entry,sizeof(route));
    if(!route) return;
    const char *(*reason)(void)=NULL, *(*implementation)(int)=NULL;
    int (*native)(int)=NULL, (*depth)(int)=NULL;
    entry=symbol(lib,"precision_last_reason"); memcpy(&reason,&entry,sizeof(reason));
    entry=symbol(lib,"precision_last_plan_k"); memcpy(&depth,&entry,sizeof(depth));
    entry=symbol(lib,"precision_last_plan_impl"); memcpy(&implementation,&entry,sizeof(implementation));
    entry=symbol(lib,"precision_last_plan_native"); memcpy(&native,&entry,sizeof(native));
    int plans=getter(lib,"precision_last_plan_count");
    if(plans<0 || plans>3) fatal("invalid mixed operation count");
    printf("MIXED,%s,%d,%d,%d,%s\n",stage,route(),getter(lib,"precision_last_products"),plans,reason());
    for(int i=0;i<plans;i++) printf("PLAN,%s,%d,%d,%d,%s\n",stage,i,depth(i),native(i),implementation(i));
}
