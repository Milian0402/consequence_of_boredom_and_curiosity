/* Included after the original guarded-parent and plugin types in views/worker.c. */
#include <openssl/evp.h>

static double thread_clock(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID,&ts)) fatal("caller CPU clock");
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}
static void matrix_digest(struct view v, char hex[65]) {
    EVP_MD_CTX *ctx=EVP_MD_CTX_new();
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned len=0;
    if(!ctx || !EVP_DigestInit_ex(ctx,EVP_sha256(),NULL)) fatal("input SHA256 initialization");
    for(int i=0;i<v.rows;++i)
        if(!EVP_DigestUpdate(ctx,v.data+(size_t)i*v.ld,(size_t)v.cols*sizeof(float))) fatal("input SHA256 update");
    if(!EVP_DigestFinal_ex(ctx,digest,&len) || len!=32) fatal("input SHA256 completion");
    EVP_MD_CTX_free(ctx);
    for(unsigned i=0;i<len;++i) snprintf(hex+2*i,3,"%02x",digest[i]);
}
static int getter(void *lib,const char *key) {
    int (*fn)(void)=NULL; void *s=dlsym(lib,key); memcpy(&fn,&s,sizeof(fn)); return fn?fn():-1;
}
static void report_hashes(struct view a,struct view b,struct view c,double first,double warm,void *lib) {
    char ha[65],hb[65],hc[65];matrix_digest(a,ha);matrix_digest(b,hb);matrix_digest(c,hc);
    printf("RAGGED,%s,%s,%s,%.12g,%.12g\n",ha,hb,hc,first,warm);
    if(lib) { int leaves=getter(lib,"rank_fastmmw_leaves");
        if(leaves>=0) printf("FAST,%d,%d\n",leaves,getter(lib,"rank_fastmmw_padded")); }
}
static void generate_reference(struct view a,struct view b,struct view c,void *lib) {
    const char *path=getenv("RAGGED_WRITE_REFERENCE"); if(!lib || !path || !*path) fatal("missing OpenBLAS oracle");
    /* The frozen oracle converts true FP32 logical values to double before calling serial DGEMM. */
    float *ca=allocate((size_t)a.rows*a.cols,sizeof(float)),*cb=allocate((size_t)b.rows*b.cols,sizeof(float));
    for(int i=0;i<a.rows;++i) memcpy(ca+(size_t)i*a.cols,a.data+(size_t)i*a.ld,(size_t)a.cols*sizeof(float));
    for(int i=0;i<b.rows;++i) memcpy(cb+(size_t)i*b.cols,b.data+(size_t)i*b.ld,(size_t)b.cols*sizeof(float));
    write_reference((shape){c.rows,c.cols,a.cols,0},ca,cb,path,lib); free(ca);free(cb);
    FILE *ref=fopen(path,"rb"); if(!ref) fatal("reference reopen");
    double worst=0;
    for(int s=0;s<23;++s) {
        int i=(int)(((size_t)123457*s+31)%c.rows),j=(int)(((size_t)456793*s+47)%c.cols);
        double sum=0,truth;
        for(int p=0;p<a.cols;++p) sum+=(double)a.data[(size_t)i*a.ld+p]*b.data[(size_t)p*b.ld+j];
        if(fseek(ref,(long)(((size_t)i*c.cols+j)*sizeof(double)),SEEK_SET) || fread(&truth,sizeof truth,1,ref)!=1) fatal("short golden spot check");
        double diff=fabs(sum-truth);if(!isfinite(diff)||diff>1e-9) fatal("DGEMM disagrees with scalar FP64");
        if(diff>worst)worst=diff;
    }
    fclose(ref);char ha[65],hb[65];matrix_digest(a,ha);matrix_digest(b,hb);
    printf("GOLD,%s,%s,%zu,23,%.12g,%d,1\n",ha,hb,(size_t)c.rows*c.cols,worst,sched_getcpu());
}
static double verify(struct view a,struct view b,struct view c,size_t *guards) {
    (void)a;(void)b;
    const char *path=getenv("RAGGED_REFERENCE");if(!path || !*path)fatal("no registered reference input");
    FILE *f=fopen(path,"rb");if(!f)fatal("reference open"); double *row=allocate(c.cols,sizeof(double)),worst=0;
    for(int i=0;i<c.rows;++i) {
        if(fread(row,sizeof(double),c.cols,f)!=(size_t)c.cols)fatal("short reference");
        for(int j=0;j<c.cols;++j){double diff=fabs((double)c.data[(size_t)i*c.ld+j]-row[j]);
            if(!isfinite(diff))worst=INFINITY;else if(diff>worst)worst=diff;}
    }
    if(fgetc(f)!=EOF||ferror(f))fatal("oversized or unreadable reference");fclose(f);free(row);
    *guards=0;
    for(size_t pos=0;pos<c.count;++pos){size_t r=pos/(size_t)c.ld,col=pos%(size_t)c.ld;
        if(r>=(size_t)c.row && r<(size_t)(c.row+c.rows)&&col>=(size_t)c.col&&col<(size_t)(c.col+c.cols))continue;
        if(get_bits(c.start+pos)!=output_guard)fatal("outside C view changed");
        ++*guards;
    }
    return worst;
}
