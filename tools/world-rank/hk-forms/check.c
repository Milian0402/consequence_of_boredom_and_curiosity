/* Compare three real public entries, and independently check every FP32 result against FP64. */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <dirent.h>
#endif

typedef void (*multiply)(int, int, int, const float *, int, const float *, int, float *, int);
typedef int (*counter)(int);
enum { GUARD = 17 };
static const uint32_t sentinel = 0x7fc4a2b1u;
static uint32_t state;

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "%s\n", message); exit(1); }
}

static void require_single_thread(void) {
#ifdef __APPLE__
    thread_act_array_t threads = NULL;
    mach_msg_type_number_t count = 0;
    require(task_threads(mach_task_self(), &threads, &count) == KERN_SUCCESS, "cannot inspect own threads");
    for (mach_msg_type_number_t i = 0; i < count; ++i)
        mach_port_deallocate(mach_task_self(), threads[i]);
    vm_deallocate(mach_task_self(), (vm_address_t)threads, count * sizeof(*threads));
    require(count == 1, "checker must have one thread");
#elif defined(__linux__)
    DIR *tasks = opendir("/proc/self/task");
    require(tasks != NULL, "cannot inspect own threads");
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(tasks))) if (entry->d_name[0] != '.') ++count;
    closedir(tasks);
    require(count == 1, "checker must have one thread");
#endif
}

typedef struct {
    int rows, cols, ld;
    size_t words;
    float *allocation, *values;
} matrix;

static matrix make(int rows, int cols, int pad) {
    matrix a = { .rows = rows, .cols = cols, .ld = cols + pad };
    a.words = (size_t)rows * a.ld + 2 * GUARD;
    a.allocation = malloc(a.words * sizeof(float));
    require(a.allocation != NULL, "allocation failed in checker");
    a.values = a.allocation + GUARD;
    for (size_t i = 0; i < a.words; ++i) memcpy(a.allocation + i, &sentinel, 4);
    return a;
}

static int guarded(const matrix *a) {
    for (int i = 0; i < GUARD; ++i)
        if (memcmp(a->allocation + i, &sentinel, 4) ||
            memcmp(a->allocation + a->words - 1 - i, &sentinel, 4)) return 0;
    for (int i = 0; i < a->rows; ++i)
        for (int j = a->cols; j < a->ld; ++j)
            if (memcmp(a->values + (size_t)i * a->ld + j, &sentinel, 4)) return 0;
    return 1;
}

static float sample(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return ((int32_t)(state >> 8) - 8388608) * 0x1p-23f;
}

static void populate(float *ar, float *br, int rows, int cols, int k, const char *mode) {
    int cancel = strcmp(mode, "cancel") == 0, zeros = strcmp(mode, "zeros") == 0;
    require(cancel || zeros || strcmp(mode, "random") == 0, "unknown input distribution");
    for (int i = 0; i < rows; ++i) {
        for (int p = 0; p < k; ++p) {
            size_t at = (size_t)i*k + p;
            float v = sample();
            if (cancel && p%2) v = ar[at-1];
            if (zeros && (i+p)%4) v = ((i^p)&1) ? -0.0f : 0.0f;
            ar[at] = v;
        }
    }
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < cols; ++j) {
            size_t at = (size_t)p*cols+j;
            float v = sample();
            if (cancel && p%2) v = -br[at-cols] + 0x1p-11f*v;
            if (zeros && (j+p)%4) v = ((j^p)&1) ? 0.0f : -0.0f;
            br[at] = v;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 14) {
        fprintf(stderr, "check OLD PLAIN TRACED M N K PAD SEED random|cancel|zeros PACKED VIEWS FORMED full|repeat\n");
        return 2;
    }
    int m = atoi(argv[4]), n = atoi(argv[5]), k = atoi(argv[6]), pad = atoi(argv[7]);
    state = (uint32_t)strtoul(argv[8], NULL, 10);
    require(m > 0 && n > 0 && k > 0 && pad >= 0 && m < 5000 && n < 5000 && k < 5000 && pad < 50 && state,
            "unsupported dimensions or seed");
    int full = strcmp(argv[13], "full") == 0;
    require(full || strcmp(argv[13], "repeat") == 0, "invalid reference basis");
    require(!full || (m <= 512 && n <= 512), "use the repeated reference basis for the full-size check");
    int rows = full || m < 17 ? m : 17, cols = full || n < 19 ? n : 19;
    matrix a = make(m,k,pad), b = make(k,n,pad), result[3];
    float *ar = malloc((size_t)rows*k*sizeof(*ar)), *br = malloc((size_t)k*cols*sizeof(*br));
    double *oracle = malloc((size_t)rows*cols*sizeof(*oracle));
    void *original_a = malloc(a.words*4), *original_b = malloc(b.words*4);
    require(ar && br && oracle && original_a && original_b, "allocation failed in reference");
    populate(ar,br,rows,cols,k,argv[9]);
    for (int i = 0; i < m; ++i)
        memcpy(a.values+(size_t)i*a.ld, ar+(size_t)(i%rows)*k, (size_t)k*sizeof(*ar));
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j) b.values[(size_t)p*b.ld+j] = br[(size_t)p*cols+(j%cols)];
    memcpy(original_a, a.allocation, a.words*4);
    memcpy(original_b, b.allocation, b.words*4);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = 0;
            for (int p = 0; p < k; ++p) sum += (double)ar[(size_t)i*k+p] * br[(size_t)p*cols+j];
            oracle[(size_t)i*cols+j] = sum;
        }
    }
    double max_error = 0;
    int counts[3] = {-1, -1, -1};
    for (int q = 0; q < 3; ++q) {
        void *handle = dlopen(argv[1+q],RTLD_NOW|RTLD_LOCAL);
        if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 2; }
        multiply run = (multiply)dlsym(handle,"cob_sgemm_rowmajor");
        counter trace = (counter)dlsym(handle,"rank_hk_packed_forms");
        require(run != NULL && (trace != NULL) == (q == 2), "unexpected public entry or diagnostics export");
        result[q] = make(m,n,pad);
        require_single_thread();
        run(m,n,k,a.values,a.ld,b.values,b.ld,result[q].values,result[q].ld);
        require_single_thread();
        if (trace) for (int id = 0; id < 3; ++id) {
            counts[id] = trace(id);
            if (counts[id] != atoi(argv[10+id])) {
                fprintf(stderr,"counter[%d]=%d expected=%s\n",id,counts[id],argv[10+id]);
                return 1;
            }
        }
        require(guarded(&result[q]) && !memcmp(original_a,a.allocation,a.words*4) &&
                !memcmp(original_b,b.allocation,b.words*4), "input or padding changed");
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                size_t off = (size_t)i*result[q].ld+j;
                float value = result[q].values[off];
                double err = fabs(value-oracle[(size_t)(i%rows)*cols+j%cols]);
                if (!isfinite(value) || !isfinite(err) || (q && memcmp(result[0].values+off,result[q].values+off,4))) {
                    fprintf(stderr,"binary %d differs or nonfinite at (%d,%d): %.9g vs %.9g\n",q,i,j,value,
                            result[0].values[off]);
                    return 1;
                }
                max_error = fmax(max_error,err);
            }
        }
        dlclose(handle);
    }
    require(max_error <= 0.002, "full-output FP64 error above 0.002");
    printf("shape=%dx%dx%d pad=%d seed=%s mode=%s checked=%zu oracle_dots=%d bits=equal "
           "guards=unchanged inputs=unchanged threads_before_after=1 packs=%d views=%d original_forms=%d max_abs=%.12g\n",
           m,n,k,pad,argv[8],argv[9],(size_t)m*n,rows*cols,counts[0],counts[1],counts[2],max_error);
    return 0;
}
