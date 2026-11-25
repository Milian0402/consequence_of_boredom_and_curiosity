/* Include after the candidate gemm.c to test the fused B pack against the existing packer. */
#include <stdio.h>

static uint32_t rng = 4130701;
static const uint32_t pad_word = 0x7fc45e61u;

static float input_value(int mode) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    if (mode == 1 && (rng & 3)) return rng & 4 ? 0.0f : -0.0f;
    return ((int32_t)(rng>>8)-8388608) * 0x1p-22f;
}

static int pack_case(int depth, int cols, int nr, int kc, int plus2, int plus3, int mode) {
    int rows = 2*depth+1, ld = 2*cols+1+5, width = (cols+nr-1)/nr*nr;
    size_t size = (size_t)rows*ld, pitch = (size_t)depth*width, words = 7*pitch;
    float *inputs = malloc(3*size*sizeof(float)), *material = malloc(size*sizeof(float));
    float *expected = malloc(words*sizeof(float)), *buf = malloc((words+16)*sizeof(float));
    float *saved = malloc(3*size*sizeof(float));
    if (!inputs || !material || !expected || !buf || !saved) abort();
    float *got = buf+8;
    for (size_t i = 0; i < 3*size; ++i) inputs[i] = input_value(mode);
    if (mode == 2) {
        /* Make the first pair cancel exactly, including its signed-zero cases. */
        for (size_t i = 0; i < size; ++i) inputs[size+i] = plus2 ? -inputs[i] : inputs[i];
    }
    memcpy(saved,inputs,3*size*sizeof(float));
    for (size_t i = 0; i < words+16; ++i) memcpy(buf+i,&pad_word,4);
    for (size_t i = 0; i < size; ++i) {
        float pair = plus2 ? inputs[i]+inputs[size+i] : inputs[i]-inputs[size+i];
        material[i] = plus3 ? pair+inputs[2*size+i] : pair-inputs[2*size+i];
    }
    cob_hk_b_form form = {inputs,inputs+size,inputs+2*size,plus2,plus3};
    cob_hk_pack_seven_b(form,ld,depth,cols,nr,width,kc,got);
    float *b11=material, *b12=b11+cols, *b21=b11+(size_t)depth*ld, *b22=b21+cols;
    cob_fused_operand terms[7] = {
        {b11,NULL,0}, {b12,b22,-1}, {b21,b11,-1}, {b11,b22,1},
        {b22,NULL,0}, {b11,b12,1}, {b21,b22,1}
    };
    for (int p=0; p<depth; p+=kc) {
        int count = cob_min_i32(depth-p,kc);
        for (int q=0; q<7; ++q)
            cob_fused_pack_b(expected+(size_t)q*pitch+(size_t)p*width,terms[q],(size_t)p*ld,
                             ld,count,cols,nr);
    }
    int equal = !memcmp(expected,got,words*sizeof(float)) && !memcmp(inputs,saved,3*size*sizeof(float));
    for (int i=0; i<8; ++i)
        equal &= !memcmp(buf+i,&pad_word,4) && !memcmp(buf+8+words+i,&pad_word,4);
    free(inputs); free(saved); free(material); free(expected); free(buf);
    if (!equal) fprintf(stderr,"packed bits/guards failed: k=%d n=%d nr=%d kc=%d signs=%d%d mode=%d\n",
                        depth,cols,nr,kc,plus2,plus3,mode);
    return equal;
}

int main(void) {
    int count=0;
    for (int nr=4; nr<=32; nr*=2) {
        int depths[]={4,5,6,255,256,257}, cols[]={1,nr-1,nr,nr+1,2*nr+3};
        for (size_t d=0;d<sizeof(depths)/sizeof(*depths);++d) {
            int kc=d<3?5:256;
            for (size_t j=0;j<sizeof(cols)/sizeof(*cols);++j)
                for (int signs=0;signs<4;++signs) for (int mode=0;mode<3;++mode) {
                    if (!pack_case(depths[d],cols[j],nr,kc,signs&1,signs>>1,mode)) return 1;
                    ++count;
                }
        }
    }
    printf("packed fixtures=%d seven_views_each=1 binary_reference=materialized_then_existing_packer "
           "k_panels=5,256 nr=4,8,16,32 input_images_and_guards=unchanged\n",count);
}
