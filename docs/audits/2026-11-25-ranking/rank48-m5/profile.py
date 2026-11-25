from pathlib import Path
import subprocess,sys
h=Path(__file__).resolve().parent
s=h/'bench/tools/world-rank/rank48'
text=(s/'adapter.c').read_text()
text='#include <time.h>\nstatic double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}\n'+text
for old,new in [
('    float *storage = malloc(', '    double start=now();\n    float *storage = malloc('),
('    make_input(h, d, a,', '    double alloc=now();\n    make_input(h, d, a,'),
('    for (int i = 0; i < PRODUCT_COUNT; ++i) {\n        products[i]', '    double inputs=now();\n    for (int i = 0; i < PRODUCT_COUNT; ++i) {\n        products[i]'),
('    for (int i = 0; i < 16; ++i)\n        outputs[i]', '    double leaves=now();\n    for (int i = 0; i < 16; ++i)\n        outputs[i]'),
('    free(storage);','    double outputs_done=now();\n    free(storage);'),
('        ++last_fringe;\n    }\n}', '''        ++last_fringe;
    }
    double end=now();
    printf("PHASE,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\\n",PRODUCT_COUNT,alloc-start,inputs-alloc,leaves-inputs,outputs_done-leaves,end-outputs_done,end-start);
}''')]:
 if text.count(old)!=1:raise ValueError(old)
 text=text.replace(old,new)
p=h/'profile-adapter.c';p.write_text(text.replace('#include "../vendor.h"','#include "vendor.h"'))
for rank in (48,49):
 subprocess.run(['clang','-O3','-std=c11','-mcpu=native','-dynamiclib','-I'+str(s.parent),'-I'+str(h/('gen'+str(rank))),p,'-o',h/f'profile{rank}.dylib'],check=True)
