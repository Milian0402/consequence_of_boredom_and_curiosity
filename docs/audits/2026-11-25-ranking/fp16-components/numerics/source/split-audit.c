#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>

void fp16_check_split(float,uint16_t*);
uint64_t fp16_control_register(void);
int main(void)
{
    unsigned long long total=0,two_miss=0,three_miss=0,low_sub=0;
    uint32_t first=0;double worst=0;
    for(int e=-14;e<=12;e++){
        uint32_t x0=(uint32_t)(e+127)<<23;
        for(uint32_t frac=0;frac<(1u<<23);frac++){
            uint32_t u=x0+frac;float x;_Float16 h[3];uint16_t words[3];
            memcpy(&x,&u,4);fp16_check_split(x,words);memcpy(h,words,6);
            double p=(double)h[0]+0x1p-11*(double)h[1];
            double full=p+0x1p-22*(double)h[2];
            if(p!=x){two_miss++;if(!first)first=u;}
            three_miss+=full!=x;
            low_sub+=((words[2]&0x7fff)>0 && (words[2]&0x7c00)==0);
            double ratio=fabs(p-x)/x;if(ratio>worst)worst=ratio;
            total++;
        }
    }
    printf("positive_inputs=%llu two_component_misses=%llu three_component_misses=%llu third_plane_subnormal=%llu worst_two_relative=%.15g first_two_miss=0x%08x fpcr=0x%" PRIx64 "\n",total,two_miss,three_miss,low_sub,worst,first,fp16_control_register());
    return three_miss!=0;
}
