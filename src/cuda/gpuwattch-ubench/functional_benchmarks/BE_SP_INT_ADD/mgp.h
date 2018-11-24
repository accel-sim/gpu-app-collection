#ifndef __gpu_mpower
#define __gpu_mpower
extern int profileKernel(char* benchmark, char* kernel);
extern int profileKernel(int sampleRate, char* benchmark, char* kernel);
extern void haltProfiling();
extern void resetKernelCount();
#endif
