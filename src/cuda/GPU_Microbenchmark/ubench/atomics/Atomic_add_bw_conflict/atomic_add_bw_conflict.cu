#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifdef TUNER

#include "../../../hw_def/hw_def.h"
#define REPEAT_TIMES 2048
#else
#define REPEAT_TIMES 16
#include "../../../hw_def/common/gpuConfig.h"
// #define THREADS_PER_BLOCK 1024
// #define THREADS_PER_SM 2048
// #define BLOCKS_NUM 160
// #define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
// #define WARP_SIZE 32

// #define ARRAY_SIZE TOTAL_THREADS

// #define gpuErrchk(ans)                                                         \
//   { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line,
//                       bool abort = true) {
//   if (code != cudaSuccess) {
//     fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
//             line);
//     if (abort)
//       exit(code);
//   }
// }

#endif

template <class T>
__global__ void atomic_bw(uint32_t *startClk, uint32_t *stopClk, T *data1,
                          T *res)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t sum;
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  for (int j = 0; j < REPEAT_TIMES; ++j)
  {
    sum = sum + atomicAdd(&data1[0], 10);
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
  res[gid] = sum;
}

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  uint32_t *startClk = (uint32_t *)malloc(config.TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(config.TOTAL_THREADS * sizeof(uint32_t));
  int32_t *data1 = (int32_t *)malloc(config.TOTAL_THREADS * sizeof(int32_t));
  int32_t *res = (int32_t *)malloc(config.TOTAL_THREADS * sizeof(int32_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  int32_t *data1_g;
  int32_t *res_g;

  for (uint32_t i = 0; i < config.TOTAL_THREADS; i++)
  {
    data1[i] = (int32_t)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, config.TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, config.TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g, config.TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&res_g, config.TOTAL_THREADS * sizeof(int32_t)));

  gpuErrchk(cudaMemcpy(data1_g, data1, config.TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyHostToDevice));

  atomic_bw<int32_t><<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(startClk_g, stopClk_g,
                                                                      data1_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, config.TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, config.TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res, res_g, config.TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));

  float bw;
  uint32_t total_time =
      *std::max_element(&stopClk[0], &stopClk[config.TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[config.TOTAL_THREADS]);
  // uint32_t total_time = stopClk[0] - startClk[0];
  bw = ((float)(REPEAT_TIMES * config.TOTAL_THREADS * 4) / (float)(total_time));
  printf("Atomic int32 bandwidth = %f (byte/clk)\n", bw);
  printf("Total Clk number = %u \n", total_time);

  return 1;
}
