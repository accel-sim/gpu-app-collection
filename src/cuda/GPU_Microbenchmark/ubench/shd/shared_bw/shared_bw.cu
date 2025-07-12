#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifdef TUNER
#include "../../../hw_def/hw_def.h"
#define SHARED_MEM_SIZE (32 * 1024 / 4) // 32 KB
#define ITERS 4096
#else
#include "../../../hw_def/common/gpuConfig.h"
#define SHARED_MEM_SIZE_BYTE (48 * 1024) // size in bytes, max 96KB for v100
#define SHARED_MEM_SIZE (SHARED_MEM_SIZE_BYTE / 4)
#define ITERS (SHARED_MEM_SIZE / 2)
#define CLK_FREQUENCY 1410 // Asumme A100 freq
// #define BLOCKS_NUM 1
// #define THREADS_PER_BLOCK 1024
// #define WARP_SIZE 32
// #define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#endif

__global__ void shared_bw(uint64_t *startClk, uint64_t *stopClk,
                          uint32_t *dsink, uint32_t stride)
{

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t n_threads = blockDim.x * gridDim.x;

  // a register to avoid compiler optimization
  // uint32_t sink0 = 0;
  register uint32_t tmp = uid;

  uint64_t start = 0;
  uint64_t stop = 0;

  __shared__ uint32_t s[SHARED_MEM_SIZE]; // static shared memory
  // uint32_t s[SHARED_MEM_SIZE];
  // one thread to initialize the pointer-chasing array
  for (uint32_t i = uid; i < (SHARED_MEM_SIZE); i += n_threads)
    s[i] = (i + stride) % SHARED_MEM_SIZE;

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  // load data from shared memory
  for (uint32_t i = 0; i < ITERS; ++i)
  {
    tmp = s[tmp];
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  // sink0 = tmp;
  // write time and data back to memory
  startClk[uid] = start;
  stopClk[uid] = stop;
  dsink[uid] = tmp;
}

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);
#ifdef TUNER
  config.BLOCKS_NUM = 1;
  config.TOTAL_THREADS = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;
  config.THREADS_PER_SM = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;

  assert(SHARED_MEM_SIZE * sizeof(uint32_t) < config.MAX_SHARED_MEM_SIZE_PER_BLOCK);
#endif
  uint64_t *startClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  uint32_t *dsink = (uint32_t *)malloc(config.TOTAL_THREADS * sizeof(uint32_t));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  uint32_t *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, config.TOTAL_THREADS * sizeof(uint32_t)));

  shared_bw<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                                             config.THREADS_PER_BLOCK);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, config.TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));

  double bw, BW;
  uint64_t total_time =
      *std::max_element(&stopClk[0], &stopClk[config.TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[config.TOTAL_THREADS]);
  bw =
      (double)(ITERS * config.TOTAL_THREADS * sizeof(uint32_t)) / ((double)total_time);
  BW = bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "Shared Memory Bandwidth = " << bw << "(byte/clk/SM), " << BW
            << "(GB/s/SM)\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return 1;
}
