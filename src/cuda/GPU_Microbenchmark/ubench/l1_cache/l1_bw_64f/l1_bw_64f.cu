// This code is a modification of L1 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
// https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the maximum read bandwidth of L1 cache for 64 bit

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define REPEAT_TIMES 256
#ifdef TUNER
#include "../../../hw_def/hw_def.h"
// array size is half the L1 size (2) * float size (4)
#define ARRAY_SIZE (L1_SIZE / 16)
#else
#include "../../../hw_def/common/gpuConfig.h"
// #define THREADS_PER_BLOCK 1024
// #define THREADS_PER_SM 1024
// #define BLOCKS_NUM 1
// #define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
// #define WARP_SIZE 32
#define ARRAY_SIZE 8192    // ARRAY_SIZE has to be less than L1_SIZE
#define L1_SIZE 16384      // L1 size in 64-bit. Volta L1 size is 128KB, i.e. 16K of 64-bit
#define CLK_FREQUENCY 1410 // Asumme A100 freq

#endif

__global__ void l1_bw(uint64_t *startClk, uint64_t *stopClk, double *dsink,
                      double *posArray)
{

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t uid = blockIdx.x * blockDim.x + tid;

  // a register to avoid compiler optimization
  double sink0 = 0;
  double sink1 = 0;

  // populate l1 cache to warm up
  for (uint32_t i = tid; i < ARRAY_SIZE; i += blockDim.x)
  {
    double *ptr = posArray + i;
    // use ca modifier to cache the load in L1
    asm volatile("{\t\n"
                 ".reg .f64 data;\n\t"
                 "ld.global.ca.f64 data, [%1];\n\t"
                 "add.f64 %0, data, %0;\n\t"
                 "}"
                 : "+d"(sink0)
                 : "l"(ptr)
                 : "memory");
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  // load data from l1 cache and accumulate
  for (uint32_t j = 0; j < REPEAT_TIMES; j++)
  {
    double *ptr = posArray + ((tid + (j * warpSize * 2)) % ARRAY_SIZE);
    asm volatile("{\t\n"
                 ".reg .f64 data<2>;\n\t"
                 "ld.global.ca.f64 data0, [%2+0];\n\t"
                 "ld.global.ca.f64 data1, [%2+256];\n\t"
                 "add.f64 %0, data0, %0;\n\t"
                 "add.f64 %1, data1, %1;\n\t"
                 "}"
                 : "+d"(sink0), "+d"(sink1)
                 : "l"(ptr)
                 : "memory");
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  // write time and data back to memory
  startClk[uid] = start;
  stopClk[uid] = stop;
  dsink[uid] = sink0 + sink1;
}

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);
#ifdef TUNER

  config.BLOCKS_NUM = 1;
  config.TOTAL_THREADS = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;
  config.THREADS_PER_SM = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;

  // ARRAY_SIZE has to be less than L1_SIZE
  assert(ARRAY_SIZE * sizeof(double) < L1_SIZE);
#endif
  uint64_t *startClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  double *posArray = (double *)malloc(ARRAY_SIZE * sizeof(double));
  double *dsink = (double *)malloc(config.TOTAL_THREADS * sizeof(double));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  double *posArray_g;
  double *dsink_g;

  for (uint32_t i = 0; i < ARRAY_SIZE; i++)
    posArray[i] = (double)i;

  gpuErrchk(cudaMalloc(&startClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(double)));
  gpuErrchk(cudaMalloc(&dsink_g, config.TOTAL_THREADS * sizeof(double)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, ARRAY_SIZE * sizeof(double),
                       cudaMemcpyHostToDevice));

  l1_bw<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                                         posArray_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, config.TOTAL_THREADS * sizeof(double),
                       cudaMemcpyDeviceToHost));

  double bw, BW;
  uint64_t total_time =
      *std::max_element(&stopClk[0], &stopClk[config.TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[config.TOTAL_THREADS]);
  bw = (double)(REPEAT_TIMES * config.THREADS_PER_SM * sizeof(double) * 2) /
       ((double)total_time);
  BW = bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "L1 bandwidth = " << bw << "(byte/clk/SM), " << BW
            << "(GB/s/SM)\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return 1;
}
