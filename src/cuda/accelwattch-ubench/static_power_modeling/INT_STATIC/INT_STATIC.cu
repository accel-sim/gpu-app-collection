// Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define DATA_TYPE uint32_t
  
// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


template <class T>
__global__ void power_microbench(T *data1, T *data2, T *res, int div, uint64_t iterations) {

  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s1 = data1[gid];
  register T s2 = data2[gid];
  register T result = 0;
  register T Value1=0;
  register T Value2=0;

  // synchronize all threads
  asm volatile ("bar.sync 0;");

  if((gid%32)<div){
  //ROI
    #pragma unroll 100
    for (uint64_t j=0 ; j<iterations ; ++j) {
      asm volatile ("{\t\n"
          "add.u32 %0, %2, %0;\n\t"
          "add.u32 %0, %1, %0;\n\t"
          // "add.u32 %2, %2, %0;\n\t"
          // "mul.lo.u32 %1, %0, %2;\n\t"
          "mad.lo.u32 %2, %2, %2 , %0;\n\t"
          "}" : "+r"(result),"+r"(s1),"+r"(s2)
      );
      // result=s1+s2;
      // Value2=s1-s2;
      // result+=Value1;
      // result*=Value1;
      // Value1=Value2+result;
      // result=Value1+Value2;
    }
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // write data back to memory
  res[gid] = result;
}

int main(int argc, char** argv){
  uint64_t iterations;
  int blocks;
  int div;
  if (argc != 4){
    fprintf(stderr,"usage: %s #iterations #cores #ActiveThreadsperWarp\n",argv[0]);
    exit(1);
  }
  else {
    iterations = atoll(argv[1]);
    blocks = atoi(argv[2]);
    div = atoi(argv[3]);
  }
 
 printf("Power Microbenchmarks with iterations %llu\n",iterations);
 int total_threads = THREADS_PER_BLOCK*blocks;


DATA_TYPE *data1 = (DATA_TYPE*) malloc(total_threads*sizeof(DATA_TYPE));
DATA_TYPE *data2 = (DATA_TYPE*) malloc(total_threads*sizeof(DATA_TYPE));
DATA_TYPE *res = (DATA_TYPE*) malloc(total_threads*sizeof(DATA_TYPE));

DATA_TYPE *data1_g;
DATA_TYPE *data2_g;
DATA_TYPE *res_g;

for (uint32_t i=0; i<total_threads; i++) {
  srand((unsigned)time(0));  
  data1[i] = (DATA_TYPE) rand() / RAND_MAX;
  srand((unsigned)time(0));
  data2[i] = (DATA_TYPE) rand() / RAND_MAX;
}

gpuErrchk( cudaMalloc(&data1_g, total_threads*sizeof(DATA_TYPE)) );
gpuErrchk( cudaMalloc(&data2_g, total_threads*sizeof(DATA_TYPE)) );
gpuErrchk( cudaMalloc(&res_g, total_threads*sizeof(DATA_TYPE)) );

gpuErrchk( cudaMemcpy(data1_g, data1, total_threads*sizeof(DATA_TYPE), cudaMemcpyHostToDevice) );
gpuErrchk( cudaMemcpy(data2_g, data2, total_threads*sizeof(DATA_TYPE), cudaMemcpyHostToDevice) );

power_microbench<DATA_TYPE><<<blocks,THREADS_PER_BLOCK>>>(data1_g, data2_g, res_g, div, iterations);
gpuErrchk( cudaPeekAtLastError() );


gpuErrchk( cudaMemcpy(res, res_g, total_threads*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost) );


cudaFree(data1_g);
cudaFree(data2_g);
cudaFree(res_g);
free(data1);
free(data2);
free(res);

  return 0;
} 
