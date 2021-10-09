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
#define DATA_TYPE float
  
// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}



__global__ void power_microbench(float *data1, float *data2, uint32_t *data3, uint32_t *data4, float *res, int div, unsigned iterations) {

  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register float s1 = data1[gid];
  register float s2 = data2[gid];
  register uint32_t s3 = data3[gid];
  register uint32_t s4 = data4[gid];
  register float result = 0;
  register float Value1=0;
  register uint32_t Value2=0;

  // synchronize all threads
  asm volatile ("bar.sync 0;");

  if((gid%32)<div){
  //ROI
    #pragma unroll 100
    for (unsigned j=0 ; j<iterations ; ++j) {
      asm volatile ("{\t\n"
          "add.f32 %0, %1, %0;\n\t"
          "add.u32 %2, %3, %2;\n\t"
          "add.u32 %2, %3, %2;\n\t"
          // "add.u32 %2, %2, %0;\n\t"
          // "mul.lo.u32 %1, %0, %2;\n\t"
          "fma.rn.f32 %1, %1, %1 , %0;\n\t"
          "mad.lo.u32 %3, %3, %3 , %2;\n\t"
          "}" : "+f"(Value1),"+f"(s1),"+r"(s3),"+r"(Value2)
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
  res[gid] = Value1 + (float)Value2;
}

int main(int argc, char** argv){
  unsigned iterations;
  int blocks;
  int div;
  if (argc != 4){
    fprintf(stderr,"usage: %s #iterations #cores #ActiveThreadsperWarp\n",argv[0]);
    exit(1);
  }
  else {
    iterations = atoi(argv[1]);
    blocks = atoi(argv[2]);
    div = atoi(argv[3]);
  }
 
 printf("Power Microbenchmarks with iterations %u\n",iterations);
 int total_threads = THREADS_PER_BLOCK*blocks;


DATA_TYPE *data1 = (DATA_TYPE*) malloc(total_threads*sizeof(DATA_TYPE));
DATA_TYPE *data2 = (DATA_TYPE*) malloc(total_threads*sizeof(DATA_TYPE));


uint32_t *data3 = (uint32_t*) malloc(total_threads*sizeof(uint32_t));
uint32_t *data4 = (uint32_t*) malloc(total_threads*sizeof(uint32_t));

DATA_TYPE *res = (DATA_TYPE*) malloc(total_threads*sizeof(DATA_TYPE));

DATA_TYPE *data1_g;
DATA_TYPE *data2_g;

uint32_t *data3_g;
uint32_t *data4_g;

DATA_TYPE *res_g;

for (uint32_t i=0; i<total_threads; i++) {
  srand((unsigned)time(0));  
  data1[i] = (DATA_TYPE) rand() / RAND_MAX;
  srand((unsigned)time(0));
  data2[i] = (DATA_TYPE) rand() / RAND_MAX;
  srand((unsigned)time(0));  
  data3[i] = (uint32_t) rand() / RAND_MAX;
  srand((unsigned)time(0));
  data4[i] = (uint32_t) rand() / RAND_MAX;
}

gpuErrchk( cudaMalloc(&data1_g, total_threads*sizeof(DATA_TYPE)) );
gpuErrchk( cudaMalloc(&data2_g, total_threads*sizeof(DATA_TYPE)) );

gpuErrchk( cudaMalloc(&data3_g, total_threads*sizeof(uint32_t)) );
gpuErrchk( cudaMalloc(&data4_g, total_threads*sizeof(uint32_t)) );

gpuErrchk( cudaMalloc(&res_g, total_threads*sizeof(DATA_TYPE)) );

gpuErrchk( cudaMemcpy(data1_g, data1, total_threads*sizeof(DATA_TYPE), cudaMemcpyHostToDevice) );
gpuErrchk( cudaMemcpy(data2_g, data2, total_threads*sizeof(DATA_TYPE), cudaMemcpyHostToDevice) );

gpuErrchk( cudaMemcpy(data3_g, data3, total_threads*sizeof(uint32_t), cudaMemcpyHostToDevice) );
gpuErrchk( cudaMemcpy(data4_g, data4, total_threads*sizeof(uint32_t), cudaMemcpyHostToDevice) );


power_microbench<<<blocks,THREADS_PER_BLOCK>>>(data1_g, data2_g, data3_g, data4_g, res_g, div, iterations);
gpuErrchk( cudaPeekAtLastError() );


gpuErrchk( cudaMemcpy(res, res_g, total_threads*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost) );


cudaFree(data1_g);
cudaFree(data2_g);
cudaFree(data3_g);
cudaFree(data4_g);
cudaFree(res_g);
free(data1);
free(data2);
free(data3);
free(data4);
free(res);

  return 0;
} 