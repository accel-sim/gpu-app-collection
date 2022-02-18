/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/********************************************************************
 *      Modified by:
 * Vijay Kandiah, Northwestern University
 ********************************************************************/
#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#define THREADS_PER_BLOCK 1024
#define DATA_TYPE float
#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 1024
#define MATRIX_N 1024
#define MATRIX_K 1024



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}



__global__ void power_microbench(float *data1, float *data2, uint32_t *data3, uint32_t *data4, float *res, int div, unsigned iterations, half *a, half *b, float *c, int M, int N, int K) {

  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register float s1 = data1[gid];
  register float s2 = data2[gid];
  register uint32_t s3 = data3[gid];
  register uint32_t s4 = data4[gid];
  register float result = 0;
  register float Value1=0;
  register uint32_t Value2=0;
   // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::load_matrix_sync(a_frag, a , lda);
  wmma::load_matrix_sync(b_frag, b , ldb);
  wmma::load_matrix_sync(c_frag, c , ldc, wmma::mem_col_major);

  // synchronize all threads
  asm volatile ("bar.sync 0;");

  
  //ROI
  #pragma unroll 100
  for (unsigned j=0 ; j<iterations ; ++j) {
    if((gid%32)<div){
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
    }
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // result=s1+s2;
    // Value2=s1-s2;
    // result+=Value1;
    // result*=Value1;
    // Value1=Value2+result;
    // result=Value1+Value2;
  }
  

  // synchronize all threads
  asm volatile("bar.sync 0;");
  wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_col_major);
  // write data back to memory
  res[gid] = Value1 + (float)Value2;
}


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

void RandomInit_fp(float* data, int n)
{
   for (int i = 0; i < n; ++i){
   data[i] = (float)rand() / RAND_MAX;
   }
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
 
  printf("Power Microbenchmarks with iterations %lu\n",iterations);

     // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);


  int total_threads = blockDim.x*blockDim.y*gridDim.x*gridDim.y;


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

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c;
  float *c_wmma;

  float *c_host_wmma;

  curandGenerator_t gen;
  
  cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

  c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));


   float *a_fp32_h = (float*) malloc(MATRIX_M * MATRIX_K*sizeof(float));
   float *b_fp32_h = (float*) malloc(MATRIX_K * MATRIX_N*sizeof(float));
   float *c_h = (float*) malloc(MATRIX_M * MATRIX_N*sizeof(float));
   RandomInit_fp(a_fp32_h, MATRIX_M * MATRIX_K);
   RandomInit_fp(b_fp32_h, MATRIX_K * MATRIX_N);
   RandomInit_fp( c_h, MATRIX_M * MATRIX_N);
   cudaErrCheck(cudaMemcpy(c, c_h, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(a_fp32, a_fp32_h, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp32, b_fp32_h, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
  convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);



  cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));


  printf("\nM = %d, N = %d, K = %d\n\n", MATRIX_M, MATRIX_N, MATRIX_K);

  gpuErrchk( cudaMalloc(&data1_g, total_threads*sizeof(DATA_TYPE)) );
  gpuErrchk( cudaMalloc(&data2_g, total_threads*sizeof(DATA_TYPE)) );

  gpuErrchk( cudaMalloc(&data3_g, total_threads*sizeof(uint32_t)) );
  gpuErrchk( cudaMalloc(&data4_g, total_threads*sizeof(uint32_t)) );

  gpuErrchk( cudaMalloc(&res_g, total_threads*sizeof(DATA_TYPE)) );

  gpuErrchk( cudaMemcpy(data1_g, data1, total_threads*sizeof(DATA_TYPE), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(data2_g, data2, total_threads*sizeof(DATA_TYPE), cudaMemcpyHostToDevice) );

  gpuErrchk( cudaMemcpy(data3_g, data3, total_threads*sizeof(uint32_t), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(data4_g, data4, total_threads*sizeof(uint32_t), cudaMemcpyHostToDevice) );


  power_microbench<<<gridDim,blockDim>>>(data1_g, data2_g, data3_g, data4_g, res_g, div, iterations, a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K);
  gpuErrchk( cudaPeekAtLastError() );


  gpuErrchk( cudaMemcpy(res, res_g, total_threads*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost) );


  cudaFree(data1_g);
  cudaFree(data2_g);
  cudaFree(data3_g);
  cudaFree(data4_g);
  cudaFree(res_g);
  cudaErrCheck(cudaFree(a_fp32));
  cudaErrCheck(cudaFree(b_fp32));
  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));
  cudaErrCheck(cudaFree(c));
  cudaErrCheck(cudaFree(c_wmma));
  free(data1);
  free(data2);
  free(data3);
  free(data4);
  free(res);
  free(c_host_wmma);
  cudaErrCheck(cudaDeviceReset());
  return 0;
} 