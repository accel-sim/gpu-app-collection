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
#include <curand.h>
#include <cublas_v2.h>

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



__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, int iterations) {
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
   
   #pragma unroll 100
   for(int i=0; i<iterations; i++){
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }

   wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_col_major);
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
   data[i] = (float) rand() / RAND_MAX;
   }
}

int main(int argc, char* argv[]) {
   int iterations;
    if(argc!=2) {
      fprintf(stderr,"usage: %s #iterations\n",argv[0]);
      exit(1);
    }
    else {
      iterations = atoi(argv[1]);
    }
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_wmma;

   float *c_host_wmma;
   
   curandGenerator_t gen;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
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
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, iterations);
   cudaErrCheck(cudaEventRecord(stopWMMA));

   
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
 

   float wmmaTime;

   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("wmma took %fms\n", wmmaTime);
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));
   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_wmma));
   free(c_host_wmma);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}
