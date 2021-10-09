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
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string>  

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define NUM_OF_BLOCKS 80

// Variables

__constant__ unsigned ConstArray1[THREADS_PER_BLOCK*NUM_OF_BLOCKS];
unsigned* h_Value;
unsigned* d_Value;


// Functions


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
  if(cudaSuccess != err){
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
	 exit(-1);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
	fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
	exit(-1);
  }
}

// end of CUDA Helper Functions




// Device code
__global__ void PowerKernal(unsigned* Value, unsigned* const1, unsigned long long iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // unsigned loadAddr = A+ i;
    // unsigned storeAddr = B+ i;
    unsigned load_value;
  unsigned sum_value = 0;
  
  #pragma unroll 100

    for(unsigned long long k=0; k<iterations;k++) {
      //load_value+=ConstArray1[i];
      __asm volatile(
        "ld.const.u32 %0, [%1];" 
        : "=r"(load_value) : "l"(&const1[i]) : "memory");
      __asm volatile("add.u32 %0, %0, %1;" : "+r"(sum_value) : "r"(load_value));
      // __asm volatile(
      //  "st.global.wb.u32 [%0], %1;"
      //  : : "l"((unsigned long)(B+i)), "r"(load_value) 
      // );

    }
    *Value = sum_value;
    __syncthreads();
}


// Host code

int main(int argc, char** argv) 
{

 unsigned long long iterations;
 char *ptr;
 if (argc != 2){
  fprintf(stderr,"usage: %s #iterations\n",argv[0]);
  exit(1);
 }
 else{
  iterations = strtoull(argv[1], &ptr, 10);
 }

 printf("Power Microbenchmark with %llu iterations\n",iterations);
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 unsigned array1[N];
 h_Value = (unsigned *) malloc(sizeof(unsigned));
 for(int i=0; i<N;i++){
	srand((unsigned)time(0));
	array1[i] = rand() / RAND_MAX;
 }


 cudaMemcpyToSymbol(ConstArray1, array1, sizeof(unsigned) * N );

 checkCudaErrors( cudaMalloc((void**)&d_Value, sizeof(unsigned)) );

 cudaEvent_t start, stop;
 float elapsedTime = 0;
 checkCudaErrors(cudaEventCreate(&start));
 checkCudaErrors(cudaEventCreate(&stop));
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);

 checkCudaErrors(cudaEventRecord(start));
 PowerKernal<<<dimGrid,dimBlock>>>(d_Value, ConstArray1, iterations);
 checkCudaErrors(cudaEventRecord(stop));

 checkCudaErrors(cudaEventSynchronize(stop));
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
 printf("gpu execution time = %.2f s\n", elapsedTime/1000);

 getLastCudaError("kernel launch failure");
 cudaThreadSynchronize();

 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 checkCudaErrors( cudaMemcpy(h_Value, d_Value, sizeof(unsigned), cudaMemcpyDeviceToHost) );

 checkCudaErrors(cudaEventDestroy(start));
 checkCudaErrors(cudaEventDestroy(stop));

 return 0;
}







