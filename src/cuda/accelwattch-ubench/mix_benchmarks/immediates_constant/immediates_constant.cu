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


// includes CUDA
#include <cuda_runtime.h>

// includes project
#include <../include/repeat.h>


#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 640

__constant__ unsigned ConstArray1[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray2[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray3[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray4[THREADS_PER_BLOCK];


unsigned* h_Value;
unsigned* d_Value;


// Functions
void CleanupResources(void);
void RandomInit(unsigned*, int);
FILE *fp;


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

texture<float,1,cudaReadModeElementType> texmem1;
texture<float,1,cudaReadModeElementType> texmem2;
texture<float,1,cudaReadModeElementType> texmem3;


__global__ void PowerKernal(unsigned* Value, int iterations)
{
	int i = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

	unsigned Value1=0;
	unsigned Value2=0;
	unsigned Value3=0;
	unsigned Value4=0;
    for(unsigned k=0; k<iterations;k++) {
		Value1=ConstArray1[(i+k)%THREADS_PER_BLOCK];
		Value2=ConstArray2[(i+k+1)%THREADS_PER_BLOCK];
		Value3=ConstArray3[(i+k+1)%THREADS_PER_BLOCK];
		Value4=ConstArray4[(i+k+1)%THREADS_PER_BLOCK];
		__asm volatile (
	    repeat2("add.u32   %0, %1, 1;\n\t")
	    : "=r"(Value1) : "r"(Value4));
	    __asm volatile (
	    repeat2("add.u32   %0, %1, 5;\n\t")
	    : "=r"(Value2) : "r"(Value1));
	    __asm volatile (
	    repeat2("add.u32   %0, %1, 1;\n\t")
	    : "=r"(Value3) : "r"(Value2));
	    __asm volatile (
	    repeat2("add.u32   %0, %1, 5;\n\t")
	    : "=r"(Value4) : "r"(Value3));
	    *Value+=Value1+Value2+Value3+Value4;
	}
}



// Host code

int main(int argc, char** argv) 
{
    int iterations;
    if (argc != 2){
        fprintf(stderr,"usage: %s #iterations\n",argv[0]);
        exit(1);
    }
    else{
        iterations = atoi(argv[1]);
    }

    printf("Power Microbenchmark with %d iterations\n",iterations);
	 unsigned array1[THREADS_PER_BLOCK];
	 h_Value = (unsigned *) malloc(sizeof(unsigned));
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand((unsigned)time(0));
		array1[i] = rand() / RAND_MAX;
	 }
	 unsigned array2[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand((unsigned)time(0));
		array2[i] = rand() / RAND_MAX;
	 }
	 unsigned array3[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand((unsigned)time(0));
		array3[i] = rand() / RAND_MAX;
	 }
	 unsigned array4[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand((unsigned)time(0));
		array4[i] = rand() / RAND_MAX;
	 }

	 cudaMemcpyToSymbol(ConstArray1, array1, sizeof(unsigned) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol(ConstArray2, array2, sizeof(unsigned) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol(ConstArray3, array3, sizeof(unsigned) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol(ConstArray4, array4, sizeof(unsigned) * THREADS_PER_BLOCK );
	 
	 checkCudaErrors( cudaMalloc((void**)&d_Value, sizeof(unsigned)) );
	 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	 dim3 dimGrid(NUM_OF_BLOCKS,1);
	 dim3 dimBlock(THREADS_PER_BLOCK,1);

	 cudaEvent_t start, stop;
	  float elapsedTime = 0;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));

	  checkCudaErrors(cudaEventRecord(start));
	  PowerKernal<<<dimGrid,dimBlock>>>(d_Value, iterations);
	  checkCudaErrors(cudaEventRecord(stop));
	 
	  checkCudaErrors(cudaEventSynchronize(stop));
	  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	  printf("gpu execution time = %.2f s\n", elapsedTime/1000);
	  getLastCudaError("kernel launch failure");
	  cudaThreadSynchronize();
	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
	  return 0;
}






