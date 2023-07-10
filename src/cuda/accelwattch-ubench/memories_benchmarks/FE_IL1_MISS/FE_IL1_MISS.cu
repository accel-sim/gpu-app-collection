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

//includes project
#include<repeat.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 640


// Variables
unsigned* h_A;
unsigned* h_C;
unsigned* d_A;
unsigned* d_C;


// Functions
void CleanupResources(void);
void RandomInit(unsigned*, int);


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
__global__ void PowerKernal(const unsigned* A,unsigned* C, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned I1=A[i];
    #pragma unroll 1
    //Excessive Logical Unit access
    for(unsigned k=0; k<iterations;k++) {
    // BLOCK-0 (For instruction size of 16 bytes)
    	__asm volatile (	
    	"B0: bra.uni B1;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B1: bra.uni B2;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")
    			
    	"B2: bra.uni B3;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B3: bra.uni B4;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B4: bra.uni B5;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B5: bra.uni B6;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B6: bra.uni B7;\n\t" 
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B7: bra.uni B8;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B8: bra.uni B9;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B9: bra.uni B10;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B10: bra.uni B11;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B11: bra.uni B12;\n\t" 
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B12: bra.uni B13;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B13: bra.uni B14;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B14: bra.uni B15;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B15: bra.uni B16;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B16: bra.uni B17;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B17: bra.uni B18;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B18: bra.uni B19;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B19: bra.uni B20;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B20: bra.uni B21;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B21: bra.uni B22;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B22: bra.uni B23;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B23: bra.uni B24;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B24: bra.uni B25;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B25: bra.uni B26;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B26: bra.uni B27;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B27: bra.uni B28;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B28: bra.uni B29;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B29: bra.uni B30;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B30: bra.uni B31;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")
    	
        "B31: bra.uni B32;\n\t"
        repeat8191("add.u32   %0, %1, 1;\n\t")

        "B32: bra.uni B33;\n\t"
        repeat8191("add.u32   %0, %1, 1;\n\t")

        "B33: bra.uni B34;\n\t"
        repeat8191("add.u32   %0, %1, 1;\n\t")

    	"B34: bra.uni LOOP;\n\t"
    	repeat8191("add.u32   %0, %1, 1;\n\t")
        
    	"LOOP:"
    	: "=r"(I1) : "r"(I1));
    	
    }
    C[i]=I1;
    __syncthreads();

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
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size = N * sizeof(unsigned);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (unsigned*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_C = (unsigned*)malloc(size);
 if (h_C == 0) CleanupResources();

 // Initialize input vectors
 RandomInit(h_A, N);

 // Allocate vectors in device memory
 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
 checkCudaErrors( cudaMalloc((void**)&d_C, size) );

 // Copy vectors from host memory to device memory
 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );

 cudaEvent_t start, stop;
 float elapsedTime = 0;
 checkCudaErrors(cudaEventCreate(&start));
 checkCudaErrors(cudaEventCreate(&stop));
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);

 checkCudaErrors(cudaEventRecord(start));
 PowerKernal<<<dimGrid,dimBlock>>>(d_A,d_C, iterations);
 checkCudaErrors(cudaEventRecord(stop));

 checkCudaErrors(cudaEventSynchronize(stop));
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
 printf("gpu execution time = %.2f s\n", elapsedTime/1000);

 getLastCudaError("kernel launch failure");
 cudaThreadSynchronize();

 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 checkCudaErrors(cudaEventDestroy(start));
 checkCudaErrors(cudaEventDestroy(stop));
 checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
 
 CleanupResources();

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
    cudaFree(d_A);
  if (d_C)
    cudaFree(d_C);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_C)
    free(h_C);

}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
    srand((unsigned)time(0));  
    data[i] = rand() / RAND_MAX;
  }
}






