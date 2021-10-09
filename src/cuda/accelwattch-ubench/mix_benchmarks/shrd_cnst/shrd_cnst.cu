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

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 640

// Variables

__constant__ float ConstArray1[THREADS_PER_BLOCK];
__constant__ float ConstArray2[THREADS_PER_BLOCK];
float* h_Value;
float* d_Value;
bool noprompt = false;
unsigned int my_timer;

// Functions
void CleanupResources(void);


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
__global__ void PowerKernal(float* Value, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    __device__  __shared__ float I1[THREADS_PER_BLOCK];
    __device__  __shared__ float I2[THREADS_PER_BLOCK];
    __device__ __shared__ float I3[THREADS_PER_BLOCK];
    __device__  __shared__ float I4[THREADS_PER_BLOCK];
 	
    I1[i]=i;
    I2[i]=i/2;
    I3[i]=i;
    I4[i]=i+1;
    
    //Do Some Computation
    float Value1;
   float Value2;
    for(unsigned k=0; k<iterations;k++) {
    	Value1=ConstArray1[(i+k)%THREADS_PER_BLOCK];
    	Value2=ConstArray2[(i+k+1)%THREADS_PER_BLOCK];
        I1[i]=Value1*2+I2[i];
        I2[i]=Value2+I4[i];
        I3[i]=Value1/2+I3[i];
        I4[i]=Value2+I1[i];
 		I1[i]=I2[(i+k)%THREADS_PER_BLOCK];
 		I2[i]=I1[(i+k+1)%THREADS_PER_BLOCK];
		I3[i]=I4[(i+k)%THREADS_PER_BLOCK];
		I4[i]=I3[(i+k+1)%THREADS_PER_BLOCK];
    }		
     __syncthreads();
    
   *Value=I1[i]+I2[i]+I3[i]+I4[i];
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
 float array1[THREADS_PER_BLOCK];
 h_Value = (float *) malloc(sizeof(float));
 for(int i=0; i<THREADS_PER_BLOCK;i++){
	srand(time(0));
	array1[i] = rand() / RAND_MAX;
 }
 float array2[THREADS_PER_BLOCK];
 for(int i=0; i<THREADS_PER_BLOCK;i++){
	srand(time(0));
	array2[i] = rand() / RAND_MAX;
 }

 cudaMemcpyToSymbol(ConstArray1, array1, sizeof(float) * THREADS_PER_BLOCK );
 cudaMemcpyToSymbol(ConstArray2, array2, sizeof(float) * THREADS_PER_BLOCK );
 checkCudaErrors( cudaMalloc((void**)&d_Value, sizeof(float)) );
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

  checkCudaErrors( cudaMemcpy(h_Value, d_Value, sizeof(float), cudaMemcpyDeviceToHost) );
  checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

  return 0;
}







