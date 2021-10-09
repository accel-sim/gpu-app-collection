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



texture<float,1,cudaReadModeElementType> texmem1;
texture<float,1,cudaReadModeElementType> texmem2;
texture<float,1,cudaReadModeElementType> texmem3;
texture<float,1,cudaReadModeElementType> texmem4;


// Functions
void CleanupResources(void);
void RandomInit_int(unsigned*, int);
void RandomInit_fp(float*, int);

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
__global__ void PowerKernal1(float *A, float *B, int N, int iterations)
{
	int tid = blockIdx.x*blockIdx.x + threadIdx.x;
    float Value1=0;
    float Value2=0;
   __device__ __shared__ float I1[THREADS_PER_BLOCK];
   __device__  __shared__ float I2[THREADS_PER_BLOCK];

   I1[tid%THREADS_PER_BLOCK] = A[tid];
   I2[tid%THREADS_PER_BLOCK] = B[tid];
   __syncthreads();

	float sum = 0.0;

	if(tid < N){
		for(unsigned i=0; i<iterations; ++i){

			sum = tex1Dfetch(texmem1,tid)+B[tid];

			for(unsigned j=1; j<=2; ++j){
				sum+=tex1Dfetch(texmem1,tid*j);
				Value1 += log2(sum);
				Value2 += Value1 + log2(I1[(i+j)%THREADS_PER_BLOCK]);
				sum+=tex1Dfetch(texmem2,tid*j);
				sum+=tex1Dfetch(texmem3,tid*j);
				Value1 += log2(sum);
				Value2 += Value1 + log2(I2[(i+j)%THREADS_PER_BLOCK]);
				sum+=tex1Dfetch(texmem4,tid*j);
			}
			A[tid*2] = sum;
			B[tid] = A[tid*2]+A[tid];
		}
	}
}

// Host code
float *h_A1, *h_A2, *h_A3;
float *d_A1, *d_A2, *d_A3;

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
	 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS*2;
	 
	 // Allocate input vectors h_A and h_B in host memory
 	 size_t size1 = N * sizeof(float);
	 h_A1 = (float*)malloc(size1);
	 if (h_A1 == 0) CleanupResources();

	 h_A2 = (float*)malloc(size1);
	 if (h_A2 == 0) CleanupResources();



	float *host_texture1 = (float*) malloc(size1);
	for (int i=0; i< N; i++) {
		host_texture1[i] = i;
	}
	float *device_texture1;
	float *device_texture2;
	float *device_texture3;
	float *device_texture4;


	cudaMalloc((void**) &device_texture1, size1);
	cudaMalloc((void**) &device_texture2, size1);
	cudaMalloc((void**) &device_texture3, size1);
	cudaMalloc((void**) &device_texture4, size1);


	cudaMemcpy(device_texture1, host_texture1, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture2, host_texture1, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture3, host_texture1, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture4, host_texture1, size1, cudaMemcpyHostToDevice);


	cudaBindTexture(0, texmem1, device_texture1, size1);
	cudaBindTexture(0, texmem2, device_texture2, size1);
	cudaBindTexture(0, texmem3, device_texture3, size1);
	cudaBindTexture(0, texmem4, device_texture4, size1);


	 dim3 dimGrid2(1,1);
	 dim3 dimBlock2(1,1);

	 // Initialize input vectors
	 RandomInit_fp(h_A1, N);
	 RandomInit_fp(h_A2, N);



	 // Allocate vectors in device memory
	 checkCudaErrors( cudaMalloc((void**)&d_A1, size1) );
	 checkCudaErrors( cudaMalloc((void**)&d_A2, size1) );


	 // Copy vectors from host memory to device memory
	 checkCudaErrors( cudaMemcpy(d_A1, h_A1, size1, cudaMemcpyHostToDevice) );
	 checkCudaErrors( cudaMemcpy(d_A2, h_A2, size1, cudaMemcpyHostToDevice) );

	 dim3 dimGrid(NUM_OF_BLOCKS,1);
	 dim3 dimBlock(THREADS_PER_BLOCK,1);


	cudaEvent_t start, stop;
  	float elapsedTime = 0;
 	checkCudaErrors(cudaEventCreate(&start));
  	checkCudaErrors(cudaEventCreate(&stop));

  	checkCudaErrors(cudaEventRecord(start));
	PowerKernal1<<<dimGrid,dimBlock>>>(d_A1, d_A2, N, iterations);
	checkCudaErrors(cudaEventRecord(stop));

	//PowerKernalEmpty<<<dimGrid2,dimBlock2>>>(d_A3, iterations);
	checkCudaErrors(cudaEventSynchronize(stop));
  	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  	printf("gpu execution time = %.2f s\n", elapsedTime/1000);
  	getLastCudaError("kernel launch failure");
  	cudaThreadSynchronize();
  	checkCudaErrors(cudaEventDestroy(start));
  	checkCudaErrors(cudaEventDestroy(stop));
	CleanupResources();

	return 0;
}

void CleanupResources(void)
{
	  // Free device memory
	  if (d_A1)
		cudaFree(d_A1);
	  if (d_A2)
		cudaFree(d_A2);
	  if (d_A3)
		cudaFree(d_A3);
	  // Free host memory
	  if (h_A1)
		free(h_A1);
	  if (h_A2)
		free(h_A2);
	  if (h_A3)
		free(h_A3);
}

// Allocates an array with random float entries.
void RandomInit_int(float* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}

void RandomInit_fp(float* data, int n)
{
   for (int i = 0; i < n; ++i){
	data[i] = rand() / RAND_MAX;
   }
}






