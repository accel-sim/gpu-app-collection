#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
// Includes
#include <stdio.h>

// includes, project
#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 60
#define ITERATIONS REPLACE_ITERATIONS
#include "../include/ContAcq-IntClk.h"

// Variables
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;
bool noprompt = false;
unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(float*, int);
void ParseArguments(int, char**);

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
__global__ void PowerKernal2(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    float Value1;
    float Value2;
    float Value3;
    float Value;
    float I1=A[i];
    float I2=B[i];

    // Excessive Addition access
    for(unsigned k=0; k<ITERATIONS;k++) {
	Value1=I1*I2;
	Value3=I1*I2;
	Value1*=Value2;
	Value1*=Value2;
	Value2=Value3*Value1;
	Value1=Value2*Value3;
//	Value1=I1*I2;
//	Value3=Value1*I1;
//	Value2=Value3*Value1;
//	Value3*=Value2;
//	Value1*=Value2;
//        Value3*=Value1;
    }
    __syncthreads();

    Value=Value1;
    C[i]=Value*Value2;
    __syncthreads();

}

int main()
{
 printf("Power Microbenchmarks\n");
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size = N * sizeof(float);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (float*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (float*)malloc(size);
 if (h_B == 0) CleanupResources();
 h_C = (float*)malloc(size);
 if (h_C == 0) CleanupResources();

 // Initialize input vectors
 RandomInit(h_A, N);
 RandomInit(h_B, N);

 // Allocate vectors in device memory
printf("before\n");
 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
 checkCudaErrors( cudaMalloc((void**)&d_B, size) );
 checkCudaErrors( cudaMalloc((void**)&d_C, size) );
printf("after\n");

 // Copy vectors from host memory to device memory
 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
 checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);
 dim3 dimGrid2(1,1);
 dim3 dimBlock2(1,1);
CUT_SAFE_CALL(cutCreateTimer(&my_timer)); 
TaskHandle taskhandle = LaunchDAQ();
CUT_SAFE_CALL(cutStartTimer(my_timer)); 
printf("execution time = %f\n", cutGetTimerValue(my_timer));



PowerKernal2<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N);
CUDA_SAFE_CALL( cudaThreadSynchronize() );
printf("execution time = %f\n", cutGetTimerValue(my_timer));


getLastCudaError("kernel launch failure");
CUDA_SAFE_CALL( cudaThreadSynchronize() );
CUT_SAFE_CALL(cutStopTimer(my_timer));
TurnOffDAQ(taskhandle, cutGetTimerValue(my_timer));
printf("execution time = %f\n", cutGetTimerValue(my_timer));
CUT_SAFE_CALL(cutDeleteTimer(my_timer)); 

#ifdef _DEBUG
 checkCudaErrors( cudaDeviceSynchronize() );
#endif

 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
 
 CleanupResources();

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
	cudaFree(d_A);
  if (d_B)
	cudaFree(d_B);
  if (d_C)
	cudaFree(d_C);

  // Free host memory
  if (h_A)
	free(h_A);
  if (h_B)
	free(h_B);
  if (h_C)
	free(h_C);

}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
  for (int i = 0; i < n; ++i){ 
	data[i] = rand() / RAND_MAX;
  }
}






