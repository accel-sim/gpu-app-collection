#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
// Includes
#include <stdio.h>
#include "../include/ContAcq-IntClk.h"

// includes, project
#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 60
#define ITERATIONS REPLACE_ITERATIONS

// Variables

__constant__ float ConstArray1[THREADS_PER_BLOCK];
__constant__ float ConstArray2[THREADS_PER_BLOCK];
float* h_Value;
float* d_Value;
bool noprompt = false;
unsigned int my_timer;

// Functions
void CleanupResources(void);
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




// Device code
__global__ void PowerKernal(float* Value)
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
    for(unsigned k=0; k<ITERATIONS;k++) {
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

int main()
{
 printf("Power Microbenchmarks\n");
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

 cudaMemcpyToSymbol("ConstArray1", array1, sizeof(float) * THREADS_PER_BLOCK );
 cudaMemcpyToSymbol("ConstArray2", array2, sizeof(float) * THREADS_PER_BLOCK );
 checkCudaErrors( cudaMalloc((void**)&d_Value, sizeof(float)) );
 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);

	CUT_SAFE_CALL(cutCreateTimer(&my_timer)); 
	TaskHandle taskhandle = LaunchDAQ();
	CUT_SAFE_CALL(cutStartTimer(my_timer)); 
 PowerKernal<<<dimGrid,dimBlock>>>(d_Value);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	printf("execution time = %f\n", cutGetTimerValue(my_timer));
	TurnOffDAQ(taskhandle, cutGetTimerValue(my_timer));
	CUT_SAFE_CALL(cutStopTimer(my_timer));
	CUT_SAFE_CALL(cutDeleteTimer(my_timer)); 
 
 getLastCudaError("kernel launch failure");

 checkCudaErrors( cudaMemcpy(h_Value, d_Value, sizeof(float), cudaMemcpyDeviceToHost) );
#ifdef _DEBUG
 checkCudaErrors( cudaDeviceSynchronize() );
#endif


 return 0;
}







