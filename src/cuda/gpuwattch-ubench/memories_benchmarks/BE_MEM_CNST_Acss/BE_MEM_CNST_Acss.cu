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

__constant__ unsigned ConstArray1[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray2[THREADS_PER_BLOCK];
unsigned* h_Value;
unsigned* d_Value;
bool noprompt = false;
unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(unsigned*, int);
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
__global__ void PowerKernal(unsigned* Value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned Value1;
    unsigned Value2;
    for(unsigned k=0; k<ITERATIONS;k++) {
	Value1=ConstArray1[(i+k)%THREADS_PER_BLOCK];
	Value2=ConstArray2[(i+k+1)%THREADS_PER_BLOCK];
    }		
         
   
    __syncthreads();
   *Value=Value1+Value2;
}


// Host code

int main()
{
 printf("Power Microbenchmarks\n");
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

 cudaMemcpyToSymbol("ConstArray1", array1, sizeof(unsigned) * THREADS_PER_BLOCK );
 cudaMemcpyToSymbol("ConstArray2", array2, sizeof(unsigned) * THREADS_PER_BLOCK );
 checkCudaErrors( cudaMalloc((void**)&d_Value, sizeof(unsigned)) );
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

 checkCudaErrors( cudaMemcpy(h_Value, d_Value, sizeof(unsigned), cudaMemcpyDeviceToHost) );
#ifdef _DEBUG
 checkCudaErrors( cudaDeviceSynchronize() );
#endif


 return 0;
}







