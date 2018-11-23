#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
// Includes
#include <stdio.h>
#include <../include/repeat.h>
#include "../include/ContAcq-IntClk.h"
// includes, project
#include <../include/sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 60
#define ITERATIONS REPLACE_ITERATIONS
__constant__ unsigned ConstArray1[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray2[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray3[THREADS_PER_BLOCK];
__constant__ unsigned ConstArray4[THREADS_PER_BLOCK];


unsigned* h_Value;
unsigned* d_Value;
bool noprompt = false;
unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(unsigned*, int);
void ParseArguments(int, char**);

FILE *fp;
// Functions
void CleanupResources(void);
void RandomInit(int*, int);
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

texture<float,1,cudaReadModeElementType> texmem1;
texture<float,1,cudaReadModeElementType> texmem2;
texture<float,1,cudaReadModeElementType> texmem3;


__global__ void PowerKernal(unsigned* Value)
{
	int i = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

	unsigned Value1=0;
	unsigned Value2=0;
	unsigned Value3=0;
	unsigned Value4=0;
    for(unsigned k=0; k<ITERATIONS;k++) {
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

	 cudaMemcpyToSymbol("ConstArray1", array1, sizeof(unsigned) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray2", array2, sizeof(unsigned) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray3", array3, sizeof(unsigned) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray4", array4, sizeof(unsigned) * THREADS_PER_BLOCK );
	 
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
}






