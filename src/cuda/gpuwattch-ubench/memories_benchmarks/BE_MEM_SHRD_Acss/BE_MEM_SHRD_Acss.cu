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
unsigned* h_C1;
float* h_C2;
unsigned* d_C1;
float* d_C2;
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
__global__ void PowerKernal(unsigned* C1, float* C2, int N)
{
    int i = threadIdx.x;
    //Do Some Computation
   __device__  __shared__ unsigned I1[THREADS_PER_BLOCK];
   __device__  __shared__ unsigned I2[THREADS_PER_BLOCK];
   __device__ __shared__ float I3[THREADS_PER_BLOCK];
   __device__  __shared__ float I4[THREADS_PER_BLOCK];
	
    I1[i]=i*2;
    I2[i]=i;
    I3[i]=i/2;
    I4[i]=i;

    __syncthreads();

    for(unsigned k=0; k<ITERATIONS ;k++) {
        		I1[i]=I2[(i+k)%THREADS_PER_BLOCK];
        		I2[i]=I1[(i+k+1)%THREADS_PER_BLOCK];
    }		
         
    for(unsigned k=0; k<ITERATIONS ;k++) {
    			I3[i]=I4[(i+k)%THREADS_PER_BLOCK];
    			I4[i]=I3[(i+k+1)%THREADS_PER_BLOCK];
    } 
    //C1[i]=I2[i];
    //C2[i]=I4[i];
    __syncthreads();

}


// Host code

int main()
{
 printf("Power Microbenchmarks\n");
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size1 = N * sizeof(unsigned);
 size_t size2 = N * sizeof(float); 

 // Allocate vectors in device memory
 h_C1 = (unsigned *) malloc(size1);
 h_C2 = (float *) malloc(size2);
 checkCudaErrors( cudaMalloc((void**)&d_C1, size1) );
 checkCudaErrors( cudaMalloc((void**)&d_C2, size2) ); 

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);

	CUT_SAFE_CALL(cutCreateTimer(&my_timer)); 
	TaskHandle taskhandle = LaunchDAQ();
	CUT_SAFE_CALL(cutStartTimer(my_timer)); 

 PowerKernal<<<dimGrid,dimBlock>>>(d_C1, d_C2, N);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	printf("execution time = %f\n", cutGetTimerValue(my_timer));
	TurnOffDAQ(taskhandle, cutGetTimerValue(my_timer));
	CUT_SAFE_CALL(cutStopTimer(my_timer));
	CUT_SAFE_CALL(cutDeleteTimer(my_timer)); 

 getLastCudaError("kernel launch failure");

#ifdef _DEBUG
 checkCudaErrors( cudaDeviceSynchronize() );
#endif

 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 checkCudaErrors( cudaMemcpy(h_C1, d_C1, size1, cudaMemcpyDeviceToHost) );
 checkCudaErrors( cudaMemcpy(h_C2, d_C2, size2, cudaMemcpyDeviceToHost) );

 CleanupResources();

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_C1)
	cudaFree(d_C1);
  if (d_C2)
	cudaFree(d_C2);


  // Free host memory
  if (h_C1)
	free(h_C1);
  if (d_C2)
	cudaFree(d_C2);

}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}






