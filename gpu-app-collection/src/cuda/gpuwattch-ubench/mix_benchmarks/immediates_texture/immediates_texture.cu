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

#define MAX_THREADS_PER_BLOCK 256

#define LINE_SIZE 	128
#define SETS		4
#define ASSOC		24
#define SIMD_WIDTH	32
#define ITERATIONS REPLACE_ITERATIONS
// Variables
int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

bool noprompt = false;
unsigned int my_timer;

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


__global__ void tex_bm_kernel( float* out, unsigned size)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

	float I1=0;
	float I2=0;
	if(tid < size){
		for(unsigned i=0; i<ITERATIONS; ++i){
			out[tid] = tex1Dfetch(texmem1,tid);
			out[tid*2] = tex1Dfetch(texmem2,tid);
			I1=out[tid];
			I2=out[tid*2];
			__asm volatile (
	    	repeat5("add.f32   %0, %1, 1.0;\n\t")
	    	: "=f"(I1) : "f"(I2));
	    	__asm volatile (
	    	repeat5("mul.f32   %0, %1, 5.0;\n\t")
	    	: "=f"(I2) : "f"(I1));
	    	
			out[tid*3] = tex1Dfetch(texmem1,tid)+I1;
			out[tid*4] = tex1Dfetch(texmem2,tid)+I2;
		}
	}

}



// Host code

int main()
{
	
 int texmem_size = LINE_SIZE*SETS*ASSOC;

 float *host_texture1 = (float*) malloc(texmem_size*sizeof(float));
 for (int i=0; i< texmem_size; i++) {
	host_texture1[i] = i;
 }
 float *device_texture1;
 float *device_texture2;
 float *device_texture3;	
 float *host_out = (float*) malloc(texmem_size*sizeof(float)*10);
 float *device_out;
 
 
 cudaMalloc((void**) &device_texture1, texmem_size);
 cudaMalloc((void**) &device_texture2, texmem_size);
 cudaMalloc((void**) &device_texture3, texmem_size);	

 cudaMalloc((void**) &device_out, texmem_size*5);

 cudaMemcpy(device_texture1, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(device_texture2, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(device_texture3, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
 
 cudaBindTexture(0, texmem1, device_texture1, texmem_size);
 cudaBindTexture(0, texmem2, device_texture2, texmem_size);
 cudaBindTexture(0, texmem3, device_texture3, texmem_size);
 
 unsigned num_blocks = (texmem_size / MAX_THREADS_PER_BLOCK) + 1;
 dim3  grid( num_blocks, 1, 1);
 dim3  threads( MAX_THREADS_PER_BLOCK, 1, 1);

 CUT_SAFE_CALL(cutCreateTimer(&my_timer));
 TaskHandle taskhandle = LaunchDAQ();
 CUT_SAFE_CALL(cutStartTimer(my_timer));

 tex_bm_kernel<<< grid, threads, 0 >>>(device_out, texmem_size);
 cudaThreadSynchronize();

 CUT_SAFE_CALL(cutStopTimer(my_timer));
 TurnOffDAQ(taskhandle, cutGetTimerValue(my_timer));
 printf("execution time = %f\n", cutGetTimerValue(my_timer));
 CUT_SAFE_CALL(cutDeleteTimer(my_timer));


 printf("Kernel DONE, probably correctly\n");
 cudaMemcpy(host_out, device_out, texmem_size*sizeof(float), cudaMemcpyDeviceToHost);
 
 return 0;
}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}






