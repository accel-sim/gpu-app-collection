#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>
// Includes
#include <stdio.h>
#include <string.h>
#include <cuda.h>

// includes, project
#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>
#include "../include/ContAcq-IntClk.h"

// includes CUDA
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK 256
#define LINE_SIZE 	128
#define SETS		4
#define ASSOC		24
#define SIMD_WIDTH	32

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

inline void __checkCudaErrors(cudaError err, const char *file, const int line ){
  if(cudaSuccess != err){
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
	 exit(-1);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line ){
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
	fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
	exit(-1);
  }
}

// end of CUDA Helper Functions




// Device code
#define ITERATIONS REPLACE_ITERATIONS

texture<float,1,cudaReadModeElementType> texmem1;
texture<float,1,cudaReadModeElementType> texmem2;
texture<float,1,cudaReadModeElementType> texmem3;
texture<float,1,cudaReadModeElementType> texmem4;
texture<float,1,cudaReadModeElementType> texmem5;
texture<float,1,cudaReadModeElementType> texmem6;
texture<float,1,cudaReadModeElementType> texmem7;
texture<float,1,cudaReadModeElementType> texmem9;
texture<float,1,cudaReadModeElementType> texmem8;
__constant__ float ConstArray1[THREADS_PER_BLOCK];
__constant__ float ConstArray2[THREADS_PER_BLOCK];
__constant__ float ConstArray3[THREADS_PER_BLOCK];
__constant__ float ConstArray4[THREADS_PER_BLOCK];
__constant__ float ConstArray5[THREADS_PER_BLOCK];
__constant__ float ConstArray6[THREADS_PER_BLOCK];
__constant__ float ConstArray7[THREADS_PER_BLOCK];
__constant__ float ConstArray8[THREADS_PER_BLOCK];


__global__ void tex_bm_kernel( float* out, unsigned size)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	__device__ __shared__ float I1[THREADS_PER_BLOCK];
    __device__  __shared__ float I2[THREADS_PER_BLOCK];
    __device__  __shared__ float I4[THREADS_PER_BLOCK];
    __device__  __shared__ float I5[THREADS_PER_BLOCK];
    __device__  __shared__ float I6[THREADS_PER_BLOCK];
    __device__  __shared__ float I7[THREADS_PER_BLOCK];
    __device__  __shared__ float I8[THREADS_PER_BLOCK];

	I1[tid%THREADS_PER_BLOCK] = tid;
	I2[tid%THREADS_PER_BLOCK] = tid/2;
	I4[tid%THREADS_PER_BLOCK] = tid+2;
	I5[tid%THREADS_PER_BLOCK] = 5*tid;
	I6[tid%THREADS_PER_BLOCK] = tid/2;
	I7[tid%THREADS_PER_BLOCK] = tid*10;
	I8[tid%THREADS_PER_BLOCK] = tid/2;
	
	if(tid < size){
		for(unsigned i=0; i<ITERATIONS; ++i){
			out[tid] = tex1Dfetch(texmem1,tid)* tex1Dfetch(texmem2,tid);
			out[tid*2] = ConstArray1[(tid)%THREADS_PER_BLOCK]/I5[(tid+i)%THREADS_PER_BLOCK];
			out[tid*3] =  I1[(tid+i)%THREADS_PER_BLOCK];
			out[tid*4] = tex1Dfetch(texmem4,tid)*I2[tid%THREADS_PER_BLOCK];
			out[tid*5] = ConstArray2[(tid+i)%THREADS_PER_BLOCK]+I8[(tid+i)%THREADS_PER_BLOCK];
			out[tid*6] = tex1Dfetch(texmem6,tid)/I6[(tid+i)%THREADS_PER_BLOCK];
			out[tid*7] = ConstArray5[(tid+i)%THREADS_PER_BLOCK]+I7[(tid+i)%THREADS_PER_BLOCK];
			out[tid*8] = (I4[tid%THREADS_PER_BLOCK]*123.45);
			out[tid*9] = (abs(ConstArray3[(tid+i)%THREADS_PER_BLOCK]));
		}
	}

}


////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	
	 float array1[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array1[i] = rand() / RAND_MAX;
	 }
	 float array2[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array2[i] = rand() / RAND_MAX;
	 }
	 float array3[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array3[i] = rand() / RAND_MAX;
	 }
	 float array4[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array4[i] = rand() / RAND_MAX;
	 }
	 float array5[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array5[i] = rand() / RAND_MAX;
	 }
	 float array6[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array6[i] = rand() / RAND_MAX;
	 }
	 float array7[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array7[i] = rand() / RAND_MAX;
	 }
	 float array8[THREADS_PER_BLOCK];
	 for(int i=0; i<THREADS_PER_BLOCK;i++){
		srand(time(0));
		array8[i] = rand() / RAND_MAX;
	 }
	 
	 cudaMemcpyToSymbol("ConstArray1", array1, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray2", array2, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray3", array3, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray4", array4, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray5", array5, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray6", array6, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray7", array7, sizeof(float) * THREADS_PER_BLOCK );
	 cudaMemcpyToSymbol("ConstArray8", array8, sizeof(float) * THREADS_PER_BLOCK );
	 
	
	int texmem_size = LINE_SIZE*SETS*ASSOC;	
	
	float *host_texture1 = (float*) malloc(texmem_size*sizeof(float));
	for (int i=0; i< texmem_size; i++) {
		host_texture1[i] = i;
	}
	float *device_texture1;
	float *device_texture2;
	float *device_texture3;
	float *device_texture4;
	float *device_texture5;
	float *device_texture6;
	float *device_texture7;
	float *device_texture8;
	float *device_texture9;

	float *host_out = (float*) malloc(texmem_size*sizeof(float)*10);
	float *device_out;

	cudaMalloc((void**) &device_texture1, texmem_size);
	cudaMalloc((void**) &device_texture2, texmem_size);
	cudaMalloc((void**) &device_texture3, texmem_size);
	cudaMalloc((void**) &device_texture4, texmem_size);
	cudaMalloc((void**) &device_texture5, texmem_size);
	cudaMalloc((void**) &device_texture6, texmem_size);
	cudaMalloc((void**) &device_texture7, texmem_size);
	cudaMalloc((void**) &device_texture8, texmem_size);
	cudaMalloc((void**) &device_texture9, texmem_size);

	cudaMalloc((void**) &device_out, texmem_size*10);

	cudaMemcpy(device_texture1, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture2, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture3, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture4, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture5, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture6, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture7, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture8, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_texture9, host_texture1, texmem_size*sizeof(float), cudaMemcpyHostToDevice);

	cudaBindTexture(0, texmem1, device_texture1, texmem_size);
	cudaBindTexture(0, texmem2, device_texture2, texmem_size);
	cudaBindTexture(0, texmem3, device_texture3, texmem_size);
	cudaBindTexture(0, texmem4, device_texture4, texmem_size);
	cudaBindTexture(0, texmem5, device_texture5, texmem_size);
	cudaBindTexture(0, texmem6, device_texture6, texmem_size);
	cudaBindTexture(0, texmem7, device_texture7, texmem_size);
	cudaBindTexture(0, texmem8, device_texture8, texmem_size);
	cudaBindTexture(0, texmem9, device_texture9, texmem_size);


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

	/*
	printf("Output: ");
	float error = false;
	for (int i=0; i< texmem_size; i++){
		printf("%.1f ", host_out[i]);
		if (host_out[i] - i > 0.0001) error = true;
	}
	printf("\n");
	if (error) printf("\nFAILED\n");
	else printf("\nPASSED\n");
	*/
}

void CleanupResources(void){
  // Free device memory


}

// Allocates an array with random float entries.
void RandomInit(int* data, int n){
  for (int i = 0; i < n; ++i)
	data[i] = (int)(rand() / RAND_MAX);
}






