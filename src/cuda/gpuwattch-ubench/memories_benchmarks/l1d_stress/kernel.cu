#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>
// Includes
#include <stdio.h>
#include "../include/ContAcq-IntClk.h"

// includes, project
#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

#include "l1d.h"

#define THREADS_PER_BLOCK 32
#define NUM_OF_BLOCKS 1
#define ITERATIONS REPLACE_ITERATIONS
#define LINE_SIZE 	128
#define SETS		32
#define ASSOC		4
#define SIMD_WIDTH	32


// Variables
int* h_A;
int* h_B;
int* h_C;
int* d_A;
int* d_B;
int* d_C;
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

#define CONFIG 50
// Device code
__global__ static void PowerKernal(int* A, int* C, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation

    int size = (LINE_SIZE*ASSOC*SETS)/sizeof(int);

    unsigned j=0, k=0;
    int m_sum=0;
	// Fill the L1 cache, Miss on first LD, Hit on subsequent LDs

#if (CONFIG==5 || CONFIG==10 || CONFIG==50 || CONFIG==100)
	for(k=0; k<ITERATIONS; ++k){
		for(j=0; (j + (THREADS_PER_BLOCK*CONFIG)) < size; j+=THREADS_PER_BLOCK){
#if CONFIG==5
			ld5_add5(m_sum, A, j);
#elif CONFIG==10
			ld10(m_sum, A, j);
#elif CONFIG==50
			ld50_add50(m_sum, A, j);
#elif CONFIG==100
			ld100(m_sum, A, j);
#endif
		}
	}
#else
	for(k=0; k<ITERATIONS; ++k){
		for(j=0; j < size; j+=THREADS_PER_BLOCK){

		}
	}
#endif
	C[tid]=m_sum;

/*
    int dest_reg = 0;
    asm_ld(dest_reg, A, tid);
	C[tid]=dest_reg;

 */


/*
    int a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
    if(tid == 0){
		a1=tid;
		a2=tid+4;
		a3=tid+8;
		a4=tid+12;
		a5=tid+16;
		a6=tid+20;
		a7=tid+24;
		a8=tid+28;
		a9=tid+32;
		a10=tid+36;
		a11=tid+40;
		a12=tid+44;
		a13=tid+48;
		a14=tid+52;
		a15=tid+56;
		a16=tid+60;
		a17=tid+64;
		a18=tid+68;
		a19=tid+72;
		a20=tid+76;
		a21=tid+80;
		a22=tid+84;
		a23=tid+88;
		a24=tid+92;
		a25=tid+96;
		a26=tid+100;
		a27=tid+104;
		a28=tid+108;
		a29=tid+112;
		a30=tid+116;
		a31=tid+120;

		dest_reg += A[a1];
		dest_reg += A[a2];
		dest_reg += A[a3];
		dest_reg += A[a4];
		dest_reg += A[a5];
		dest_reg += A[a6];
		dest_reg += A[a7];
		dest_reg += A[a8];
		dest_reg += A[a9];
		dest_reg += A[a10];
		dest_reg += A[a11];
		dest_reg += A[a12];
		dest_reg += A[a13];
		dest_reg += A[a14];
		dest_reg += A[a15];
		dest_reg += A[a16];
		dest_reg += A[a17];
		dest_reg += A[a18];
		dest_reg += A[a19];
		dest_reg += A[a20];
		dest_reg += A[a21];
		dest_reg += A[a22];
		dest_reg += A[a23];
		dest_reg += A[a24];
		dest_reg += A[a25];
		dest_reg += A[a26];
		dest_reg += A[a27];
		dest_reg += A[a28];
		dest_reg += A[a29];
		dest_reg += A[a30];
		dest_reg += A[a31];

		dest_reg += A[a1];
		dest_reg += A[a2];
		dest_reg += A[a3];
		dest_reg += A[a4];
		dest_reg += A[a5];
		dest_reg += A[a6];
		dest_reg += A[a7];
		dest_reg += A[a8];
		dest_reg += A[a9];
		dest_reg += A[a10];
		dest_reg += A[a11];
		dest_reg += A[a12];
		dest_reg += A[a13];
		dest_reg += A[a14];
		dest_reg += A[a15];
		dest_reg += A[a16];
		dest_reg += A[a17];
		dest_reg += A[a18];
		dest_reg += A[a19];
		dest_reg += A[a20];
		dest_reg += A[a21];
		dest_reg += A[a22];
		dest_reg += A[a23];
		dest_reg += A[a24];
		dest_reg += A[a25];
		dest_reg += A[a26];
		dest_reg += A[a27];
		dest_reg += A[a28];
		dest_reg += A[a29];
		dest_reg += A[a30];
		dest_reg += A[a31];


		C[tid]=dest_reg;
    }
*/


}


// Host code

int main(){

	 printf("Power Microbenchmarks\n");
	 //int N = LINE_SIZE*SETS*ASSOC;
	 int N = NUM_OF_BLOCKS*THREADS_PER_BLOCK;
	 size_t size = N * sizeof(int) * NUM_OF_BLOCKS;

	 // Allocate input vectors h_A and h_B in host memory
	 h_A = (int*)malloc(size);
	 if (h_A == 0) CleanupResources();
	 //h_B = (float*)malloc(size);
	 //if (h_B == 0) CleanupResources();
	 h_C = (int*)malloc(size);
	 if (h_C == 0) CleanupResources();

	 // Initialize input vectors
	 RandomInit(h_A, N);
	 //RandomInit(h_B, N);

	 // Allocate vectors in device memory
	 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
	 //checkCudaErrors( cudaMalloc((void**)&d_B, size) );
	 checkCudaErrors( cudaMalloc((void**)&d_C, size) );

	 // Copy vectors from host memory to device memory
	 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
	 //checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

	 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	 dim3 dimGrid(NUM_OF_BLOCKS,1);
	 dim3 dimBlock(THREADS_PER_BLOCK,1);

	CUT_SAFE_CALL(cutCreateTimer(&my_timer));
	TaskHandle taskhandle = LaunchDAQ();
	CUT_SAFE_CALL(cutStartTimer(my_timer));

	 PowerKernal<<<dimGrid,dimBlock>>>(d_A, d_C, N);

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
	 checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );

	 CleanupResources();

	 return 0;
}

void CleanupResources(void){
  // Free device memory
  if (d_A)
	cudaFree(d_A);
  //if (d_B)
//	cudaFree(d_B);
  if (d_C)
	cudaFree(d_C);

  // Free host memory
  if (h_A)
	free(h_A);
 // if (h_B)
//	free(h_B);
  if (h_C)
	free(h_C);

}

// Allocates an array with random float entries.
void RandomInit(int* data, int n){
  for (int i = 0; i < n; ++i)
	data[i] = (int)(rand() / RAND_MAX);
}






