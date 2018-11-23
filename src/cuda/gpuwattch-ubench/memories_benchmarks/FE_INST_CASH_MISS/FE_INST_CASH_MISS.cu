#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
// Includes
#include <stdio.h>
#include <repeat.h>
// includes, project
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>
#include "../include/ContAcq-IntClk.h"

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 60
#define ITERATIONS REPLACE_ITERATIONS
// Variables
unsigned* h_A;
unsigned* h_C;
unsigned* d_A;
unsigned* d_C;
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
__global__ void PowerKernal(const unsigned* A,unsigned *C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned I1=A[i];

    //Excessive Logical Unit access
    for(unsigned k=0; k<ITERATIONS*(blockDim.x+300);k++) {
    // BLOCK-0 (For instruction size of 8 bytes and Block size of 32B
    	__asm volatile (	
    	"B0: bra.uni B1;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B1: bra.uni B2;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")
    			
    	"B2: bra.uni B3;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B3: bra.uni B4;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B4: bra.uni B5;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B5: bra.uni B6;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B6: bra.uni B7;\n\t" 
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B7: bra.uni B8;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B8: bra.uni B9;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B9: bra.uni B10;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B10: bra.uni B11;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B11: bra.uni B12;\n\t" 
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B12: bra.uni B13;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B13: bra.uni B14;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B14: bra.uni B15;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B15: bra.uni B16;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B16: bra.uni B17;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B17: bra.uni B18;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")


    	"B18: bra.uni B19;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B19: bra.uni B20;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B20: bra.uni B21;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B21: bra.uni B22;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B22: bra.uni B23;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B23: bra.uni B24;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B24: bra.uni B25;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B25: bra.uni B26;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B26: bra.uni B27;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B27: bra.uni B28;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")


    	"B28: bra.uni B29;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	"B29: bra.uni B30;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")

    	  
    	"B30: bra.uni B31;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")
    	
    	"B31: bra.uni LOOP;\n\t"
    	repeat31("add.u32   %0, %1, 1;\n\t")
    	"LOOP:"
    	: "=r"(I1) : "r"(I1));
    	
    }
    C[i]=I1;
    __syncthreads();

}


// Host code

int main()
{
 printf("Power Microbenchmarks\n");
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size = N * sizeof(unsigned);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (unsigned*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_C = (unsigned*)malloc(size);
 if (h_C == 0) CleanupResources();

 // Initialize input vectors
 RandomInit(h_A, N);

 // Allocate vectors in device memory
 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
 checkCudaErrors( cudaMalloc((void**)&d_C, size) );

 // Copy vectors from host memory to device memory
 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);


CUT_SAFE_CALL(cutCreateTimer(&my_timer)); 
TaskHandle taskhandle = LaunchDAQ();
CUT_SAFE_CALL(cutStartTimer(my_timer)); 
 PowerKernal<<<dimGrid,dimBlock>>>(d_A,d_C);

CUDA_SAFE_CALL( cudaThreadSynchronize() );
CUT_SAFE_CALL(cutStopTimer(my_timer));
TurnOffDAQ(taskhandle, cutGetTimerValue(my_timer));
printf("execution time = %f\n", cutGetTimerValue(my_timer));
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

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
	cudaFree(d_A);
  if (d_C)
	cudaFree(d_C);

  // Free host memory
  if (h_A)
	free(h_A);
  if (h_C)
	free(h_C);

}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}






