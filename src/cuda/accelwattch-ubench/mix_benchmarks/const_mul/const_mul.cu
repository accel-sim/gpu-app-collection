// Includes
#include <stdio.h>
#include <stdlib.h>


// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 640

// Variables

__constant__ float ConstArray1[THREADS_PER_BLOCK];
__constant__ float ConstArray2[THREADS_PER_BLOCK];
__constant__ float ConstArray3[THREADS_PER_BLOCK];

float* h_Value;
float* d_Value;

// Functions
void CleanupResources(void);


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
__global__ void PowerKernal(float* Value, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    //Do Some Computation
    float Value1;
    float Value2;
    float Value3;
    *Value=1;
    for(unsigned k=0; k<iterations;k++) {
    	Value1=ConstArray1[(i+k)%THREADS_PER_BLOCK];
    	Value2=ConstArray2[(i+k+1)%THREADS_PER_BLOCK];
    	Value3=ConstArray3[(i+k+5)%THREADS_PER_BLOCK];
    	Value2*=Value1;
    	Value3*=Value2;
    	*Value*=Value3;
    }		
     __syncthreads();
    
}


// Host code

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
 float array3[THREADS_PER_BLOCK];
 for(int i=0; i<THREADS_PER_BLOCK;i++){
	srand(time(0));
	array3[i] = rand() / RAND_MAX;
 }

 cudaMemcpyToSymbol(ConstArray1, array1, sizeof(float) * THREADS_PER_BLOCK );
 cudaMemcpyToSymbol(ConstArray2, array2, sizeof(float) * THREADS_PER_BLOCK );
 cudaMemcpyToSymbol(ConstArray3, array3, sizeof(float) * THREADS_PER_BLOCK );
 
 checkCudaErrors( cudaMalloc((void**)&d_Value, sizeof(float)) );
 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);

  cudaEvent_t start, stop;
  float elapsedTime = 0;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));
  PowerKernal<<<dimGrid,dimBlock>>>(d_Value, iterations);
  checkCudaErrors(cudaEventRecord(stop));
 
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("gpu execution time = %.2f s\n", elapsedTime/1000);
  getLastCudaError("kernel launch failure");
  cudaThreadSynchronize();
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  return 0;
}







