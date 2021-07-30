#include <stdio.h>
#include <stdlib.h>
// Includes
#include <stdio.h>

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 240

// Variables
unsigned* h_A;
unsigned* h_B;
unsigned* h_C;
unsigned* d_A;
unsigned* d_B;
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
__global__ void PowerKernal1(const unsigned* A, const unsigned* B, unsigned* C, int N, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned Value1=0;
    unsigned Value2=0;
    unsigned Value3=0;
    unsigned Value=0;
    unsigned I1=A[i];
    unsigned I2=B[i];


    //Excessive Logical Unit access
    for(unsigned k=0; k<iterations * ( blockDim.x+200 );k++) {

		Value1=I1 & I2;
		Value2 |= (I1 | I2);
		Value3=I1^Value2;
		Value2|=Value1;
		Value2=Value3 & Value2;
		Value1=Value2 ^ Value3;

    }

    __syncthreads();
    Value=Value1;

    C[i]=Value;
    __syncthreads();

}

__global__ void PowerKernal2(const unsigned* A, const unsigned* B, unsigned* C, int N, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned Value1=0;
    unsigned Value2=0;
    unsigned Value3=0;
    unsigned Value=0;
    unsigned I1=A[i];
    unsigned I2=B[i];


    // Excessive Addition access
    for(unsigned k=0; k<iterations * ( blockDim.x+200 );k++) {

		Value2= I1+I2;
		Value3=I1-I2;
		Value1-=Value2;
		Value3+=Value1;
		Value2-=Value3;
		Value1+=Value3;

    }
    __syncthreads();
 
    Value=Value1;

    C[i]=Value;
    __syncthreads();

}

__global__ void PowerKernal3(const unsigned* A, const unsigned* B, unsigned* C, int N, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned Value1;
    unsigned Value2;
    unsigned Value3;
    unsigned Value;
    unsigned I1=A[i];
    unsigned I2=B[i];

    // Excessive Multiplication
    for(unsigned k=0; k<iterations*( blockDim.x+200 );k++) {
    	Value1=I1*I2;
    	Value1*=Value2;
    	Value3=Value1*I2;
    	Value2*=I1*Value3;
    	Value1*=Value2;
        Value3*=Value1;
    }

    __syncthreads();
    Value=Value3;

    C[i]=Value;
    __syncthreads();

}

__global__ void PowerKernal4(const unsigned* A, const unsigned* B, unsigned* C, int N, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned Value1=0;
    unsigned Value2=0;
    unsigned Value3=1000;
    unsigned Value=0;
    unsigned I1=A[i];
    unsigned I2=B[i];


    __syncthreads();
   // Excessive Mod/Div Operations
    for(unsigned k=0; k<iterations*(blockDim.x+20);k++) {

    	Value1=I1/(I2+1);
    	Value2=Value1/(I2+1);
    	Value3/= (I1/(I2+1) +1);
    	Value1/=(Value2+1);
    	Value3%=(Value2+1);
    	Value2/=(Value3+1);
        Value1%=(Value+1);
        Value3/=(Value1+1);
    }

    __syncthreads();
    Value=Value3;

    C[i]=Value;
    __syncthreads();

}

__global__ void PowerKernalEmpty(const unsigned* A, const unsigned* B, unsigned* C, int N, int iterations)
// Host code
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    unsigned Value1=0;
    unsigned Value2=0;
    unsigned Value3=0;
    unsigned Value=0;
    unsigned I1=A[i];
    unsigned I2=B[i];
	unsigned m;
    

    __syncthreads();
   // Excessive Mod/Div Operations
    for(unsigned long k=0; k<iterations*(blockDim.x + 299);k++) {
    	//Value1=(I1)+k;
        //Value2=(I2)+k;
        //Value3=(Value2)+k;
        //Value2=(Value1)+k;
       	__asm volatile ("{\n\t"
        			"B0: bra.uni B1;\n\t"
        			"B1: bra.uni B2;\n\t"
        			"B2: bra.uni B3;\n\t"
        			"B3: bra.uni B4;\n\t"
        			"B4: bra.uni B5;\n\t"
        			"B5: bra.uni B6;\n\t"
        			"B6: bra.uni B7;\n\t"
        			"B7: bra.uni B8;\n\t"
        			"B8: bra.uni B9;\n\t"
        			"B9: bra.uni B10;\n\t"
        			"B10: bra.uni B11;\n\t"
        			"B11: bra.uni B12;\n\t"
        			"B12: bra.uni B13;\n\t"
        			"B13: bra.uni B14;\n\t"
        			"B14: bra.uni B15;\n\t"
        			"B15: bra.uni B16;\n\t"
        			"B16: bra.uni B17;\n\t"
        			"B17: bra.uni B18;\n\t"
        			"B18: bra.uni B19;\n\t"
        			"B19: bra.uni B20;\n\t"
        			"B20: bra.uni B21;\n\t"
        			"B21: bra.uni B22;\n\t"
        			"B22: bra.uni B23;\n\t"
        			"B23: bra.uni B24;\n\t"
        			"B24: bra.uni B25;\n\t"
        			"B25: bra.uni B26;\n\t"
        			"B26: bra.uni B27;\n\t"
        			"B27: bra.uni B28;\n\t"
        			"B28: bra.uni B29;\n\t"
        			"B29: bra.uni B30;\n\t"
        			"B30: bra.uni B31;\n\t"
        			"B31: bra.uni LOOP;\n\t"
        			"LOOP:\n\t"
					"}"
        			);

    }


    C[i]=I1;
    __syncthreads();

}

int main(int argc, char** argv)
{
	int iterations;
	if (argc != 2){
		fprintf(stderr,"usage: %s #iterations\n",argv[0]);
		exit(1);
	}
	else {
		iterations = atoi(argv[1]);
	}
	printf("Power Microbenchmarks\n");
	int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
	size_t size = N * sizeof(unsigned);
	// Allocate input vectors h_A and h_B in host memory
	h_A = (unsigned*)malloc(size);
	if (h_A == 0) CleanupResources();
	h_B = (unsigned*)malloc(size);
	if (h_B == 0) CleanupResources();
	h_C = (unsigned*)malloc(size);
	if (h_C == 0) CleanupResources();

	// Initialize input vectors
	RandomInit(h_A, N);
	RandomInit(h_B, N);

	// Allocate vectors in device memory
	checkCudaErrors( cudaMalloc((void**)&d_A, size) );
	checkCudaErrors( cudaMalloc((void**)&d_B, size) );
	checkCudaErrors( cudaMalloc((void**)&d_C, size) );

	// Copy vectors from host memory to device memory
	checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

	//VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	dim3 dimGrid(NUM_OF_BLOCKS,1);
	dim3 dimBlock(THREADS_PER_BLOCK,1);
	dim3 dimGrid2(1,1);
	dim3 dimBlock2(1,1);

	cudaThreadSynchronize();
	PowerKernalEmpty<<<dimGrid2,dimBlock2>>>(d_A, d_B, d_C, N, iterations);
	 cudaThreadSynchronize(); 
	//printf("execution time = %f\n", cutGetTimerValue(my_timer));

	dimGrid.y = NUM_OF_BLOCKS;
	for (int i=0; i<3; i++) {
		dimGrid.y /= 3;
		 PowerKernal1<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
		 PowerKernalEmpty<<<dimGrid2,dimBlock2>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
	}
	 
	dimGrid.y = NUM_OF_BLOCKS;
	for (int i=0; i<3; i++) {
		dimGrid.y /= 3;
	 
		 PowerKernal2<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
		 
		 PowerKernalEmpty<<<dimGrid2,dimBlock2>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
	}
	 
	dimGrid.y = NUM_OF_BLOCKS;
	for (int i=0; i<3; i++) {
		dimGrid.y /= 3;
		 PowerKernal3<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
		 
		 PowerKernalEmpty<<<dimGrid2,dimBlock2>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize(); 
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
	}
	 
	 
	dimGrid.y = NUM_OF_BLOCKS;
	for (int i=0; i<3; i++) {
		 PowerKernal4<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
		//printf("execution time = %f\n", cutGetTimerValue(my_timer));
		 PowerKernalEmpty<<<dimGrid2,dimBlock2>>>(d_A, d_B, d_C, N, iterations);
		 cudaThreadSynchronize();
	}

	getLastCudaError("kernel launch failure");
	 cudaThreadSynchronize();
	//printf("execution time = %f\n", cutGetTimerValue(my_timer));

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
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}