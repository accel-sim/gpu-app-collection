//This code is a modification of L2 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of GPU memory
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

//This code have been tested on Volta V100 architecture
//You can check the mem BW from the NVPROF (dram_read_throughput+dram_write_throughput)

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKS_NUM 160
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM*THREADS_NUM)
#define ARRAY_SIZE 8388608   //Array size has to exceed L2 size to avoid L2 cache residence
#define WARP_SIZE 32 
#define L2_SIZE 1572864 //number of floats L2 can store

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
Four Vector Addition using flost4 types
Send as many as float4 read requests on the flight to increase Row buffer locality of DRAM and hit the max BW
*/

__global__ void mem_bw (float* A,  float* B, float* C, float* D, float* E, float* F){
	// block and thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = idx; i < ARRAY_SIZE/4; i += blockDim.x * gridDim.x) {
		float4 a1 = reinterpret_cast<float4*>(A)[i];
		float4 b1 = reinterpret_cast<float4*>(B)[i];
		float4 d1 = reinterpret_cast<float4*>(D)[i];
		float4 e1 = reinterpret_cast<float4*>(E)[i];
		float4 f1 = reinterpret_cast<float4*>(F)[i];
		float4 c1;
		
		c1.x = a1.x + b1.x + d1.x + e1.x + f1.x;
		c1.y = a1.y + b1.y + d1.y + e1.y + f1.y;
		c1.z = a1.z + b1.z + d1.z + e1.z + f1.z;
		c1.w = a1.w + b1.w + d1.w + e1.w + f1.w;
		
		reinterpret_cast<float4*>(C)[i] = c1;
	}
}

int main(){
	float *A = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *B = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *C = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *D = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *E = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *F = (float*) malloc(ARRAY_SIZE*sizeof(float));


	float *A_g = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *B_g = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *C_g = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *D_g = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *E_g = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *F_g = (float*) malloc(ARRAY_SIZE*sizeof(float));


        for (uint32_t i=0; i<ARRAY_SIZE; i++){
                A[i] = (float)i;
		B[i] = (float)i;
		D[i] = (float)i;
		E[i] = (float)i;
		F[i] = (float)i;
		
	}

        gpuErrchk( cudaMalloc(&A_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&B_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&C_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&D_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&E_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&F_g, ARRAY_SIZE*sizeof(float)) );


        gpuErrchk( cudaMemcpy(A_g, A, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(B_g, B, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(D_g, D, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(E_g, E, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(F_g, F, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );

        mem_bw<<<BLOCKS_NUM,THREADS_NUM>>>(A_g, B_g, C_g, D_g, E_g, F_g);
	gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(C, C_g, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost) );


}
