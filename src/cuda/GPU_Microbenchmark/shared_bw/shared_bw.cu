//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L1 cache for 64 bit read

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define SHARED_MEM_SIZE_BYTE (48*1024) //size in bytes, max 96KB for v100
#define SHARED_MEM_SIZE (SHARED_MEM_SIZE_BYTE/4)
#define ITERS (SHARED_MEM_SIZE/2)

#define BLOCKS_NUM 1
#define THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void shared_bw(uint32_t *startClk, uint32_t *stopClk, float *dsink){
    
    // thread index
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid*blockDim.x+tid;
    uint32_t n_threads = blockDim.x * gridDim.x;

    __shared__ float s[SHARED_MEM_SIZE]; //static shared memory

    // one thread to initialize the pointer-chasing array
	for (uint32_t i=uid; i<(SHARED_MEM_SIZE); i+=n_threads)
        s[i] = (float)i;

	// a register to avoid compiler optimization
	float sink0 = 0;
	float sink1 = 0;
	float sink2 = 0;
	float sink3 = 0;
	
	// synchronize all threads
	asm volatile ("bar.sync 0;");
	
	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// load data from l1 cache and accumulate
	for(uint32_t i=0; i<ITERS; ++i){
        for(uint32_t j=uid*4; j<(SHARED_MEM_SIZE-ITERS); j+=(4*n_threads)){
            sink0 += s[j+0+i];
            sink1 += s[j+1+i];
            sink2 += s[j+2+i];
            sink3 += s[j+3+i];
        }
	}

	// synchronize all threads
	asm volatile("bar.sync 0;");
	
	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[uid] = start;
	stopClk[uid] = stop;
	dsink[uid] = sink0+sink1+sink2+sink3;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	float *dsink = (float*) malloc(TOTAL_THREADS*sizeof(float));
	
	uint32_t *startClk_g;
    uint32_t *stopClk_g;
    float *dsink_g;
		
	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(float)) );
	
	shared_bw<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g);
    gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );

    float bw;
	bw = (float)(ITERS*(SHARED_MEM_SIZE-ITERS)*4*4)/((float)(stopClk[0]-startClk[0]));
	printf("Shared Memory Bandwidth = %f (byte/clk/SM)\n", bw);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
}