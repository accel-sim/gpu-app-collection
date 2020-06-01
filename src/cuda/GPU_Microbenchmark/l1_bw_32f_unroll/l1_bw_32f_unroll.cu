//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L1 cache for 64 bit read

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_SM 1024
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 1024
#define ARRAY_SIZE 16384 //ARRAY_SIZE has to be less than L1_SIZE
#define L1_SIZE 32768  //L1 size in 64-bit. Volta L1 size is 128KB, i.e. 16K of 64-bit

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void l1_bw(uint32_t *startClk, uint32_t *stopClk, float *dsink, float *posArray){
	
	// thread index
	uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
	
	// a register to avoid compiler optimization
	float sink0 = 0;
	float sink1 = 0;
	float sink2 = 0;
	float sink3 = 0;
	
	// populate l1 cache to warm up
	for (uint32_t i = tid; i<ARRAY_SIZE; i+=THREADS_PER_BLOCK) {
		float* ptr = posArray + i;
		// use ca modifier to cache the load in L1
		asm volatile ("{\t\n"
			".reg .f32 data;\n\t"
			"ld.global.ca.f32 data, [%1];\n\t"
			"add.f32 %0, data, %0;\n\t"
			"}" : "+f"(sink0) : "l"(ptr) : "memory"
		);
	}
	
	// synchronize all threads
	asm volatile ("bar.sync 0;");
	
	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// load data from l1 cache and accumulate
	for(uint32_t j=0; j<REPEAT_TIMES; j++){
        	        float* ptr = posArray + ((tid + (j*WARP_SIZE))%ARRAY_SIZE);
                	asm volatile ("{\t\n"
                        	".reg .f32 data;\n\t"
				"ld.global.ca.f32 data, [%1+0];\n\t"
				"add.f32 %0, data, %0;\n\t"
				"}" : "+f"(sink0) : "l"(ptr) : "memory"
                	);
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
	float *posArray = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *dsink = (float*) malloc(TOTAL_THREADS*sizeof(float));
	
	uint32_t *startClk_g;
        uint32_t *stopClk_g;
        float *posArray_g;
        float *dsink_g;
	
	for (uint32_t i=0; i<ARRAY_SIZE; i++)
		posArray[i] = (float)i;
		
	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(float)) );
	
	gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );

	l1_bw<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g, posArray_g);
        gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );

        float bw;
	bw = (float)(REPEAT_TIMES*THREADS_PER_SM*4)/((float)(stopClk[0]-startClk[0]));
	printf("L1 bandwidth = %f (byte/clk/SM)\n", bw);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
} 
