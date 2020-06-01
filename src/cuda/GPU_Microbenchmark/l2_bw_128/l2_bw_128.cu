//This code is a modification of L2 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L2 cache for 32f
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

//This code have been tested on Volta V100 architecture

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKS_NUM 80
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM * THREADS_NUM)
#define REPEAT_TIMES 256 
#define WARP_SIZE 32 
#define ARRAY_SIZE (TOTAL_THREADS*4 + REPEAT_TIMES*WARP_SIZE*4)  //Array size must not exceed L2 size 
#define L2_SIZE 1572864 //L2 size in 32-bit. Volta L2 size is 6MB.

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/

__global__ void l2_bw (uint32_t*startClk, uint32_t*stopClk, float*dsink, float*posArray){
	// block and thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t uid = bid * blockDim.x + tid;

	// a register to avoid compiler optimization
	float sink0 = 0;
	float sink1 = 0;
	float sink2 = 0;
	float sink3 = 0;
	
	// warm up l2 cache
	for(uint32_t i = uid; i<ARRAY_SIZE; i+=TOTAL_THREADS){
		float* ptr = posArray+i;
		// every warp loads all data in l2 cache
		// use cg modifier to cache the load in L2 and bypass L1
		asm volatile("{\t\n"
			".reg .f32 data;\n\t"
			"ld.global.cg.f32 data, [%1];\n\t"
			"add.f32 %0, data, %0;\n\t"
			"}" : "+f"(sink0) : "l"(ptr) : "memory"
		);
	}
	
	asm volatile("bar.sync 0;");	

	// start timing
	uint32_t start = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// load data from l2 cache and accumulate,
	for(uint32_t i = 0; i<REPEAT_TIMES; i++){
			float* ptr = posArray+(((i*WARP_SIZE*4)+uid*4)%ARRAY_SIZE);
		asm volatile ("{\t\n"
			".reg .f32 data<4>;\n\t"
			"ld.global.cg.v4.f32 {data0,data1,data2,data3}, [%4];\n\t"
			"add.f32 %0, data0, %0;\n\t"
                        "add.f32 %1, data1, %1;\n\t"
                        "add.f32 %2, data2, %2;\n\t"
                        "add.f32 %3, data3, %3;\n\t"
			"}" : "+f"(sink0), "+f"(sink1), "+f"(sink2), "+f"(sink3)  : "l"(ptr) : "memory"
		);
	}
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// store the result
	startClk[uid] = start;
	stopClk[uid] = stop;
	dsink[uid] = sink0+sink1+sink2+sink3;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
        uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));

	float *posArray = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *dsink = (float*) malloc(TOTAL_THREADS*sizeof(float));

	float *posArray_g;
	float *dsink_g;
	uint32_t *startClk_g;
        uint32_t *stopClk_g;

        for (int i=0; i<ARRAY_SIZE; i++)
                posArray[i] = (float)i;

        gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(float)) );
        gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(float)) );
        gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
        gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );

        gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );


        l2_bw<<<BLOCKS_NUM,THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g, posArray_g);
	gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );

	float bw;
	unsigned long long data = (unsigned long long)TOTAL_THREADS*REPEAT_TIMES*4*4;
	bw = (float)(data)/((float)(stopClk[0]-startClk[0]));
	printf("L2 bandwidth = %f (byte/cycle)\n", bw);
        printf("Total Clk number = %u \n", (stopClk[0]-startClk[0]));

        return 0;
}
