//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the latency of GPU memory

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_NUM 4  //HERE, we launch four threads, to ensure that one request is equal to DRAM trascation, 4 thread * 8 bytes = 32 bytes (= min DRAM trascation)
#define WARP_SIZE 32
#define ITERS 32768    //1MB of pointer chasing, ITERS*THREADS_NUM*8 bytes
#define ARRAY_SIZE 917504   //pointer-chasing array size in 64-bit. total array size is 7 MB which larger than L2 cache size (6 MB in Volta) to avoid l2 cache resident from the copy engine
#define BLOCKS 160
#define THREADS_PER_BLOCK 1024
#define TOTAL_THREADS  BLOCKS*THREADS_PER_BLOCK

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}


__global__ void mem_lat(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){
	// thread index
	uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;

	// initialize pointer-chasing array
	for (uint32_t i=uid; i<(ARRAY_SIZE-THREADS_NUM); i+=TOTAL_THREADS)
		posArray[i] = (uint64_t)(posArray + i + THREADS_NUM);


	if(uid < THREADS_NUM){   //only THREADS_NUM has to be active here

		// initialize the tail to reference to the head of the array
		posArray[ARRAY_SIZE-(THREADS_NUM-tid)] = (uint64_t)posArray + tid;

		uint64_t *ptr = posArray + tid;
		uint64_t ptr1, ptr0;
	
		// initialize the pointers with the start address
		//Here, we use cache volatile modifier to ignore the L2 cache
		asm volatile ("{\t\n"
			"ld.global.cv.u64 %0, [%1];\n\t"
			"}" : "=l"(ptr1) : "l"(ptr) : "memory"
		);
	
		// synchronize all threads
		asm volatile ("bar.sync 0;");

		uint32_t start = 0;
		uint32_t stop = 0;

		// start timing
		asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

		// pointer-chasing ITERS times
		//Here, we use cache volatile modifier to ignore the L2 cache
		for(uint32_t i=tid; i<ITERS-THREADS_NUM; i+=THREADS_NUM) {	
			asm volatile ("{\t\n"
				"ld.global.cv.u64 %0, [%1];\n\t"
				"}" : "=l"(ptr0) : "l"((uint64_t*)ptr1) : "memory"
			);
			ptr1 = ptr0;   //swap the register for the next load
		}

		// stop timing
		asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

		// write time and data back to memory
		startClk[tid] = start;
		stopClk[tid] = stop;
		dsink[tid] = ptr1;
	}
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(THREADS_NUM*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(THREADS_NUM*sizeof(uint32_t));
	uint64_t *dsink = (uint64_t*) malloc(THREADS_NUM*sizeof(uint64_t));
	
	uint32_t *startClk_g;
        uint32_t *stopClk_g;
        uint64_t *posArray_g;
        uint64_t *dsink_g;
	
	gpuErrchk( cudaMalloc(&startClk_g, THREADS_NUM*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, THREADS_NUM*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&dsink_g, THREADS_NUM*sizeof(uint64_t)) );
	
	mem_lat<<<BLOCKS,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, THREADS_NUM*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, THREADS_NUM*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, THREADS_NUM*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	printf("Mem latency = %12.4f cycles \n", (float)(stopClk[0]-startClk[0])/(float)(ITERS/THREADS_NUM));
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
} 
