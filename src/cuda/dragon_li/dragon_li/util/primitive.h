#pragma once

namespace dragon_li {
namespace util {

// A very straight-forward PrefixSum with bad performance
template< int CTA_SIZE, typename DataType >
__device__ DataType prefixSumCta(DataType input, DataType &total, DataType carryIn = 0) {

	__shared__ DataType sharedMem[CTA_SIZE];

	if(threadIdx.x < CTA_SIZE)
		sharedMem[threadIdx.x] = input;

	__syncthreads();

	for(int step = 1; step < CTA_SIZE; step <<= 1) {
		if(threadIdx.x >= step)
			input = input + sharedMem[threadIdx.x - step];
		__syncthreads();
		sharedMem[threadIdx.x] = input;
		__syncthreads();
	}

	total = sharedMem[CTA_SIZE - 1] + carryIn;

	if(threadIdx.x == 0)
		return carryIn;
	else
		return sharedMem[threadIdx.x - 1] + carryIn;
}

template< int CTA_SIZE, typename DataType >
__device__ DataType * memcpyCta(DataType *output, const DataType *input, int count) {

    for(int i = threadIdx.x; i < count; i+= CTA_SIZE)
        output[i] = input[i];

    return output + count;
}

template< int CTA_SIZE, typename DataType >
__global__ void scanBlocks(DataType * input,
	int inputCnt, DataType * totals)
{
	int step     = blockDim.x;
	int id       = threadIdx.x + blockIdx.x * step;

	DataType max = 0;

	DataType value  = 0;
	DataType result = 0;


	if(id < inputCnt) value = input[id];

	result = prefixSumCta<CTA_SIZE, DataType>(value, max);
	__syncthreads();

	if(id < inputCnt) input[id] = result;
	
	if(threadIdx.x == 0) totals[blockIdx.x] = max;
}

template< int CTA_SIZE, typename DataType >
__global__ void scanTotals(DataType * input,
	int inputCnt, DataType * totals)
{
	int step     = blockDim.x;
	int ctas     = (inputCnt + step - 1) / step;
	int steps    = (ctas + step - 1) / step;
	DataType carryIn  = 0;
	
	for(unsigned int i = 0; i < steps; ++i)
	{
		int id     = step * i + threadIdx.x;
		DataType value  = 0;
		DataType result = 0;

		if(id < ctas) value = totals[id];

		result = prefixSumCta<CTA_SIZE, DataType>(value, carryIn, carryIn);
		__syncthreads();
	
		if(id < ctas) totals[id] = result;
	}
}

template <typename DataType>
__global__ void scanDistribute(DataType * histogram,
	int inputCnt, DataType * totals)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id < inputCnt) histogram[id] += totals[blockIdx.x];
}


//Inter-CTA prefixsum
template< int CTA_SIZE, typename DataType>
int prefixScan(DataType * input, int inputCnt)
{
	int ctas     = (inputCnt + CTA_SIZE - 1) / CTA_SIZE;
	DataType * totals  = 0;
	cudaError_t status;
	
	if(status = cudaMalloc(&totals, ctas * sizeof(unsigned int))) {
		errorCuda(status);
		return -1;
	}

	scanBlocks<CTA_SIZE, DataType><<<ctas, CTA_SIZE>>>(input, inputCnt, totals);
	
	if(ctas > 1)
	{
		scanTotals<CTA_SIZE, DataType><<<1, CTA_SIZE>>>(input, inputCnt, totals);
		scanDistribute<DataType><<<ctas, CTA_SIZE>>>(input, inputCnt, totals);
	}
	
	if(status = cudaFree(totals)) {
		errorCuda(status);
		return -1;
	}

	return 0;
}


}
}
