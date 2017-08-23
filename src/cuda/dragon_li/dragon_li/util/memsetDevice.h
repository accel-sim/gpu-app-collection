#pragma once

#include <dragon_li/util/debug.h>

namespace dragon_li {
namespace util {

template< typename DataType, typename SizeType >
__global__ void memsetKernel(DataType * devDst, DataType val, SizeType count) {
	
	SizeType startId = threadIdx.x + blockIdx.x * blockDim.x;
	SizeType step = blockDim.x * gridDim.x;

	for(SizeType id = startId; id < count; id += step)
		devDst[id] = val;

}

template< int GRID_SIZE, int CTA_SIZE, typename DataType, typename SizeType >
int memsetDevice(DataType * devDst, DataType val, SizeType count) {

	memsetKernel< DataType, SizeType >
		<<<GRID_SIZE, CTA_SIZE>>>(devDst, val, count);

	cudaError_t retVal;
	if(retVal = cudaDeviceSynchronize()) {
		errorCuda(retVal);
		return -1;
	}

	return 0;
}

}
}
