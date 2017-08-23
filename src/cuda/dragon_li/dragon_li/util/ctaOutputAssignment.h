#pragma once

namespace dragon_li {
namespace util {

template< typename SizeType >
class CtaOutputAssignment {

public:

	SizeType *devOutputOffset;

	CtaOutputAssignment() : devOutputOffset(NULL) {}

	int setup(SizeType startOffset) {

		cudaError_t retVal;

		if(retVal = cudaMalloc(&devOutputOffset, sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMemcpy(devOutputOffset, &startOffset, sizeof(SizeType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}


	int reset() {

		cudaError_t retVal;

		SizeType outputOffset = 0;
		if(retVal = cudaMemcpy(devOutputOffset, &outputOffset, sizeof(SizeType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

	int getGlobalSize(SizeType &globalSize) {
		cudaError_t retVal;

		if(retVal = cudaMemcpy(&globalSize, devOutputOffset, sizeof(SizeType),
						cudaMemcpyDeviceToHost)) {
			errorCuda(retVal);
			return -1;
		}

		return 0;

	}

	__device__ SizeType getCtaOutputAssignment(SizeType outputCount) {

		return atomicAdd(devOutputOffset, outputCount);
	}

	__device__ SizeType getGlobalSize() {
		return *devOutputOffset;
	}

};

}
}
