#pragma once

namespace dragon_li {
namespace util {

template< typename Settings >
class ThreadWorkAssignment {

	typedef typename Settings::SizeType SizeType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:
	SizeType totalWorkSize;
	SizeType workOffset;
	SizeType workSize;

	__device__ ThreadWorkAssignment(SizeType _totalWorkSize) : 
		totalWorkSize(_totalWorkSize),
		workOffset(-1),
		workSize(0) {}

	__device__ void getThreadWorkAssignment() {
		SizeType totalThreads = THREADS * CTAS;
		if(workOffset == -1) { //first time
			workOffset = min(blockIdx.x * blockDim.x + threadIdx.x, totalWorkSize);
		}
		else {
			workOffset = min(workOffset + totalThreads, totalWorkSize);
		}
		workSize = min(totalWorkSize - workOffset, 1);
	}


};

}
}
