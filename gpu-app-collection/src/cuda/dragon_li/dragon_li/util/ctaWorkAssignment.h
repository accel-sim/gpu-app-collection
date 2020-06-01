#pragma once

namespace dragon_li {
namespace util {

template< typename Settings >
class CtaWorkAssignment {

	typedef typename Settings::SizeType SizeType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:
	SizeType totalWorkSize;
	SizeType workOffset;
	SizeType workSize;

	__device__ CtaWorkAssignment(SizeType _totalWorkSize) : 
		totalWorkSize(_totalWorkSize),
		workOffset(-1),
		workSize(0) {}

	__device__ void getCtaWorkAssignment() {
		SizeType totalThreads = THREADS * CTAS;
		if(workOffset == -1) { //first time
			workOffset = min(blockIdx.x * THREADS, totalWorkSize);
		}
		else {
			workOffset = min(workOffset + totalThreads, totalWorkSize);
		}
		workSize = min(THREADS, totalWorkSize - workOffset);
	}
};

}
}
