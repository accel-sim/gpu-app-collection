#pragma once

#include <dragon_li/sssp/ssspBase.h>
#include <dragon_li/sssp/ssspRegDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace sssp {

template< typename Settings >
class SsspReg : public SsspBase< Settings > {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:

	SsspReg() : SsspBase< Settings >() {}

	void swapInOut() {

			//Swap devFrontierIn and devFrontierOut
			VertexIdType * tmp = this->devFrontierIn;
			this->devFrontierIn = this->devFrontierOut;
			this->devFrontierOut = tmp;
			
	}

	int search() {
		while(this->frontierSize > 0) {
			report("Start SSSP Search in regular mode... (" << CTAS << ", " << THREADS << ")");
			report("Iteration " << this->iteration );

			if(this->ctaOutputAssignment.reset())
				return -1;
	
			report("Expand...");

			if(expand())
				return -1;
		
			if(this->ctaOutputAssignment.getGlobalSize(this->frontierSize))
				return -1;

			if(this->frontierSize > this->maxFrontierSize) {
				this->frontierOverflow = true;
				errorMsg("Frontier overflow! Please increase frontier scale factor!");
				return -1;
			}

			if(this->displayIteration())
				return -1;

			this->iteration++;

			swapInOut();

		}

		return 0;

	}

	virtual int expand() {
				
		ssspRegExpandKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devColumnIndices,
				this->devColumnWeights,
				this->devRowOffsets,
				this->devSearchDistance,
				this->devFrontierIn,
				this->devFrontierOut,
				this->maxFrontierSize,
				this->frontierSize,
				this->ctaOutputAssignment,
				this->iteration);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

};

}
}
