#pragma once

#include <dragon_li/amr/amrBase.h>
#include <dragon_li/amr/amrRegDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrReg : public AmrBase< Settings > {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	DataType * devActiveGridData;
	SizeType * devActiveGridPointer;

	SizeType iteration;
	SizeType processGridOffset;

	AmrReg() : AmrBase<Settings>(),
		iteration(0),
		processGridOffset(0){}	

	virtual int refine() {

		while(iteration < this->maxRefineLevel && !this->gridSizeOverflow) {

			SizeType processGridSize = this->activeGridSize - processGridOffset;
            if(this->verbose)
			    std::cout << "Iteration " << iteration << ": " <<
                    "ProcessGridSize " << processGridSize << ", ";

			amrRegRefineKernel< Settings >
				<<< CTAS, THREADS >>> (
					this->devGridData,
					this->devGridPointer,
					this->maxGridDataSize,
					processGridOffset,
					processGridSize,
					this->maxRefineLevel,
					this->gridRefineThreshold,
					this->ctaOutputAssignment
				);
	
			cudaError_t retVal;
			if(retVal = cudaDeviceSynchronize()) {
				errorCuda(retVal);
				return -1;
			}

			processGridOffset = this->activeGridSize;

			if(this->ctaOutputAssignment.getGlobalSize(this->activeGridSize))
				return -1;

            if(this->verbose)
    			std::cout << "activeGridSize = " << this->activeGridSize 
                    << "\n";

			iteration++;
			if(this->activeGridSize == processGridOffset) //no new cells generated
				break;

			if(iteration >= this->maxRefineLevel) {
				report("Max Refine Level reached! Consider increasing maxRefineLevel!");
				break;
			}
			if(this->activeGridSize > this->maxGridDataSize) {
				this->gridSizeOverflow = true;
				errorMsg("Grid Data Size overflow! Consider increasing maxGridDataSize!");
				break;
			}



		}

		return 0;

	}

	int getDevGridData() {
		
		cudaError_t retVal;

		if(this->gridData.empty()) {
			this->gridData.resize(this->activeGridSize);

			if(retVal = cudaMemcpy((void *)(this->gridData.data()), this->devGridData,
							this->activeGridSize * sizeof(DataType), cudaMemcpyDeviceToHost)) {

				errorCuda(retVal);
				return -1;
			}

		}

		if(this->gridPointer.empty()) {
			this->gridPointer.resize(this->activeGridSize);

			if(retVal = cudaMemcpy((void *)(this->gridPointer.data()), this->devGridPointer,
							this->activeGridSize * sizeof(SizeType), cudaMemcpyDeviceToHost)) {
				errorCuda(retVal);
				return -1;
			}
		}

		return 0;
	}

	virtual int displayResult() {

		std::cout << "GPU AMR refine depth = " << iteration 
			<< ", Grid Size = " << this->activeGridSize << "\n";

		if(this->veryVerbose) {

			if(getDevGridData())
				return -1;

			std::cout << "Grid: index, data, pointer\n";
			for(SizeType i = 0; i < this->activeGridSize; i++) {

				std::cout << i << ", " << this->gridData[i] << ", " << this->gridPointer[i] << "\n";
			}
		}

		return 0;

	}
};

}
}
