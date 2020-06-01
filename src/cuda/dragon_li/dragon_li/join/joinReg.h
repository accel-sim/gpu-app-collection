#pragma once

#include <algorithm>
#include <vector>

#include <dragon_li/util/memsetDevice.h>
#include <dragon_li/join/joinBase.h>
#include <dragon_li/join/joinRegDevice.h>
#include <dragon_li/join/joinData.h>

#undef REPORT_BASE
#define REPORT_BASE 1 

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinReg : public JoinBase< Settings > {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	//Processing temporary storage
	SizeType *devLowerBounds;
	SizeType *devUpperBounds;
	SizeType *devOutBounds;
	SizeType *devHistogram;
	SizeType *devJoinLeftOutIndicesScattered;
	SizeType *devJoinRightOutIndicesScattered;
	
	SizeType estJoinOutCount;


	JoinReg() : JoinBase< Settings >(), 
		devLowerBounds(NULL),
		devUpperBounds(NULL),
		devOutBounds(NULL),
		devHistogram(NULL),
		devJoinLeftOutIndicesScattered(NULL),
		devJoinRightOutIndicesScattered(NULL),
		estJoinOutCount(0) {}

	virtual int setup(JoinData<Types> & joinData,
				typename JoinBase<Settings>::UserConfig & userConfig) {

		//call setup from base class
		if(JoinBase<Settings>::setup(joinData, userConfig))
			return -1;

		cudaError_t retVal;
	
		if(retVal = cudaMalloc(&devLowerBounds, CTAS * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMemset(devLowerBounds, 
			0, CTAS * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devUpperBounds, CTAS * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMemset(devUpperBounds, 
			0, CTAS * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devOutBounds, (CTAS + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMemset(devOutBounds, 
			0, (CTAS + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devHistogram, (CTAS + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMemset(devHistogram, 
			0, (CTAS + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		estJoinOutCount = std::max(this->inputCountLeft, this->inputCountRight) * Settings::JOIN_SF; 

		if(retVal = cudaMalloc(&devJoinLeftOutIndicesScattered, estJoinOutCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemset(devJoinLeftOutIndicesScattered, 
			0, estJoinOutCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMalloc(&devJoinRightOutIndicesScattered, estJoinOutCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemset(devJoinRightOutIndicesScattered, 
			0, estJoinOutCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}


		return 0;
	
	}

	int findBounds() {
		
		joinRegFindBoundsKernel< Settings >
			<<< (CTAS + THREADS - 1)/THREADS, THREADS >>> (
			this->devJoinInputLeft,
			this->inputCountLeft,
			this->devJoinInputRight,
			this->inputCountRight,
			devLowerBounds,
			devUpperBounds,
			devOutBounds
		);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		if(this->veryVerbose) {
			std::vector<SizeType> upper(CTAS), lower(CTAS);
			cudaMemcpy(lower.data(), devLowerBounds, CTAS * sizeof(SizeType), cudaMemcpyDeviceToHost);
			cudaMemcpy(upper.data(), devUpperBounds, CTAS * sizeof(SizeType), cudaMemcpyDeviceToHost);
		

			std::cout << "Right Bounds:\n";
			for(int i = 0; i < CTAS; i++)
				std::cout << "[" << lower[i] << ", " << upper[i] << "], ";
			std::cout << "\n\n";

		}
		
		if(util::prefixScan<THREADS, SizeType>(devOutBounds, CTAS + 1)) {
			errorMsg("Prefix Sum for outBounds fails");
			return -1;
		}

		if(this->veryVerbose) {
			std::vector<SizeType> outBounds(CTAS+1);
			cudaMemcpy(outBounds.data(), devOutBounds, (CTAS+1) * sizeof(SizeType), cudaMemcpyDeviceToHost);
			std::cout << "Out Bounds:\n";
			for(int i = 0; i < CTAS+1; i++)
				std::cout << outBounds[i] << ", ";
			std::cout << "\n\n";
		}

		return 0;
		
	}

	virtual int mainJoin() {

		joinRegMainJoinKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devJoinInputLeft,
				this->inputCountLeft,
				this->devJoinInputRight,
				this->inputCountRight,
				devJoinLeftOutIndicesScattered,
				devJoinRightOutIndicesScattered,
				devHistogram,
				devLowerBounds,
				devUpperBounds,
				devOutBounds);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

//		std::vector<SizeType> upper(estJoinOutCount);
//		cudaMemcpy(upper.data(), devJoinRightOutIndicesScattered, (estJoinOUtCount) * sizeof(SizeType), cudaMemcpyDeviceToHost);
//		for(int i = 0; i < 1200; i++)
//			std::cout << "u" << i << ": " << upper[i] << "\n";
		

		return 0;
	}

	int gather() {

		if(util::prefixScan<THREADS, SizeType>(devHistogram, CTAS + 1)) {
			errorMsg("Prefix Sum for histogram fails");
			return -1;
		}
		if(this->veryVerbose) {
			std::vector<SizeType> histogram(CTAS+1);
			cudaMemcpy(histogram.data(), devHistogram, (CTAS+1) * sizeof(SizeType), cudaMemcpyDeviceToHost);
			std::cout << "Histogram:\n";
			for(int i = 0; i < CTAS+1; i++)
				std::cout << histogram[i] << ", ";
			std::cout << "\n\n";
		}

		cudaError_t retVal;
		if(retVal = cudaMemcpy(&this->outputCount, devHistogram + CTAS, sizeof(SizeType), cudaMemcpyDeviceToHost)) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&this->devJoinLeftOutIndices, this->outputCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemset(this->devJoinLeftOutIndices, 
			0, this->outputCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&this->devJoinRightOutIndices, this->outputCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemset(this->devJoinRightOutIndices, 
			0, this->outputCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		joinRegGatherKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devJoinLeftOutIndices,
				this->devJoinRightOutIndices,
				devJoinLeftOutIndicesScattered,
				devJoinRightOutIndicesScattered,
				estJoinOutCount,
				devOutBounds,
				devHistogram,
				this->devJoinOutputCount
			);

		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		std::cout << "Join output size " << this->outputCount << "\n";

		if(this->veryVerbose) {

			if(this->getDevJoinResult())
				return -1;

			std::cout << "Output Indices:\n";
			for(int i = 0; i < this->outputCount; i++)
				std::cout << "[" << this->outputIndicesLeft[i] 
					<< ", " << this->outputIndicesRight[i] << "], ";
			std::cout << "\n\n";

		}

		return 0;
	}

	int join() {

		if(this->verbose)
			std::cout << "Finding bounds...\n";

		if(findBounds())
			return -1;

		if(this->verbose)
			std::cout << "Main Join...\n";

		if(mainJoin())
			return -1;

		if(this->verbose)
			std::cout << "Gathering...\n";
		
		if(gather())
			return -1;

		return 0;
			
	}
};

}
}
