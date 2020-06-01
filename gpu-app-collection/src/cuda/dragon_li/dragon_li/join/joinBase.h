#pragma once


#include <dragon_li/util/userConfig.h>
#include <dragon_li/util/debug.h>
#include <dragon_li/join/joinData.h>

#undef REPORT_BASE
#define REPORT_BASE

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinBase {

public:

	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	class UserConfig : public dragon_li::util::UserConfig {
	public:

		UserConfig(
			bool _verbose,
			bool _veryVerbose
			) :
				dragon_li::util::UserConfig(_verbose, _veryVerbose)
				{}
	};



	//User control
	bool verbose;
	bool veryVerbose;

	//Join information
	SizeType inputCountLeft;
	SizeType inputCountRight;
	SizeType outputCount;
	std::vector<SizeType> outputIndicesLeft;
	std::vector<SizeType> outputIndicesRight;

	//Join Device information
	DataType * devJoinInputLeft;
	DataType * devJoinInputRight;
	SizeType * devJoinLeftOutIndices;
	SizeType * devJoinRightOutIndices;
	SizeType * devJoinOutputCount;

	JoinBase() : 
		verbose(false),
		veryVerbose(false),
		inputCountLeft(0),
		inputCountRight(0),
		outputCount(0),
		devJoinInputLeft(NULL),
		devJoinInputRight(NULL),
		devJoinLeftOutIndices(NULL),
		devJoinRightOutIndices(NULL),
		devJoinOutputCount(NULL) {}

	virtual int join() = 0;

	virtual int setup(JoinData<Types> & joinData,
					UserConfig & userConfig) {

		verbose = userConfig.verbose;
		veryVerbose = userConfig.veryVerbose;

		inputCountLeft = joinData.inputCountLeft;
		inputCountRight = joinData.inputCountRight;

		cudaError_t retVal;

		if(retVal = cudaMalloc(&devJoinInputLeft, inputCountLeft * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(devJoinInputLeft, 
								joinData.inputLeft.data(), 
								inputCountLeft * sizeof(DataType),
								cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMalloc(&devJoinInputRight, inputCountRight * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(devJoinInputRight, 
								joinData.inputRight.data(), 
								inputCountRight * sizeof(DataType),
								cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		if(veryVerbose) {
			std::cout << "Input Left: \n";
			for(int i = 0; i < inputCountLeft; i++)
				std::cout << joinData.inputLeft[i] << ", ";

			std::cout << "\n\nInput Right: \n";
			for(int i = 0; i < inputCountRight; i++)
				std::cout << joinData.inputRight[i] << ", ";

			std::cout << "\n\n";
		}
		if(retVal = cudaMalloc(&devJoinOutputCount, sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemset(devJoinOutputCount, 0, sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

#ifndef NDEBUG
		util::debugInit();
#endif

		return 0;
	}

	int getDevJoinResult() {

		cudaError_t retVal;

		if(outputIndicesLeft.empty() && outputIndicesRight.empty()) {

			outputIndicesLeft.resize(outputCount);
			outputIndicesRight.resize(outputCount);
		
			if(retVal = cudaMemcpy(outputIndicesLeft.data(), 
				devJoinLeftOutIndices, 
				outputCount * sizeof(SizeType), 
				cudaMemcpyDeviceToHost)) {

				errorCuda(retVal);
				return -1;
			}

			if(retVal = cudaMemcpy(outputIndicesRight.data(), 
				devJoinRightOutIndices, 
				outputCount * sizeof(SizeType), 
				cudaMemcpyDeviceToHost)) {

				errorCuda(retVal);
				return -1;
			}
		}

		return 0;
		
	}

	virtual int verifyResult(std::vector<SizeType> &cpuJoinLeftIndices,
		std::vector<SizeType> &cpuJoinRightIndices, 
		JoinData<Types> &joinData) {

		if(getDevJoinResult())
			return -1;

		if(cpuJoinLeftIndices.size() != outputCount ||
			cpuJoinRightIndices.size() != outputCount) {

			std::cout << "Error: cpuSize " << cpuJoinLeftIndices.size()
				<< ", gpuSize " << outputCount << "\n";
			return 1;
		}

		for(SizeType i = 0; i < outputCount; i++) {
			SizeType cpuLeftId = cpuJoinLeftIndices[i];
			SizeType cpuRightId = cpuJoinRightIndices[i];
			SizeType gpuLeftId = outputIndicesLeft[i];
			SizeType gpuRightId = outputIndicesRight[i];

			if((joinData.inputLeft[cpuLeftId] != 
				joinData.inputLeft[gpuLeftId] ) || 
				(joinData.inputRight[cpuRightId] !=
				joinData.inputRight[gpuRightId]))
				return 1;
		}

		return 0;
	}

	virtual int displayResult() {
		return 0;
	}

	virtual int finish() {

		cudaError_t retVal;
		if(retVal = cudaFree(devJoinInputLeft)) {
			errorCuda(retVal);
			return -1;
		}
		
		if(retVal = cudaFree(devJoinInputRight)) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaFree(devJoinLeftOutIndices)) {
			errorCuda(retVal);
			return -1;
		}
		
		if(retVal = cudaFree(devJoinRightOutIndices)) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaFree(devJoinOutputCount)) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}
};

}
}
