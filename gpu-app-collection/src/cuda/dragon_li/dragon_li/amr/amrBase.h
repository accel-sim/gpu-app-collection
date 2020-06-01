#pragma once

#include <list>

#include <dragon_li/util/userConfig.h>
#include <dragon_li/util/debug.h>
#include <dragon_li/util/random.h>
#include <dragon_li/util/memsetDevice.h>
#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/debug.h>
#include <dragon_li/amr/amrCpu.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template < typename Settings >
class AmrCpu;

template < typename Settings >
class AmrBase {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;
	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;

	class UserConfig : public dragon_li::util::UserConfig {
	public:
	
		SizeType maxGridDataSize;
		SizeType maxRefineLevel;
		DataType maxGridValue;
		DataType gridRefineThreshold;
	
		UserConfig(
			bool _verbose,
			bool _veryVerbose,
			SizeType _maxGridDataSize,
			DataType _maxRefineLevel,
			DataType _maxGridValue,
			DataType _gridRefineThreshold) :
				dragon_li::util::UserConfig(_verbose, _veryVerbose),
				maxGridDataSize(_maxGridDataSize),
				maxRefineLevel(_maxRefineLevel),
				maxGridValue(_maxGridValue),
				gridRefineThreshold(_gridRefineThreshold) {}
	};

	//User config
	bool verbose;
	bool veryVerbose;

	//AMR Data
	SizeType maxGridDataSize;
	SizeType maxRefineLevel;
	DataType maxGridValue;
	DataType startGridValue;
	DataType gridRefineThreshold;
	SizeType activeGridSize;
	bool gridSizeOverflow;

	DataType * devGridData;
	SizeType * devGridPointer;
	std::vector< DataType > gridData;
	std::vector< SizeType > gridPointer;

	//Cta Output Assignement
	CtaOutputAssignment ctaOutputAssignment;

	AmrBase() :
			verbose(false),
			veryVerbose(false),
			maxGridDataSize(0),
			maxRefineLevel(0),
			maxGridValue(0),
			gridRefineThreshold(0),
			activeGridSize(0),
			gridSizeOverflow(false),
			devGridData(NULL),
			devGridPointer(NULL) {}

	virtual int refine() = 0;

	virtual int setup(UserConfig & userConfig) {

		verbose = userConfig.verbose;
		veryVerbose = userConfig.veryVerbose;
		maxGridDataSize = userConfig.maxGridDataSize;
		maxRefineLevel = userConfig.maxRefineLevel;
		maxGridValue = userConfig.maxGridValue;
		gridRefineThreshold = userConfig.gridRefineThreshold;

		if(maxGridDataSize <= 0) {
			errorMsg("Invalid maxGridDataSize " << maxGridDataSize);
			return -1;
		}
		if(maxRefineLevel <= 0) {
			errorMsg("Invalid maxRefineLevel " << maxRefineLevel);
			return -1;
		}

		cudaError_t retVal;
		if(retVal = cudaMalloc(&devGridData, maxGridDataSize * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}

		dragon_li::util::Random<DataType, SizeType>::random(&startGridValue, 1, 0, maxGridValue);
        if(verbose)
    		std::cout << "Start Grid Value: " << startGridValue << "\n"; 
		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, DataType, SizeType>
			(devGridData, startGridValue, 1))
			return -1;

		activeGridSize = 1;

		if(retVal = cudaMalloc(&devGridPointer, maxGridDataSize * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, SizeType, SizeType>
			(devGridPointer, -1, maxGridDataSize))
			return -1;

		//Output starts after activeGridSize
		if(ctaOutputAssignment.setup(activeGridSize) != 0)
			return -1;

#ifndef NDEBUG
        //debugger Init
        util::debugInit();
#endif

		return 0;

	}

	DataType getStartGridValue() {
		return startGridValue;
	}

    virtual int verifyResult(std::list< struct AmrCpu<Settings>::AmrCpuData > &cpuAmrData) {
        
        //Quick verify size only
        if(activeGridSize == cpuAmrData.size())
            return 0;
        else
            return 1;
    }

	virtual int displayResult() {
		return 0;
	}

	virtual int finish() {
        cudaDeviceReset();
		return 0;
	}

};

}
}
