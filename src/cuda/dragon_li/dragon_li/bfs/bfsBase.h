#pragma once

#include <hydrazine/interface/debug.h>
#include <dragon_li/util/graphCsrDevice.h>
#include <dragon_li/util/userConfig.h>
#include <dragon_li/util/memsetDevice.h>
#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/timer.h>
#include <dragon_li/util/debug.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsBase {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::MaskType MaskType;
	typedef typename dragon_li::util::GraphCsrDevice<Types> GraphCsrDevice;
	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;


	class UserConfig : public dragon_li::util::UserConfig {
	public:
		double frontierScaleFactor;

		UserConfig(
			bool _verbose,
			bool _veryVerbose,
			double _frontierScaleFactor) :
				dragon_li::util::UserConfig(_verbose, _veryVerbose),
				frontierScaleFactor(_frontierScaleFactor) {}
	};

	//User control
	bool verbose;
	bool veryVerbose;

	//Graph CSR information
	SizeType vertexCount;
	SizeType edgeCount;
	VertexIdType * devColumnIndices;
	SizeType * devRowOffsets;
	SizeType * devSearchDistance;
	std::vector<SizeType> searchDistance;

	//Frontiers for bfs
	SizeType maxFrontierSize;
	SizeType frontierSize;
	VertexIdType * devFrontierContract;
	VertexIdType * devFrontierExpand;
	bool frontierOverflow;

	//Visited mask
	MaskType * devVisitedMasks;

	//Iteration count
	SizeType iteration;

	//Cta Output Assignement
	CtaOutputAssignment ctaOutputAssignment;

    //Timer
    util::CpuTimer cpuTimer;
    util::GpuTimer gpuTimer;

	BfsBase() : 
		verbose(false),
		veryVerbose(false),
		vertexCount(0),
		edgeCount(0),
		devColumnIndices(NULL),
		devRowOffsets(NULL),
		devSearchDistance(NULL),
		maxFrontierSize(0),
		frontierSize(0),
		devFrontierContract(NULL),
		devFrontierExpand(NULL),
		frontierOverflow(false),
		devVisitedMasks(NULL),
		iteration(0) {}

	virtual int search() = 0;

	virtual int setup(
					GraphCsrDevice &graphCsrDevice,
					UserConfig &userConfig) {
		return setup(
				graphCsrDevice.vertexCount,
				graphCsrDevice.edgeCount,
				graphCsrDevice.devColumnIndices,
				graphCsrDevice.devRowOffsets,
				userConfig
			);
	}

	virtual int setup(
			SizeType _vertexCount,
			SizeType _edgeCount,
			VertexIdType * _devColumnIndices,
			SizeType * _devRowOffsets,
			UserConfig & userConfig
		) {

		verbose = userConfig.verbose;
		veryVerbose = userConfig.veryVerbose;
	
		if(!_vertexCount || !_edgeCount
			|| !_devColumnIndices
			|| !_devRowOffsets) {
			errorMsg("Invalid parameters when setting up bfs base!");
			return -1;
		}

		vertexCount = _vertexCount;
		edgeCount = _edgeCount;
		devColumnIndices = _devColumnIndices;
		devRowOffsets = _devRowOffsets;

		cudaError_t retVal;
		if(retVal = cudaMalloc(&devSearchDistance, vertexCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, SizeType, SizeType>
			(devSearchDistance, -1, vertexCount))
			return -1;

		report("frontierSF " << userConfig.frontierScaleFactor);
		maxFrontierSize = (SizeType)((double)edgeCount * userConfig.frontierScaleFactor);
		if(maxFrontierSize <= 0.0) {
			errorMsg("frontier scale factor is too small");
			return -1;
		}

		report("MaxFrontierSize " << maxFrontierSize);
		if(retVal = cudaMalloc(&devFrontierContract, maxFrontierSize * sizeof(VertexIdType))) {
			errorCuda(retVal);
			return -1;
		}

		frontierSize = 1; //always start with one vertex in frontier

		VertexIdType startVertexId = 0; //always expand from id 0
		if(retVal = cudaMemcpy(devFrontierContract, &startVertexId, sizeof(VertexIdType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devFrontierExpand, maxFrontierSize * sizeof(VertexIdType))) {
			errorCuda(retVal);
			return -1;
		}

		//Init visited mask
		SizeType visitedMaskSize = (vertexCount + sizeof(MaskType) - 1) / sizeof(MaskType);
		if(retVal = cudaMalloc(&devVisitedMasks, visitedMaskSize * sizeof(MaskType))) {
			errorCuda(retVal);
			return -1;
		}

		if(dragon_li::util::memsetDevice< Settings::CTAS, Settings::THREADS, MaskType, SizeType >
			(devVisitedMasks, 0, visitedMaskSize))
			return -1;

		if(ctaOutputAssignment.setup(0) != 0)
			return -1;

#ifndef NDEBUG
        //debugger Init
        util::debugInit();
#endif

		return 0;
	}

	virtual int getDevSearchDistance() {

		cudaError_t retVal;

		if(searchDistance.empty()) {
			searchDistance.resize(vertexCount);

			if(retVal = cudaMemcpy((void *)(searchDistance.data()), devSearchDistance, 
						vertexCount * sizeof(SizeType), cudaMemcpyDeviceToHost)) {
				errorCuda(retVal);
				return -1;
			}
		}
		return 0;
	}

	virtual int verifyResult(std::vector<SizeType> &cpuSearchDistance) {

		if(getDevSearchDistance())
				return -1;

		if(std::equal(cpuSearchDistance.begin(), 
				cpuSearchDistance.end(), 
				searchDistance.begin()))
			return 0;
		else
			return 1;
	}

	virtual int displayIteration() {

		cudaError_t retVal;
		if(verbose || veryVerbose) {
	
			std::cout << "Iteration " << iteration <<": frontier size "
				<< frontierSize << "\n";
		}
	
		if(veryVerbose) {
		
			std::vector< VertexIdType > frontier(frontierSize);
		
			if(retVal = cudaMemcpy((void *)(frontier.data()), devFrontierContract, 
							frontierSize * sizeof(VertexIdType), 
							cudaMemcpyDeviceToHost)) {
				errorCuda(retVal);
				return -1;
			}

			std::cout << "Frontier: ";
			for(SizeType i = 0; i < frontierSize; i++) {
				std::cout << frontier[i] << ", ";
			}
			std::cout << "\n";

		}
		return 0;
	}

	virtual int displayResult() {

		std::cout << "GPU search depth = " << iteration << "\n";

		if(veryVerbose) {
		

			if(getDevSearchDistance())
				return -1;

			std::cout << "Search Distance: vertex_id(distance)\n";
			for(SizeType i = 0; i < vertexCount; i++) {
				std::cout << i << "(" << searchDistance[i] << ")\n";
			}
		}

		return 0;

	}


	virtual int finish() { 
        cudaDeviceReset();
        return 0;
    }

};

}
}
