#pragma once

#include <hydrazine/interface/debug.h>
#include <dragon_li/util/graphCsrDevice.h>
#include <dragon_li/util/userConfig.h>
#include <dragon_li/util/memsetDevice.h>
#include <dragon_li/util/ctaOutputAssignment.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace sssp {

template< typename Settings >
class SsspBase {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	typedef typename Settings::SizeType SizeType;
	typedef typename dragon_li::util::GraphCsrDevice<Types> GraphCsrDevice;
	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;

	static const EdgeWeightType INF_WEIGHT = Settings::INF_WEIGHT;

	class UserConfig : public dragon_li::util::UserConfig {
	public:
		double frontierScaleFactor;
		SizeType startVertexId;

		UserConfig(
			bool _verbose,
			bool _veryVerbose,
			double _frontierScaleFactor,
			SizeType _startVertexId = 0) :
				dragon_li::util::UserConfig(_verbose, _veryVerbose),
				frontierScaleFactor(_frontierScaleFactor),
				startVertexId(_startVertexId) {}
	};

	//User control
	bool verbose;
	bool veryVerbose;

	//Graph CSR information
	SizeType vertexCount;
	SizeType edgeCount;
	VertexIdType * devColumnIndices;
	EdgeWeightType * devColumnWeights;
	SizeType * devRowOffsets;
	EdgeWeightType* devSearchDistance;
	std::vector<SizeType> searchDistance;

	//Frontiers for sssp
	VertexIdType startVertexId;
	SizeType maxFrontierSize;
	SizeType frontierSize;
	VertexIdType * devFrontierIn;
	VertexIdType * devFrontierOut;
	bool frontierOverflow;

	//Iteration count
	SizeType iteration;

	//Cta Output Assignement
	CtaOutputAssignment ctaOutputAssignment;

	SsspBase() : 
		verbose(false),
		veryVerbose(false),
		vertexCount(0),
		edgeCount(0),
		devColumnIndices(NULL),
		devColumnWeights(NULL),
		devRowOffsets(NULL),
		devSearchDistance(NULL),
		maxFrontierSize(0),
		frontierSize(0),
		devFrontierIn(NULL),
		devFrontierOut(NULL),
		frontierOverflow(false),
		iteration(0) {}

	virtual int search() = 0;

	virtual int setup(
					GraphCsrDevice &graphCsrDevice,
					UserConfig &userConfig) {
		return setup(
				graphCsrDevice.vertexCount,
				graphCsrDevice.edgeCount,
				graphCsrDevice.devColumnIndices,
				graphCsrDevice.devColumnWeights,
				graphCsrDevice.devRowOffsets,
				userConfig
			);
	}

	virtual int setup(
			SizeType _vertexCount,
			SizeType _edgeCount,
			VertexIdType * _devColumnIndices,
			EdgeWeightType * _devColumnWeights,
			SizeType * _devRowOffsets,
			UserConfig & userConfig
		) {

		verbose = userConfig.verbose;
		veryVerbose = userConfig.veryVerbose;
		startVertexId = userConfig.startVertexId;
	
		if(!_vertexCount || !_edgeCount
			|| !_devColumnIndices
			|| !_devRowOffsets) {
			errorMsg("Invalid parameters when setting up sssp base!");
			return -1;
		}

		vertexCount = _vertexCount;
		edgeCount = _edgeCount;
		devColumnIndices = _devColumnIndices;
		devColumnWeights = _devColumnWeights;
		devRowOffsets = _devRowOffsets;

		cudaError_t retVal;
		if(retVal = cudaMalloc(&devSearchDistance, vertexCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		//initialize search distance as infinity
		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, EdgeWeightType, SizeType>
			(devSearchDistance, INF_WEIGHT, vertexCount))
			return -1;

		//initialize start vertex search distance as 0
		if(retVal = cudaMemset(devSearchDistance + startVertexId, 0, sizeof(EdgeWeightType))) {
			errorCuda(retVal);
			return -1;
		}

		report("frontierSF " << userConfig.frontierScaleFactor);
		maxFrontierSize = (SizeType)((double)edgeCount * userConfig.frontierScaleFactor);
		if(maxFrontierSize <= 0.0) {
			errorMsg("frontier scale factor is too small");
			return -1;
		}

		report("MaxFrontierSize " << maxFrontierSize);

		if(retVal = cudaMalloc(&devFrontierIn, maxFrontierSize * sizeof(VertexIdType))) {
			errorCuda(retVal);
			return -1;
		}

		frontierSize = 1; //always start with one vertex in frontier

		report("Source Vertex Id " << userConfig.startVertexId);
		if(retVal = cudaMemcpy(devFrontierIn, &startVertexId, sizeof(VertexIdType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devFrontierOut, maxFrontierSize * sizeof(VertexIdType))) {
			errorCuda(retVal);
			return -1;
		}

		if(ctaOutputAssignment.setup(0) != 0)
			return -1;

#ifndef NDEBUG
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
		
			if(retVal = cudaMemcpy((void *)(frontier.data()), devFrontierOut, 
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


	virtual int finish() { return 0;}

};

}
}
