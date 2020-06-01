#ifdef ENABLE_CDP
#pragma once

#include <dragon_li/sssp/ssspReg.h>
#include <dragon_li/sssp/ssspCdpDevice.h>

namespace dragon_li {
namespace sssp {

template< typename Settings >
class SsspCdp : public SsspReg< Settings > {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	typedef typename dragon_li::util::GraphCsrDevice<typename Settings::Types> GraphCsrDevice;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:
	
	SsspCdp() : SsspReg< Settings >() {}

    int setup(
	    	GraphCsrDevice &graphCsrDevice,
			typename SsspReg<Settings>::UserConfig &userConfig) {
		return setup(
				graphCsrDevice.vertexCount,
				graphCsrDevice.edgeCount,
				graphCsrDevice.devColumnIndices,
				graphCsrDevice.devColumnWeights,
				graphCsrDevice.devRowOffsets,
				userConfig
			);
	}

    int setup(
		SizeType _vertexCount,
		SizeType _edgeCount,
		VertexIdType * _devColumnIndices,
		EdgeWeightType * _devColumnWeights,
		SizeType * _devRowOffsets,
		typename SsspReg<Settings>::UserConfig & userConfig) {
       
        int status = 0;
 
        //Base class setup
        status = SsspReg< Settings >::setup(
            _vertexCount,
            _edgeCount,
            _devColumnIndices,
			_devColumnWeights,
            _devRowOffsets,
            userConfig
        );
        if(status)
            return status;
		
        cudaError_t result;
		if(result = cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 131072)) {
			errorCuda(result);
			return -1;
		}

        return 0;
    }

	int expand() {

		ssspCdpExpandKernel< Settings >
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

#ifndef NDEBUG
		util::printCdpKernelCount();
#endif

		return 0;

	}
};

}
}
#endif
