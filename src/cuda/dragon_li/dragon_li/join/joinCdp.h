#ifdef ENABLE_CDP
#pragma once

#include <dragon_li/join/joinReg.h>
#include <dragon_li/join/joinCdpDevice.h>

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinCdp : public JoinReg< Settings > {

	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:
	
	JoinCdp() : JoinReg< Settings >() {}

	int setup(JoinData<Types> & joinData,
				typename JoinBase<Settings>::UserConfig & userConfig) {


		if(JoinReg<Settings>::setup(joinData, userConfig))
			return -1;

        cudaError_t result;
		if(result = cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 131072)) {

			errorCuda(result);
			return -1;
		}

		return 0;
	}

	
	int mainJoin() {

		joinCdpMainJoinKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devJoinInputLeft,
				this->inputCountLeft,
				this->devJoinInputRight,
				this->inputCountRight,
				this->devJoinLeftOutIndicesScattered,
				this->devJoinRightOutIndicesScattered,
				this->devHistogram,
				this->devLowerBounds,
				this->devUpperBounds,
				this->devOutBounds);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

#ifndef NDEBUG
		util::printCdpKernelCount();
#endif

//		std::vector<SizeType> upper(estJoinOutCount);
//		cudaMemcpy(upper.data(), devJoinRightOutIndicesScattered, (estJoinOUtCount) * sizeof(SizeType), cudaMemcpyDeviceToHost);
//		for(int i = 0; i < 1200; i++)
//			std::cout << "u" << i << ": " << upper[i] << "\n";
		
		return 0;
	}

};

}
}
#endif
