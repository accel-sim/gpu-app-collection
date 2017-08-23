#ifdef ENABLE_CDP
#pragma once

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace join {

template< typename Settings >
__global__ void joinCdpThreadOutputKernel(
	typename Settings::SizeType *devOutLeft,
	typename Settings::SizeType *devOutRight,
	typename Settings::SizeType outputOffset,
	typename Settings::SizeType outputCount,
	typename Settings::SizeType leftStartId,
	typename Settings::SizeType rightId) {
		
	typedef typename Settings::SizeType SizeType;

	SizeType outputId = threadIdx.x + blockIdx.x * blockDim.x;
	if(outputId < outputCount) {

		devOutLeft[outputOffset + outputId]	= leftStartId + outputId;
		devOutRight[outputOffset + outputId] = rightId;

	}
}

}
}
#endif
