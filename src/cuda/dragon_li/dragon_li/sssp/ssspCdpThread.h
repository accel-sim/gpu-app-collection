#ifdef ENABLE_CDP
#pragma once

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace sssp {

template< typename Settings >
__global__ void ssspCdpThreadExpandKernel(
	typename Settings::SizeType rowOffset,
	typename Settings::SizeType rowLength,
	typename Settings::EdgeWeightType srcDistance,
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::EdgeWeightType * devColumnWeights,
	typename Settings::EdgeWeightType * devSearchDistance,
	typename Settings::VertexIdType * devFrontierOut,
	typename util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment,
	typename Settings::SizeType maxFrontierSize) {
		
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;

	SizeType columnId = threadIdx.x + blockIdx.x * blockDim.x;
	if(columnId < rowLength) {

			VertexIdType neighborVertexId = devColumnIndices[rowOffset + columnId];
			EdgeWeightType weight = devColumnWeights[rowOffset + columnId];

			EdgeWeightType dstDistance = devSearchDistance[neighborVertexId];
			EdgeWeightType newDstDistance = srcDistance + weight;

			if(newDstDistance < dstDistance) {
				EdgeWeightType prevDstDistance = atomicMin(&devSearchDistance[neighborVertexId], newDstDistance);

				if(newDstDistance < prevDstDistance) {
					SizeType globalOffset = ctaOutputAssignment.getCtaOutputAssignment(1);
				if(ctaOutputAssignment.getGlobalSize() > maxFrontierSize) //overflow
					return;
					devFrontierOut[globalOffset] = neighborVertexId;
					reportDevice("CDP %d.%d, neighborid %d, outputoffset %d", blockIdx.x, threadIdx.x, neighborVertexId, globalOffset);
				}
			} 

//		VertexIdType expandedVertex = devColumnIndices[rowOffset + columnId];
//		devFrontierExpand[outputOffset + columnId]	= expandedVertex;
//
//		reportDevice("CDP %d.%d: vertex %d, outputoffset %d", 
//			blockIdx.x, threadIdx.x, expandedVertex, outputOffset + columnId);
	}
}

}
}
#endif
