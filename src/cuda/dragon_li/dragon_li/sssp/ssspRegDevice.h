#pragma once

//#include <dragon_li/util/primitive.h>
#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace sssp {

template< typename Settings >
class SsspRegDevice {
	
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	typedef typename Settings::SizeType SizeType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;

public:
	static __device__ void ssspRegCtaExpand(
		CtaWorkAssignment &ctaWorkAssignment,
		VertexIdType * devColumnIndices,
		EdgeWeightType * devColumnWeights,
		SizeType * devRowOffsets,
		EdgeWeightType * devSearchDistance,
		VertexIdType * devFrontierIn,
		VertexIdType * devFrontierOut,
		SizeType maxFrontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {


		VertexIdType vertexId = -1;
		SizeType rowOffset = -1;
		SizeType nextRowOffset = -1;
		SizeType rowLength = 0;
		EdgeWeightType srcDistance = 0;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			vertexId = devFrontierIn[ctaWorkAssignment.workOffset + threadIdx.x];

			srcDistance = devSearchDistance[vertexId];
	//		if(searchDistance == -1)
	//			devSearchDistance[vertexId] = iteration;
			rowOffset = devRowOffsets[vertexId];
			nextRowOffset = devRowOffsets[vertexId + 1];
			rowLength = nextRowOffset - rowOffset;
		}

//		SizeType totalOutputCount;
//		SizeType localOffset; //output offset within cta
//		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(rowLength, 
//				totalOutputCount);
//
//		__shared__ SizeType globalOffset;

//		if(threadIdx.x == 0 && totalOutputCount > 0) {
//			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalOutputCount);
//		}
//
//		__syncthreads();
//
//		if(ctaOutputAssignment.getGlobalSize() > maxFrontierSize) //overflow
//			return;

		for(SizeType columnId = 0; columnId < rowLength; columnId++) {
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
					reportDevice("%d.%d, neighborid %d, outputoffset %d", blockIdx.x, threadIdx.x, neighborVertexId, globalOffset);
				}
			} 
		}
		
	}

	static __device__ void ssspRegExpandKernel(
		VertexIdType * devColumnIndices,
		EdgeWeightType * devColumnWeights,
		SizeType * devRowOffsets,
		EdgeWeightType * devSearchDistance,
		VertexIdType * devFrontierIn,
		VertexIdType * devFrontierOut,
		SizeType maxFrontierSize,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			if(ctaWorkAssignment.workSize > 0)
				ssspRegCtaExpand(
					ctaWorkAssignment,
					devColumnIndices,
					devColumnWeights,
					devRowOffsets,
					devSearchDistance,
					devFrontierIn,
					devFrontierOut,
					maxFrontierSize,
					ctaOutputAssignment,
					iteration);
		}


	}

};


template< typename Settings >
__global__ void ssspRegExpandKernel(
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::EdgeWeightType * devColumnWeights,
	typename Settings::SizeType * devRowOffsets,
	typename Settings::EdgeWeightType * devSearchDistance,
	typename Settings::VertexIdType * devFrontierIn,
	typename Settings::VertexIdType * devFrontierOut,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment,
	typename Settings::SizeType iteration) {

	SsspRegDevice< Settings >::ssspRegExpandKernel(
					devColumnIndices,
					devColumnWeights,
					devRowOffsets,
					devSearchDistance,
					devFrontierIn,
					devFrontierOut,
					maxFrontierSize,
					frontierSize,
					ctaOutputAssignment,
					iteration);

}


}
}
