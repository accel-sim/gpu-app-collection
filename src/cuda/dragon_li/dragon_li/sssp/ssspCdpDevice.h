#ifdef ENABLE_CDP
#pragma once

#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>

#include <dragon_li/sssp/ssspCdpThread.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace sssp {

template< typename Settings >
class SsspCdpDevice {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;
public:

	static __device__ void ssspCdpCtaExpand(
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

		if(rowLength >= Settings::CDP_THRESHOLD) { //call cdp kernel

#ifndef NDEBUG
			util::cdpKernelCountInc();
#endif

			SizeType CDP_THREADS = Settings::CDP_THREADS;
			SizeType cdpCtas = rowLength >> Settings::CDP_THREADS_BITS;

			cudaStream_t s;
			cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
			ssspCdpThreadExpandKernel<Settings>
				<<< cdpCtas, CDP_THREADS, 0, s>>> (
					rowOffset,
					rowLength,
					srcDistance,
					devColumnIndices,
					devColumnWeights,
					devSearchDistance,
					devFrontierOut,
					ctaOutputAssignment,
					maxFrontierSize);


//			checkErrorDevice();

			rowLength -= (CDP_THREADS * cdpCtas);
			rowOffset += (CDP_THREADS * cdpCtas);
//			localOffset += (CDP_THREADS * cdpCtas);
		}

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


	static __device__ void ssspCdpExpandKernel(
		VertexIdType * devColumnIndices,
		EdgeWeightType * devColumnWeights,
		SizeType * devRowOffsets,
		EdgeWeightType * devSearchDistance,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		SizeType maxFrontierSize,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			ssspCdpCtaExpand(
				ctaWorkAssignment,
				devColumnIndices,
				devColumnWeights,
				devRowOffsets,
				devSearchDistance,
				devFrontierContract,
				devFrontierExpand,
				maxFrontierSize,
				ctaOutputAssignment,
				iteration);
		}


	}

};


template< typename Settings >
__global__ void ssspCdpExpandKernel(
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::EdgeWeightType * devColumnWeights,
	typename Settings::SizeType * devRowOffsets,
	typename Settings::EdgeWeightType * devSearchDistance,
	typename Settings::VertexIdType * devFrontierContract,
	typename Settings::VertexIdType * devFrontierExpand,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment,
	typename Settings::SizeType iteration) {

	SsspCdpDevice< Settings >::ssspCdpExpandKernel(
					devColumnIndices,
					devColumnWeights,
					devRowOffsets,
					devSearchDistance,
					devFrontierContract,
					devFrontierExpand,
					maxFrontierSize,
					frontierSize,
					ctaOutputAssignment,
					iteration);

}

}
}
#endif
