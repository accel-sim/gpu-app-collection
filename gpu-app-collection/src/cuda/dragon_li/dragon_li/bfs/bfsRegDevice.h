#pragma once

#include <dragon_li/util/primitive.h>
#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsRegDevice {
	
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::MaskType MaskType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;

public:
	static __device__ void bfsRegCtaExpand(
		CtaWorkAssignment &ctaWorkAssignment,
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		SizeType * devSearchDistance,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		SizeType maxFrontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {


		VertexIdType vertexId = -1;
		SizeType rowOffset = -1;
		SizeType nextRowOffset = -1;
		SizeType rowLength = 0;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			vertexId = devFrontierContract[ctaWorkAssignment.workOffset + threadIdx.x];

			SizeType searchDistance = devSearchDistance[vertexId];
			if(searchDistance == -1)
				devSearchDistance[vertexId] = iteration;
			rowOffset = devRowOffsets[vertexId];
			nextRowOffset = devRowOffsets[vertexId + 1];
			rowLength = nextRowOffset - rowOffset;
		}

		SizeType totalOutputCount;
		SizeType localOffset; //output offset within cta
		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(rowLength, 
				totalOutputCount);

		__shared__ SizeType globalOffset;

		if(threadIdx.x == 0 && totalOutputCount > 0) {
			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalOutputCount);
		}

		__syncthreads();

		if(ctaOutputAssignment.getGlobalSize() > maxFrontierSize) //overflow
			return;

		for(SizeType columnId = 0; columnId < rowLength; columnId++) {
			VertexIdType neighborVertexId = devColumnIndices[rowOffset + columnId];
			devFrontierExpand[globalOffset + localOffset + columnId] = neighborVertexId;
			reportDevice("%d.%d, neighborid %d, outputoffset %d", blockIdx.x, threadIdx.x, neighborVertexId, globalOffset + localOffset + columnId);
		}
		
	}


	static __device__ void bfsRegCtaContract(
		CtaWorkAssignment &ctaWorkAssignment,
		MaskType * devVisitedMasks,
		VertexIdType * devOriginalFrontier,
		VertexIdType * devContractedFrontier,
		CtaOutputAssignment & ctaOutputAssignment) {

		VertexIdType vertexId = -1;

		if(threadIdx.x < ctaWorkAssignment.workSize) {

			vertexId = devOriginalFrontier[ctaWorkAssignment.workOffset + threadIdx.x];

			SizeType maskLocation = vertexId >> Settings::MASK_BITS; 

			SizeType maskBitLocation = 1 << (vertexId & Settings::MASK_MASK);

			MaskType entireMask = devVisitedMasks[maskLocation];

			if(entireMask & maskBitLocation) { //visited
				vertexId = -1;	
			}
			else { //not visited
				entireMask |= maskBitLocation;
				devVisitedMasks[maskLocation] = entireMask;
			}
		}

		SizeType validVertex = (vertexId == -1 ? 0 : 1);
		SizeType totalOutputCount;
		SizeType localOffset;
		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(validVertex,
				totalOutputCount);

		__shared__ SizeType globalOffset;

		if(threadIdx.x == 0 && totalOutputCount > 0) {
			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalOutputCount);
		}

		__syncthreads();
		
		if(vertexId != -1) {
			devContractedFrontier[globalOffset + localOffset] = vertexId;
			reportDevice("%d.%d, vertex %d, outputoffset %d", blockIdx.x, threadIdx.x, vertexId, globalOffset + localOffset);
		}

	}

	static __device__ void bfsRegExpandKernel(
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		SizeType * devSearchDistance,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		SizeType maxFrontierSize,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			if(ctaWorkAssignment.workSize > 0)
				bfsRegCtaExpand(
					ctaWorkAssignment,
					devColumnIndices,
					devRowOffsets,
					devSearchDistance,
					devFrontierContract,
					devFrontierExpand,
					maxFrontierSize,
					ctaOutputAssignment,
					iteration);
		}


	}

	static __device__ void bfsRegContractKernel(
		MaskType * devVisitedMasks,
		VertexIdType * devOriginalFrontier,
		VertexIdType * devContractedFrontier,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);

		while(ctaWorkAssignment.workOffset < frontierSize) {

			ctaWorkAssignment.getCtaWorkAssignment();

			if(ctaWorkAssignment.workSize > 0)
				bfsRegCtaContract(
					ctaWorkAssignment,
					devVisitedMasks,
					devOriginalFrontier,
					devContractedFrontier,
					ctaOutputAssignment);

		}

	}
		
};


template< typename Settings >
__global__ void bfsRegExpandKernel(
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::SizeType * devRowOffsets,
	typename Settings::SizeType * devSearchDistance,
	typename Settings::VertexIdType * devFrontierContract,
	typename Settings::VertexIdType * devFrontierExpand,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment,
	typename Settings::SizeType iteration) {

	BfsRegDevice< Settings >::bfsRegExpandKernel(
					devColumnIndices,
					devRowOffsets,
					devSearchDistance,
					devFrontierContract,
					devFrontierExpand,
					maxFrontierSize,
					frontierSize,
					ctaOutputAssignment,
					iteration);

}

template< typename Settings >
__global__ void bfsRegContractKernel(
	typename Settings::MaskType * devVisistedMasks,
	typename Settings::VertexIdType * devOriginalFrontier,
	typename Settings::VertexIdType * devContractedFrontier,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	BfsRegDevice< Settings >::bfsRegContractKernel(
					devVisistedMasks,
					devOriginalFrontier,
					devContractedFrontier,
					frontierSize,
					ctaOutputAssignment);
	
	
}

}
}
