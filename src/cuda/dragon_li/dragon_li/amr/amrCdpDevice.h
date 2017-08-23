#ifdef ENABLE_CDP

#pragma once

#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>
#include <dragon_li/util/debug.h>

#include <dragon_li/amr/amrRegDevice.h>
#include <dragon_li/amr/amrCdpThread.h>

#undef REPORT_BASE
#define REPORT_BASE 1

namespace dragon_li {
namespace amr {

template< typename Settings>
class AmrCdpDevice {

	typedef typename Settings::DataType DataType;
	typedef typename Settings::SizeType SizeType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;
	static const SizeType GRID_REFINE_SIZE = Settings::GRID_REFINE_SIZE;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;
public:

	static __device__ void amrCdpCtaRefine(
		CtaWorkAssignment & ctaWorkAssignment,
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType processGridOffset,
		SizeType maxGridDataSize,
		SizeType maxRefineLevel,
		SizeType refineLevel,
		DataType gridRefineThreshold,
		CtaOutputAssignment & ctaOutputAssignment
		) {

		DataType * devGridDataStart = devGridData + processGridOffset;
		SizeType * devGridPointerStart = devGridPointer + processGridOffset;
		DataType gridData;
		SizeType gridPointer;
		SizeType refineSize = 0;

		SizeType threadWorkOffset = ctaWorkAssignment.workOffset + threadIdx.x;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			gridData = devGridDataStart[threadWorkOffset];
			gridPointer = devGridPointerStart[threadWorkOffset];

			if(gridPointer == -1) { //Not processed
				devGridPointerStart[threadWorkOffset] = -2; //processed
				if(gridData >= gridRefineThreshold) {
					refineSize = GRID_REFINE_SIZE; 
				}
			//	reportDevice("threshod %f, %f, %d\n", gridRefineThreshold, gridData, refineSize);

			}
		}

		SizeType totalRefineSize = 0;
		SizeType localOffset; //output offset within cta
		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(refineSize, 
				totalRefineSize);

		__shared__ SizeType globalOffset;

		if(threadIdx.x == 0 && totalRefineSize > 0) {
			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalRefineSize);
		}

		__syncthreads();

		if(ctaOutputAssignment.getGlobalSize() > maxGridDataSize) //overflow
			return;


		DataType energy = 0;
		if(refineSize > 0) {
			devGridPointerStart[threadWorkOffset] = globalOffset + localOffset; //point to child cells
			energy = AmrRegDevice<Settings>::computeEnergy(gridData);
		}


		refineLevel++;
		if(refineLevel < maxRefineLevel) {

			if(refineSize > 0) {
				//reportDevice("launch data %f, %d\n", gridData, refineSize);
				SizeType cdpCtas = (refineSize + Settings::CDP_THREADS - 1) >> Settings::CDP_THREADS_BITS;
				//reportDevice("%d.%d: cdpCtas %d\n", blockIdx.x, threadIdx.x, cdpCtas);
#ifndef NDEBUG
            util::cdpKernelCountInc();
#endif
				cudaStream_t s;
				cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
				amrCdpThreadRefineKernel<Settings>
					<<< cdpCtas, Settings::CDP_THREADS, 0, s >>> (
						refineSize,
						devGridData,
						devGridPointer,
						energy,
						maxGridDataSize,
						maxRefineLevel,
						gridRefineThreshold,
						globalOffset + localOffset,
						refineLevel,
						ctaOutputAssignment
						);
				//checkErrorDevice();
				//reportDevice("End launch threshod %f, %d\n", gridData, refineSize);
			}
		}

	}
	
	static __device__ void amrCdpRefineKernel(
	DataType * devGridData,
	SizeType * devGridPointer,
	SizeType maxGridDataSize,
	SizeType activeGridSize,
	SizeType maxRefineLevel,
	DataType gridRefineThreshold,
	CtaOutputAssignment ctaOutputAssignment) {

		SizeType refineLevel = 0;

		CtaWorkAssignment ctaWorkAssignment(activeGridSize);


		while(ctaWorkAssignment.workOffset < activeGridSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			if(ctaWorkAssignment.workSize >  0) {
				amrCdpCtaRefine(
					ctaWorkAssignment,
					devGridData,
					devGridPointer,
					0,
					maxGridDataSize,
					maxRefineLevel,
					refineLevel,
					gridRefineThreshold,
					ctaOutputAssignment
					);
			}
		}


	}

};


template< typename Settings >
__global__ void amrCdpRefineKernel(
	typename Settings::DataType * devGridData,
	typename Settings::SizeType * devGridPointer,
	typename Settings::SizeType maxGridDataSize,
	typename Settings::SizeType activeGridSize,
	typename Settings::SizeType maxRefineLevel,
	typename Settings::DataType gridRefineThreshold,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	AmrCdpDevice< Settings >::amrCdpRefineKernel(
		devGridData,
		devGridPointer,
		maxGridDataSize,
		activeGridSize,
		maxRefineLevel,
		gridRefineThreshold,
		ctaOutputAssignment);

}


}
}

#endif
