#ifdef ENABLE_CDP

#pragma once

#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>
#include <dragon_li/util/debug.h>
#include <dragon_li/amr/amrRegDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template< typename Settings >
__global__ void amrCdpThreadRefineKernel(
	typename Settings::SizeType processSize,
	typename Settings::DataType * devGridData,
	typename Settings::SizeType * devGridPointer,
	typename Settings::DataType inputEnergy,
	typename Settings::SizeType maxGridDataSize,
	typename Settings::SizeType maxRefineLevel,
	typename Settings::DataType gridRefineThreshold,
	typename Settings::SizeType outputOffset,
	typename Settings::SizeType refineLevel,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	typedef typename Settings::DataType DataType;
	typedef typename Settings::SizeType SizeType;
	const SizeType CDP_THREADS = Settings::CDP_THREADS;
	const SizeType GRID_REFINE_SIZE = Settings::GRID_REFINE_SIZE;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;

	SizeType refineId = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(refineId < processSize) {

		DataType refineData = AmrRegDevice<Settings>::computeTemperature(inputEnergy, refineId); 
		devGridData[outputOffset + refineId] = refineData;
		reportDevice("%d.%d, offset %d, inputEnergy %f, data %f\n", blockIdx.x, threadIdx.x, outputOffset + refineId, inputEnergy, refineData); 

	}

	DataType * devGridDataStart = devGridData + outputOffset;
	SizeType * devGridPointerStart = devGridPointer + outputOffset;
	DataType gridData;
	SizeType gridPointer;
	SizeType refineSize = 0;


	if(refineId < processSize) {
		gridData = devGridDataStart[refineId];
		gridPointer = devGridPointerStart[refineId];

		if(gridPointer == -1) { //Not processed
			devGridPointerStart[refineId] = -2; //processed
			if(gridData >= gridRefineThreshold) {
				refineSize = GRID_REFINE_SIZE; 
			}
			//reportDevice("threshod %f, %f, %d\n", gridRefineThreshold, gridData, refineSize);

		}
	}

	SizeType totalRefineSize = 0;
	SizeType localOffset; //output offset within cta
	localOffset = dragon_li::util::prefixSumCta<CDP_THREADS, SizeType>(refineSize, 
			totalRefineSize);

//    if(threadIdx.x == 0)
//		reportDevice("refineLevel %d, total %d\n", refineLevel, totalRefineSize);

	__shared__ SizeType globalOffset;

	if(threadIdx.x == 0 && totalRefineSize > 0) {
		globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalRefineSize);
		reportDevice("refineLevel %d, total %d, offset %d\n", refineLevel, totalRefineSize, globalOffset);
	}

	__syncthreads();

	if(ctaOutputAssignment.getGlobalSize() > maxGridDataSize) //overflow
		return;


	DataType energy = 0;
	if(refineSize > 0) {
		devGridPointerStart[refineId] = globalOffset + localOffset; //point to child cells
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

}
}

#endif
