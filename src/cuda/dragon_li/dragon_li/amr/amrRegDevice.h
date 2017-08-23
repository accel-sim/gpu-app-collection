#pragma once

#include <dragon_li/util/debug.h>
#include <dragon_li/util/primitive.h>
#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrRegDevice {

	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;
	static const SizeType GRID_REFINE_SIZE = Settings::GRID_REFINE_SIZE;
	static const SizeType GRID_REFINE_X = Settings::GRID_REFINE_X;
	static const SizeType GRID_REFINE_Y = Settings::GRID_REFINE_Y;
	static const SizeType GRID_REFINE_Z = Settings::GRID_REFINE_Z;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;

public:

	static __host__ __device__ DataType computeEnergy(DataType temperature) {
		DataType energy = 4.6 * temperature * temperature / 100000.0 + 
				0.14816 * temperature + 962.62;

		return energy;
	}

	static __host__ __device__ DataType computeTemperature(DataType energy, SizeType refineId) {

			SizeType distanceX, distanceY, distanceZ, remainXY;
			distanceZ = refineId / (GRID_REFINE_X * GRID_REFINE_Y);
			remainXY = refineId % (GRID_REFINE_X * GRID_REFINE_Y);
			distanceY = remainXY / GRID_REFINE_X;
			distanceX = remainXY % GRID_REFINE_X;

			DataType refinedEnergy = (energy - 962.62) / (distanceX * distanceX + 
				distanceY * distanceY + distanceZ * distanceZ + 2.0) + 962.62;

			DataType refinedTemp = (-0.1486 + sqrt(0.1486 * 0.1486 - 4 * 4.6 * 
				(962.62 - refinedEnergy) / 100000)) / 2 / 4.6 * 100000;

			return refinedTemp;
	}

	static __device__ void amrRegCtaRefine(
		CtaWorkAssignment & ctaWorkAssignment,
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType processGridOffset,
		SizeType maxGridDataSize,
		SizeType maxRefineLevel,
		DataType gridRefineThreshold,
		CtaOutputAssignment & ctaOutputAssignment) {

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

			}
		}

		SizeType totalRefineSize;
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
			energy = computeEnergy(gridData);
		}


		for(SizeType refineId = 0; refineId < refineSize; refineId++) {

			DataType refineData = computeTemperature(energy, refineId); 
			devGridData[globalOffset + localOffset + refineId] = refineData;
			reportDevice("%d.%d, offset %d, energy %f, data %f\n", blockIdx.x, threadIdx.x, globalOffset + localOffset + refineId, energy, refineData); 
		}


	}

	static __device__ void amrRegRefineKernel(
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType maxGridDataSize,
		SizeType processGridOffset,
		SizeType processGridSize,
		SizeType maxRefineLevel,
		DataType gridRefineThreshold,
		CtaOutputAssignment & ctaOutputAssignment) {


		CtaWorkAssignment ctaWorkAssignment(processGridSize);
	
		while(ctaWorkAssignment.workOffset < processGridSize) {
			ctaWorkAssignment.getCtaWorkAssignment();
//			if(threadIdx.x == 0)
//				reportDevice("%d: %d\n", blockIdx.x, ctaWorkAssignment.workOffset);	
			amrRegCtaRefine(
				ctaWorkAssignment,
				devGridData,
				devGridPointer,
				processGridOffset,
				maxGridDataSize,
				maxRefineLevel,
				gridRefineThreshold,
				ctaOutputAssignment);
		}

	}
};

template< typename Settings >
__global__  void amrRegRefineKernel(
	typename Settings::DataType * devGridData,
	typename Settings::SizeType * devGridPointer,
	typename Settings::SizeType maxGridDataSize,
	typename Settings::SizeType processGridOffset,
	typename Settings::SizeType processGridSize,
	typename Settings::SizeType maxRefineLevel,
	typename Settings::DataType gridRefineThreshold,
	typename dragon_li::util::CtaOutputAssignment<typename Settings::SizeType> ctaOutputAssignment) {

	AmrRegDevice< Settings >::amrRegRefineKernel(
		devGridData,
		devGridPointer,
		maxGridDataSize,
		processGridOffset,
		processGridSize,
		maxRefineLevel,
		gridRefineThreshold,
		ctaOutputAssignment);
}

}
}
