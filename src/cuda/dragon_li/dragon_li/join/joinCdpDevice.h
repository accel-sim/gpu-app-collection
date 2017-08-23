#ifdef ENABLE_CDP
#pragma once

#include <dragon_li/util/threadWorkAssignment.h>
#include <dragon_li/util/primitive.h>

#include <dragon_li/join/joinRegDevice.h>
#include <dragon_li/join/joinCdpThread.h>

#undef REPORT_BASE
#define REPORT_BASE 0


namespace dragon_li {
namespace join {

template< typename Settings >
class JoinCdpDevice {
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;
	static const SizeType CDP_THREADS = Settings::CDP_THREADS;

	typedef typename dragon_li::util::ThreadWorkAssignment<Settings> ThreadWorkAssignment;

public:


    static __device__ SizeType joinCdpJoinBlock(
        SizeType* outLeft, 
        SizeType * outRight,
    	const SizeType leftStartId, 
        const DataType* left,  
        const SizeType leftElements,
    	const SizeType rightStartId, 
        const DataType* right, 
        const SizeType rightElements)
    {
//    	__shared__ SizeType cacheLeft[JOIN_BLOCK_CACHE_SIZE];
//    	__shared__ SizeType cacheRight[JOIN_BLOCK_CACHE_SIZE];
    
    
    	const DataType* r = right + threadIdx.x;
    	
    	DataType rKey = 0;
    	SizeType foundCount = 0;	
    	
        SizeType lower = 0;
    	SizeType higher = 0;
    
    	if(threadIdx.x < rightElements)
    	{
    		rKey = *r;
    		
			lower  = JoinRegDevice<Settings>::joinRegThreadFindLowerBound(rKey, left, leftElements);
    		higher = JoinRegDevice<Settings>::joinRegThreadFindUpperBound(rKey, left, leftElements);
    		
    		foundCount = higher - lower;
    	}
    	
    	SizeType total = 0;
    	SizeType index = util::prefixSumCta<THREADS, DataType>(foundCount, total);
    
    	__syncthreads();

		SizeType startId = 0;

		if(foundCount >= Settings::CDP_THRESHOLD) { //enough parallelism to call CDP
			SizeType cdpCtas = foundCount >> Settings::CDP_THREADS_BITS;

#ifndef NDEBUG
			util::cdpKernelCountInc();
#endif
			
			cudaStream_t s;
			cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	
			joinCdpThreadOutputKernel<Settings>
				<<<cdpCtas, CDP_THREADS, 0, s>>>(
				outLeft,
				outRight,
				index,
				foundCount,
				leftStartId + lower,
				rightStartId + threadIdx.x
			);

			startId += cdpCtas * CDP_THREADS;

		}
		//process remaining
		for(SizeType c = startId; c < foundCount; c++) {
			SizeType leftId = leftStartId + lower + c;
			outLeft[index + c] = leftId;
			
			SizeType rightId = rightStartId + threadIdx.x;
			outRight[index + c] = rightId;
		}

    	
    	return total;
    }

	static __device__ void joinCdpMainJoinKernel(
		const DataType * devJoinInputLeft,
		const SizeType inputCountLeft,
		const DataType * devJoinInputRight,
		const SizeType inputCountRight,
		SizeType * devJoinLeftOutIndicesScattered,
		SizeType * devJoinRightOutIndicesScattered,
		SizeType * devHistogram,
		const SizeType * devLowerBounds,
		const SizeType * devUpperBounds,
		const SizeType * devOutBounds
	) {
    	__shared__ DataType leftCache[THREADS];
    	__shared__ DataType rightCache[THREADS];
    
    	SizeType id = blockIdx.x;
    	
		SizeType partitions = CTAS;
		SizeType partitionSize = (inputCountLeft + partitions - 1) / partitions;
    	
    	SizeType leftId = MIN(partitionSize * id, inputCountLeft);
        const DataType* l    = devJoinInputLeft + leftId;
    	const DataType* lend = devJoinInputLeft + MIN(partitionSize * (id + 1), inputCountLeft);
    
   
        SizeType rightId = devLowerBounds[id];
    	const DataType* r    = devJoinInputRight + rightId;
    	const DataType* rend = devJoinInputRight + devUpperBounds[id];
    
    	
    	SizeType* oBeginLeft = devJoinLeftOutIndicesScattered + devOutBounds[id] - devOutBounds[0];
    	SizeType* oLeft      = oBeginLeft;
    	SizeType* oBeginRight = devJoinRightOutIndicesScattered + devOutBounds[id] - devOutBounds[0];
    	SizeType* oRight      = oBeginRight;
    
    	while(l != lend && r != rend)
    	{
    		SizeType leftBlockSize  = MIN(lend - l, THREADS);
    		SizeType rightBlockSize = MIN(rend - r, THREADS);
    
    		util::memcpyCta<THREADS, DataType>(leftCache,  l, leftBlockSize);
    		util::memcpyCta<THREADS, DataType>(rightCache, r, rightBlockSize);
    
    		__syncthreads();
    
    		DataType lMaxValue = *(leftCache + leftBlockSize - 1);
    		DataType rMinValue = *rightCache;
    	
    		if(lMaxValue < rMinValue)
    		{
                leftId += leftBlockSize;
    			l += leftBlockSize;
    		}
    		else
    		{
    			DataType lMinValue = *leftCache;
    			DataType rMaxValue = *(rightCache + rightBlockSize - 1);
    			
    			if(rMaxValue < lMinValue)
    			{
                    rightId += rightBlockSize;
    				r += rightBlockSize;
    			}
    			else
    			{
   
    				SizeType joined = joinCdpJoinBlock(oLeft, oRight,
    					leftId, leftCache,  leftBlockSize,
    					rightId, rightCache, rightBlockSize);
    				{	
    					oLeft += joined;
                        oRight += joined;

                        SizeType rId = rightId + rightBlockSize;
    					const DataType* ri = r + rightBlockSize;
    			
    					for(; ri != rend;)
    					{
    						rightBlockSize = MIN(THREADS, rend - ri);
    						util::memcpyCta<THREADS, DataType>(rightCache, ri, rightBlockSize);
    					
    						__syncthreads();
    						rMinValue = *rightCache;
       				
    						if(lMaxValue < rMinValue) break;
    
    						joined = joinCdpJoinBlock(oLeft, oRight,
    							leftId, leftCache,  leftBlockSize,
    							rId, rightCache, rightBlockSize);

    						oLeft += joined;
                            oRight += joined;

                            rId += rightBlockSize;
    						ri += rightBlockSize;
    
    					}
    				}
            	
                    leftId += leftBlockSize;
    				l += leftBlockSize;
    			}
    			__syncthreads();
    		}
    	}
    
    
    	if(threadIdx.x == 0) {
			devHistogram[id] = oLeft - oBeginLeft;
		}

	}

};


template< typename Settings >
__global__ void joinCdpMainJoinKernel(
	const typename Settings::DataType * devJoinInputLeft,
	const typename Settings::SizeType inputCountLeft,
	const typename Settings::DataType * devJoinInputRight,
	const typename Settings::SizeType inputCountRight,
	typename Settings::SizeType * devJoinLeftOutIndicesScattered,
	typename Settings::SizeType * devJoinRightOutIndicesScattered,
	typename Settings::SizeType * devHistogram,
	const typename Settings::SizeType * devLowerBounds,
	const typename Settings::SizeType * devUpperBounds,
	const typename Settings::SizeType * devOutBounds
	) {

	JoinCdpDevice< Settings >::joinCdpMainJoinKernel(
		devJoinInputLeft,
		inputCountLeft,
		devJoinInputRight,
		inputCountRight,
		devJoinLeftOutIndicesScattered,
		devJoinRightOutIndicesScattered,
		devHistogram,
		devLowerBounds,
		devUpperBounds,
		devOutBounds);
}

}
}
#endif
