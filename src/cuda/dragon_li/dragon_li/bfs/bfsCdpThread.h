#ifdef ENABLE_CDP
#pragma once

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace bfs {

template< typename Settings >
__global__ void bfsCdpThreadExpandKernel(
	typename Settings::SizeType rowOffset,
	typename Settings::SizeType rowLength,
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::VertexIdType * devFrontierExpand,
	typename Settings::SizeType outputOffset) {
		
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::VertexIdType VertexIdType;

	SizeType columnId = threadIdx.x + blockIdx.x * blockDim.x;
	if(columnId < rowLength) {

		VertexIdType expandedVertex = devColumnIndices[rowOffset + columnId];
		devFrontierExpand[outputOffset + columnId]	= expandedVertex;

		reportDevice("CDP %d.%d: vertex %d, outputoffset %d", 
			blockIdx.x, threadIdx.x, expandedVertex, outputOffset + columnId);
	}
}

}
}
#endif
