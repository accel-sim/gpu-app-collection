#pragma once

#include <hydrazine/interface/debug.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace util {

template < typename Types >
class GraphCsrDevice {

	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:

	SizeType vertexCount; // + 1 =lengthOf(rowOffsets)
	SizeType edgeCount; // = lengthOf(columnIndices)

	VertexIdType * devColumnIndices;
	EdgeWeightType * devColumnWeights;
	SizeType * devRowOffsets;

	GraphCsrDevice() : 
		vertexCount(0), 
		edgeCount(0),
		devColumnIndices(NULL),
		devColumnWeights(NULL),
		devRowOffsets(NULL) {}

	int setup(GraphCsr<Types> &graphCsr) {
		return setup(
				graphCsr.vertexCount,
				graphCsr.edgeCount,
				graphCsr.columnIndices,
				graphCsr.columnWeights,
				graphCsr.rowOffsets);
	}

	int setup(
		SizeType _vertexCount,
		SizeType _edgeCount,
		std::vector< VertexIdType > & columnIndices,
		std::vector< EdgeWeightType > & columnWeights,
		std::vector< SizeType > & rowOffsets)  {

		report("GraphCsrDevice setup!");

		vertexCount = _vertexCount;
		edgeCount = _edgeCount;

		assertM( vertexCount + 1 == rowOffsets.size(), 
			"Vertex Count + 1 != rowOffsets.size()");
		assertM( edgeCount == columnIndices.size(),
			"Edge count != columnIndices.size()");

		cudaError_t retVal;
		
		report("devColumnIndices size " << edgeCount * sizeof(VertexIdType));
		if(retVal = cudaMalloc(&devColumnIndices, edgeCount * sizeof(VertexIdType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(
						devColumnIndices, 
						columnIndices.data(), 
						edgeCount * sizeof(VertexIdType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		report("devColumnWeights size " << edgeCount * sizeof(EdgeWeightType));
		if(retVal = cudaMalloc(&devColumnWeights, edgeCount * sizeof(EdgeWeightType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(
						devColumnWeights, 
						columnWeights.data(), 
						edgeCount * sizeof(EdgeWeightType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}


		report("devRowOffsets size " << (vertexCount + 1) * sizeof(SizeType));
		if(retVal = cudaMalloc(&devRowOffsets, (vertexCount + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(
						devRowOffsets, 
						rowOffsets.data(), 
						(vertexCount + 1) * sizeof(SizeType),
						cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

	int finish() {
		cudaError_t retVal;
		if(devColumnIndices) {
			if(retVal = cudaFree(&devColumnIndices)) {
				errorCuda(retVal);
				return -1;
			}
		}
		if(devColumnWeights) {
			if(retVal = cudaFree(&devColumnWeights)) {
				errorCuda(retVal);
				return -1;
			}
		}
		if(devRowOffsets) {
			if(retVal = cudaFree(&devRowOffsets)) {
				errorCuda(retVal);
				return -1;
			}
		}

		report("GraphCsrDevice finishes!");
		return 0;
	}
					
};

} 
}
