#pragma once

#include <vector>
#include <list>
#include <iostream>

#include <hydrazine/interface/debug.h>

#include <dragon_li/util/graphFile.h>
#include <dragon_li/util/graphFileGR.h>
#include <dragon_li/util/graphFileMetis.h>
#include <dragon_li/util/types.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace util {

template < typename Types >
class GraphCsr {

	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

	typedef typename GraphFileVertexData< Types >::GraphFileEdgeData GraphFileEdgeData;

public:
	SizeType vertexCount;
	SizeType edgeCount;

	std::vector< VertexIdType > columnIndices;
	std::vector< EdgeWeightType > columnWeights;
	std::vector< SizeType > rowOffsets;
	std::vector< SizeType > histogram;

	GraphCsr();

	int buildFromFile(const std::string & fileName, const std::string & format);

	int displayCsr(bool veryVerbose);

};

template < typename Types >
GraphCsr< Types >::GraphCsr():
	vertexCount(0), edgeCount(0) {
}

template< typename Types >
int GraphCsr< Types >::buildFromFile(const std::string & fileName, 
	const std::string & format) {
	
	GraphFile< Types > *graphFile;

	if(!format.compare("gr")) {
		graphFile = new GraphFileGR< Types >();
	}
	else if(!format.compare("metis")) {
		graphFile = new GraphFileMetis< Types >();
	}
	else {
		errorMsg("Unrecoginized graph format");
		return -1;
	}


	if(graphFile->build(fileName))
		return -1;

	histogram = graphFile->histogram;

	vertexCount = graphFile->vertexCount;
	edgeCount = graphFile->edgeCount;

	columnIndices.resize(edgeCount);
	columnWeights.resize(edgeCount);
	rowOffsets.resize(vertexCount + 1);

	for(size_t i = 0; i < vertexCount; i++) {
		if(i == 0)
			rowOffsets[0] = 0;
		else {
			rowOffsets[i] = 
				rowOffsets[i - 1] + graphFile->vertices[i - 1].degree;
		}

		std::list< GraphFileEdgeData > &edges = 
			graphFile->vertices[i].edges;

		size_t startId = rowOffsets[i];
		for(typename std::list< GraphFileEdgeData >::iterator 
				edge = edges.begin(); edge != edges.end(); edge++) {
			assertM(edge->fromVertex == i, "from vertex " << edge->fromVertex
				<< " does not match vertexId " << i);

			columnIndices[startId] = edge->toVertex;
			columnWeights[startId++] = edge->weight;
		}

	}

	rowOffsets[vertexCount] = rowOffsets[vertexCount - 1] + 
		graphFile->vertices[vertexCount - 1].degree;

	delete graphFile;

	return 0;
}

template< typename Types >
int GraphCsr< Types >::displayCsr(bool veryVerbose) {
	std::cout << "CSR Graph: vertex count " << vertexCount << ", edge count " << edgeCount << "\n";

	if(veryVerbose) {
		for (size_t vertex = 0; vertex < vertexCount; vertex++) {
			std::cout << vertex << ": ";
			for (size_t edge = rowOffsets[vertex]; edge < rowOffsets[vertex + 1]; edge++) {
				std::cout << columnIndices[edge] << 
				"(" << columnWeights[edge] << ")" << ", ";
			}
			std::cout << "total " << rowOffsets[vertex + 1] - rowOffsets[vertex] << "\n";
		}
	}

	std::cout << "Degree Histogram\n";
	int histogramSize = histogram.size();
	for(int i = -1; i < histogramSize - 1; i++) {
		std::cout << "\tDegree 2^" << i << ": " << histogram[i + 1] << "\n";\
	}
	std::cout << "\n";

	return 0;
}


}

}
