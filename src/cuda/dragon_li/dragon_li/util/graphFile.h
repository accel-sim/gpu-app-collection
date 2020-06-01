#pragma once

#include <list>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>

#include <hydrazine/interface/debug.h>

#include <dragon_li/util/debug.h>
#include <dragon_li/util/types.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace util {


template<typename Types>
class GraphFileVertexData {
	
	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:
	class GraphFileEdgeData {
	public:
		VertexIdType fromVertex;
		VertexIdType toVertex;
		EdgeWeightType weight;
	
		GraphFileEdgeData() {}
		GraphFileEdgeData(VertexIdType from, VertexIdType to, EdgeWeightType w):
			fromVertex(from), toVertex(to), weight(w) {}
	};

	VertexIdType vertexId;
	SizeType degree; //count of outgoding edges

	//Edge list
	std::list< GraphFileEdgeData > edges;

	GraphFileVertexData() : vertexId(-1), degree(0) {}

	GraphFileVertexData(VertexIdType id, SizeType d = 0) : vertexId(id), degree(d) {}

	GraphFileVertexData & operator= (const GraphFileVertexData & vertex) {
		if (this == &vertex) return *this;
		degree = vertex.degree;
		vertexId = vertex.vertexId;
		edges = vertex.edges;
		return *this;
	}

	GraphFileEdgeData & insertEdge(
		VertexIdType to, EdgeWeightType w) {
		edges.push_back(GraphFileEdgeData(vertexId, to, w)); 
		degree++;
		return edges.back();
	}
	GraphFileEdgeData & insertEdge(
		GraphFileEdgeData & edge) {
		edges.push_back(edge);
		degree++;
		return edges.back();
	}
};

template< typename Types >
class GraphFile {

	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:
	SizeType vertexCount;
	SizeType edgeCount;

	std::vector< GraphFileVertexData< Types > > vertices;

	std::vector< SizeType > histogram;
	
	GraphFile() : vertexCount(0), edgeCount(0) {}

	virtual int build(const std::string & fileName) = 0;

	int computeHistogram() {

		if(vertices.empty())
			errorMsg("Graph has not been built!");

		SizeType maxLogDegree = -1;
		histogram.resize(maxLogDegree + 2, 0);
		for(SizeType i = 0; i < vertexCount; i++) {
			SizeType degree = vertices[i].degree;
			SizeType logDegree = -1;
			while(degree > 0) {
				degree >>= 1;
				logDegree++;
			}
			report("Log degree " << logDegree);
			if(logDegree > maxLogDegree) {
				maxLogDegree = logDegree;
				histogram.resize(maxLogDegree + 2, 0);
			}
			histogram[logDegree + 1]++;
		}
	
		return 0;
	}
};
     	
}    	
}    	
     	
