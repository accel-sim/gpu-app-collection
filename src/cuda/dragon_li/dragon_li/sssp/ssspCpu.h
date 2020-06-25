#pragma once

#include <queue>

namespace dragon_li {
namespace sssp {

template< typename Settings >
class SsspCpu {
	
public:

	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	
	static const EdgeWeightType INF_WEIGHT = Settings::INF_WEIGHT;
	
	static std::vector<EdgeWeightType> cpuSearchDistance;

	static int ssspCpu(dragon_li::util::GraphCsr< Types > & graph, VertexIdType srcVertexId, EdgeWeightType infWeight) {

		cpuSearchDistance.resize(graph.vertexCount,infWeight);
	
		std::queue<SizeType> ssspQueue;
		ssspQueue.push(srcVertexId); //start from src;
		cpuSearchDistance[srcVertexId] = 0;

		ssspQueue.push(-1); //Depth Marker
		SizeType depth = 0;
	
		while(!ssspQueue.empty()) {
	
			VertexIdType nextVertex = ssspQueue.front();
			ssspQueue.pop();

			if(nextVertex == -1) { //Depth Marker
				depth++;

				if(!ssspQueue.empty()) { //Not the last depth marker
					ssspQueue.push(-1); //Push a new depth marker
				}
				continue;
			}
			
			SizeType vertexDistance = cpuSearchDistance[nextVertex];
	
			SizeType rowStart = graph.rowOffsets[nextVertex];
			SizeType rowEnd =  graph.rowOffsets[nextVertex + 1];

			for(SizeType i = rowStart; i < rowEnd; i++) {
				VertexIdType neighborVertex = graph.columnIndices[i];
				EdgeWeightType neighborWeight = graph.columnWeights[i];
				EdgeWeightType neighborDistance = cpuSearchDistance[neighborVertex];
				EdgeWeightType newDistance = vertexDistance + neighborWeight;
				if(newDistance < neighborDistance) {
					cpuSearchDistance[neighborVertex] = newDistance;
					ssspQueue.push(neighborVertex);
				}
			}

		}

		std::cout << "CPU search depth = " << depth << "\n";

		return 0;
	}

};

template<typename Settings>
std::vector<typename Settings::EdgeWeightType> SsspCpu<Settings>::cpuSearchDistance;

}
}
