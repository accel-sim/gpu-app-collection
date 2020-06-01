#pragma once

#include <queue>

namespace dragon_li {
namespace bfs {

template< typename Types >
class BfsCpu {
	
public:

	typedef typename Types::SizeType SizeType;
	typedef typename Types::VertexIdType VertexIdType;
	
	static std::vector<SizeType> cpuSearchDistance;

	static int bfsCpu(dragon_li::util::GraphCsr< Types > & graph) {

		cpuSearchDistance.resize(graph.vertexCount, -1);
	
		std::queue<SizeType> bfsQueue;
		bfsQueue.push(0); //start from 0;
		cpuSearchDistance[0] = 0;
		SizeType depth = 0;
	
		while(!bfsQueue.empty()) {
	
			VertexIdType nextVertex = bfsQueue.front();
			bfsQueue.pop();
			
			SizeType vertexDistance = cpuSearchDistance[nextVertex];
	
			SizeType rowStart = graph.rowOffsets[nextVertex];
			SizeType rowEnd =  graph.rowOffsets[nextVertex + 1];

			for(SizeType i = rowStart; i < rowEnd; i++) {
				VertexIdType neighborVertex = graph.columnIndices[i];
				if(cpuSearchDistance[neighborVertex] == -1) {
					cpuSearchDistance[neighborVertex] = vertexDistance + 1;
					bfsQueue.push(neighborVertex);
					depth = vertexDistance + 1;
				}
			}

		}

		std::cout << "CPU search depth = " << depth + 1 << "\n";

		return 0;
	}

};

template<typename Types>
std::vector<typename Types::SizeType> BfsCpu<Types>::cpuSearchDistance;

}
}
