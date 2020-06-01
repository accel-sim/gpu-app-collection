#pragma once

#include <dragon_li/util/graphFile.h>

#undef REPORT_BASE
#define REPORT_BASE 1

namespace dragon_li {
namespace util {

template<typename Types> 
class GraphFileMetis : public GraphFile<Types>{
	
	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:
	GraphFileMetis() : GraphFile<Types>() {}
	
	int build(const std::string & fileName) {
		std::ifstream metisFile(fileName.c_str());
		if(!metisFile.is_open()) {
			errorMsg("Error opening file " << fileName);
			return -1;
		}
	
		char keyWord;
	
		metisFile.get(keyWord);
	
		SizeType currentVertex = -1;
		SizeType edgeRead = 0;
	
		while(!metisFile.fail()) {

			if( keyWord == '%' ) { //comment
				while(!metisFile.eof()) {
					metisFile.get(keyWord);
					if(keyWord == '\n')
						break;
				}
			}
			else if(keyWord == ' ' || keyWord == '\t') {
				//skip
			}
			else if(keyWord == '\n') {
				currentVertex++;
			}
			else {
				metisFile.unget();
	
				if(currentVertex == -1) { 
					//Proglem line
					//get vertex and edge count
					metisFile >> this->vertexCount;
					metisFile >> this->edgeCount;

					report("vertex count " << this->vertexCount << ", edge count " << this->edgeCount);

					if(!this->vertexCount || !this->edgeCount) {
						errorMsg("Unrecognized graph file!");
						return -1;
					}
					this->edgeCount *= 2; //directed edges
	
					this->vertices.resize(this->vertexCount);
	
					//initialize vertex data
					for(SizeType i = 0; i < this->vertexCount; i++)
						this->vertices[i] = GraphFileVertexData< Types >(i);
				}
				else {
					//Edge description line for current vertex
					SizeType toVertexId;
	
					metisFile >> toVertexId;
	
					toVertexId--;
	
					edgeRead++;
	
					if(toVertexId >= this->vertexCount) {
						errorMsg("VertexId " << toVertexId << " exceeds limit " << this->vertexCount);
						metisFile.close();
						return -1;
					}
	
					if(edgeRead > this->edgeCount) {
						errorMsg("Exceeding edge count " << this->edgeCount);
						metisFile.close();
						return -1;
					}
	
					//insert edge to vertex data
					this->vertices[currentVertex].insertEdge(toVertexId, 1/*always 1 for metis graph*/);
				}
			}
	
	
			metisFile.get(keyWord);
	
		}
	
		metisFile.close();
	
		if(this->computeHistogram())
			return -1;
	
		return 0;
	
	}
     	
};

}
}
