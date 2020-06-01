#pragma once

#include <dragon_li/util/graphFile.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace util {

template<typename Types> 
class GraphFileGR : public GraphFile<Types>{
	
	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:
	GraphFileGR() : GraphFile<Types>() {}
	
	int build(const std::string & fileName) {

		std::ifstream grFile(fileName.c_str());
		if(!grFile.is_open()) {
			errorMsg("Error opening file " << fileName);
			return -1;
		}
	
		char keyWord;
		char tmpFileBuf[256];
		SizeType edgeRead = 0;
	
		grFile >> keyWord;
	
		while(!grFile.fail()) {
	
			if( keyWord == 'p') { 
			
				//Problem line, format: p sp total_vertex_count total_edge_count
	
				grFile >> tmpFileBuf;
	
				//p followed by sp
				if(!std::strcmp(tmpFileBuf, "sp")) {
	
					//get vertex and edge count
					grFile >> this->vertexCount;
					grFile >> this->edgeCount;
	
					this->vertices.resize(this->vertexCount);
	
					//initialize vertex data
					for(SizeType i = 0; i < this->vertexCount; i++)
						this->vertices[i] = GraphFileVertexData< Types >(i);
				}
				else{
					errorMsg("Error GR File format for " << fileName);
					grFile.close();
					return -1;
				}
	
			}
			else if( keyWord == 'a') { //Arc or edge description line
				//format: a from_vertex to_vertex edge_weight
	
				SizeType fromVertexId, toVertexId;
				EdgeWeightType weight;
	
				//get edge
				grFile >> fromVertexId >> toVertexId >> weight;
	
				//GR File always start vertex ID from 1
				fromVertexId--;
				toVertexId--;
	
				edgeRead++;
	
				//check boundary
				if(fromVertexId >= this->vertexCount) {
					errorMsg("VertexId " << fromVertexId << " exceeds limit");
					grFile.close();
					return -1;
				}
				if(toVertexId >= this->vertexCount) {
					errorMsg("VertexId " << toVertexId << " exceeds limit");
					grFile.close();
					return -1;
				}
	
				if(edgeRead > this->edgeCount) {
					errorMsg("Exceeding edge count " << this->edgeCount);
					grFile.close();
					return -1;
				}
	
				//insert edge to vertex data
				this->vertices[fromVertexId].insertEdge(toVertexId, weight);
	
			}
			else if( keyWord != 'c') { //not comment line, then unknown keyword
				errorMsg("Error GR File format for " << fileName);
				grFile.close();
				return -1;
			}
	
			grFile.getline(tmpFileBuf, 256); //skip to next line
	
			grFile >> keyWord;
	
		}
	
		grFile.close();
	
		if(this->computeHistogram())
			return -1;
	
		return 0;
	
	}
};    	


}
}
