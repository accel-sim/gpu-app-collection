#pragma once

#include <vector>

namespace dragon_li {
namespace join {

template< typename Types >
class JoinCpu {
	typedef typename Types::SizeType SizeType;
	typedef typename Types::DataType DataType;

public:

	static std::vector<SizeType> cpuJoinLeftIndices;
	static std::vector<SizeType> cpuJoinRightIndices;

	static int joinCpu(JoinData<Types> &joinData) {
		SizeType l = 0, r = 0;
	
		while (l < joinData.inputCountLeft && 
			r < joinData.inputCountRight) {
		
		    DataType lKey = joinData.inputLeft[l];
		    DataType rKey = joinData.inputRight[r];
		
		    if(lKey < rKey)
		        ++l;
		    else if(rKey < lKey)
		        ++r;
		    else {
		        for(SizeType i = r; i < joinData.inputCountRight; ++i) {
		            rKey = joinData.inputRight[i];
		
		            if(lKey < rKey) break;
		
		            assert(lKey == rKey);
					cpuJoinLeftIndices.push_back(l);
					cpuJoinRightIndices.push_back(i);
		        }
		
		        ++l;
		    }
		}

//		std::cout << "CPU output indices: \n";
//		for(int i = 0; i < cpuJoinLeftIndices.size(); i++)
//			std::cout << "[" << cpuJoinLeftIndices[i] 
//				<< ", " << cpuJoinRightIndices[i] << "], ";
//		std::cout << "\n\n";
		return 0;

	}

	
};

template<typename Types>
std::vector<typename Types::SizeType> JoinCpu<Types>::cpuJoinLeftIndices;
template<typename Types>
std::vector<typename Types::SizeType> JoinCpu<Types>::cpuJoinRightIndices;
}
}
