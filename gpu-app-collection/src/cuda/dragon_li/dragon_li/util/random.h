#pragma once 

#include <cstdlib>
#include <ctime>

namespace dragon_li {
namespace util {

template<typename DataType, typename SizeType>
class Random {
public:
	static int random(DataType * data, 
				SizeType count,
				DataType rangeStart,
				DataType rangeEnd) {
	
		errorMsg("Not implemented");
		return -1;
	}
};
	

template<typename SizeType>
class Random<int, SizeType> {
public:
	static int random(int * data, 
				SizeType count,
				int rangeStart,
				int rangeEnd) {
	
//		std::srand(std::time(0));
		for(SizeType i = 0; i < count; i++) {
			int randomNum = std::rand() % (rangeEnd - rangeStart) + rangeStart;
			data[i] = randomNum;
	
		}
	
		return 0;
	
	}
};

template<typename SizeType>
class Random<float, SizeType> {
public:
	static int random(float * data, 
				SizeType count,
				float rangeStart,
				float rangeEnd) {
	
//		std::srand(std::time(0));
		for(SizeType i = 0; i < count; i++) {
			int randomNum = std::rand();

			float output = (float)randomNum / (float)RAND_MAX * (rangeEnd - rangeStart) + rangeStart;
			data[i] = output;
	
		}
	
		return 0;
	
	}
};


}
}
