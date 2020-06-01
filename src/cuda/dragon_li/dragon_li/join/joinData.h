#pragma once

#include <dragon_li/util/random.h>

#undef REPORT_BASE
#define REPORT_BASE

namespace dragon_li {
namespace join {

template< typename Types >
class JoinData {

public:
	typedef typename Types::SizeType SizeType;
	typedef typename Types::DataType DataType;

	SizeType inputCountLeft;
	SizeType inputCountRight;

	std::vector<DataType> inputLeft;
	std::vector<DataType> inputRight;

	JoinData() : 
		inputCountLeft(0),
		inputCountRight(0) {}

	int generateRandomData(SizeType countLeft, SizeType countRight, DataType maxValue) {
		
		inputCountLeft = countLeft;
		inputCountRight = countRight;

		inputLeft.resize(countLeft);
		inputRight.resize(countRight);

		dragon_li::util::Random<DataType, SizeType>::random(inputLeft.data(), countLeft, 0, maxValue);
		dragon_li::util::Random<DataType, SizeType>::random(inputRight.data(), countRight, 0, maxValue);

        std::sort(inputLeft.begin(), inputLeft.end());
        std::sort(inputRight.begin(), inputRight.end());

		return 0;

	}

	int readFromDataFile() {
		return 0;
	}

};

}
}
