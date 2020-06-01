#pragma once

#include <list>

#include <dragon_li/amr/amrRegDevice.h>

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrCpu {
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType GRID_REFINE_SIZE = Settings::GRID_REFINE_SIZE;

public:

	struct AmrCpuData {
		DataType data;
		typename std::list<AmrCpuData>::iterator childPtr;

		AmrCpuData(DataType _data,
			typename std::list<AmrCpuData>::iterator _childPtr) {

			data = _data;
			childPtr = _childPtr;
		}
	};

	static std::list<AmrCpuData> cpuAmrData;

	static int amrCpu(DataType startGridValue, DataType gridRefineThreshold) {

		cpuAmrData.push_back(AmrCpuData(startGridValue, cpuAmrData.end()));

		typename std::list<AmrCpuData>::iterator curIt = cpuAmrData.begin();
		while (curIt != cpuAmrData.end()) {
			DataType gridData = curIt->data;

			if(gridData > gridRefineThreshold) {
                DataType energy = AmrRegDevice< Settings >::computeEnergy(gridData);

                //Insert first refined child, and set the child start pointer
                SizeType refineId = 0;
                DataType refineData = AmrRegDevice< Settings >::computeTemperature(energy, refineId);
                curIt->childPtr = cpuAmrData.insert(cpuAmrData.end(), AmrCpuData(refineData, cpuAmrData.end()));

                //Insert the remaining refined child
                for(refineId = 1; refineId < GRID_REFINE_SIZE; refineId++) {

                    DataType refineData = AmrRegDevice< Settings >::computeTemperature(energy, refineId);
                    cpuAmrData.push_back(AmrCpuData(refineData, cpuAmrData.end()));
                } 
            }

            curIt++;
		}

		return 0;
		

	}
};

template<typename Types>
std::list<typename AmrCpu<Types>::AmrCpuData> AmrCpu<Types>::cpuAmrData;
}
}
