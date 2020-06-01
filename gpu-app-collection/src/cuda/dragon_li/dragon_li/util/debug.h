#pragma once

#include <stdio.h>
#include <hydrazine/interface/debug.h>

#define errorMsg(x) \
	std::cout << "(" << hydrazine::_debugTime() << ") " \
		<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
		<< " Error: " << x << "\n";

#define errorCuda(x) errorMsg(cudaGetErrorString(x))

#ifndef NDEBUG
	#define reportDevice(...) \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL) { \
			printf(__VA_ARGS__); \
		}
#else
	#define reportDevice(...)
#endif


#ifndef NDEBUG
	#define checkErrorDevice() { \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL) { \
			cudaError_t retVal = cudaGetLastError(); \
			if(retVal) \
				printf(cudaGetErrorString(retVal)); \
		}\
	}
#else
	#define checkErrorDevice()
#endif

namespace dragon_li {
namespace util {


bool inline testIteration(int iteration) {

    return true;

//    switch(iteration) {
//    case 2: 
//    case 3: 
//    case 4:
//    case 5:
//    case 6:
//    case 7:
//        return true;
//    default:
//        return false;
//    }
}


#ifndef NDEBUG

#ifdef ENABLE_CDP
__constant__ int *devCdpKernelCount;
#endif

int debugInit() {

#ifdef ENABLE_CDP
{
	void * devPtr;
	cudaError_t status;

	if(status = cudaMalloc(&devPtr, sizeof(int))) {
		errorCuda(status);
		return -1;
	}

	if(status = cudaMemset(devPtr, 0, sizeof(int))) {
		errorCuda(status);
		return -1;
	}

	if(status = cudaMemcpyToSymbol(devCdpKernelCount, &devPtr, sizeof(int *))) {
		errorCuda(status);
		return -1;
	}
}
#endif

	return 0;
}

#ifdef ENABLE_CDP
__device__ void cdpKernelCountInc() {

	atomicAdd(devCdpKernelCount, 1);
}

void resetCdpKernelCount() {
    void * devPtr;
    cudaMemcpyFromSymbol(&devPtr, devCdpKernelCount, sizeof(int *));
    cudaMemset(devPtr, 0, sizeof(int));
}

int printCdpKernelCount() {

	void * devPtr;
	cudaError_t status;
	if(status = cudaMemcpyFromSymbol(&devPtr, devCdpKernelCount, sizeof(int *))) {
		errorCuda(status);
		return -1;
	}

	int cdpKernelCount;
	if(status = cudaMemcpy(&cdpKernelCount, devPtr, sizeof(int), cudaMemcpyDeviceToHost)) {
		errorCuda(status);
		return -1;
	}

	std::cout << "CDP Kernel Count " << cdpKernelCount << "\n";

	return 0;
}
#endif

#endif

}
}
