/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This is a simple test showing huge access speed gap
 * between aligned and misaligned structures
 * (those having/missing __align__ keyword).
 * It measures per-element copy throughput for 
 * aligned and misaligned structures on 
 * big chunks of data.
 */


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>
#include <shrUtils.h>

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // General GPU Device CUDA Initialization
    int gpuDeviceInit(int devID)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        if (devID < 0) 
            devID = 0;
        if (devID > deviceCount-1) {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        if (deviceProp.major < 1) {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  \
        }

        checkCudaErrors( cudaSetDevice(devID) );
        printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
        return devID;
    }

    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
	    int current_device   = 0, sm_per_multiproc = 0;
	    int max_compute_perf = 0, max_perf_device  = 0;
	    int device_count     = 0, best_SM_arch     = 0;
	    cudaDeviceProp deviceProp;

	    cudaGetDeviceCount( &device_count );
	    // Find the best major SM Architecture GPU device
	    while ( current_device < device_count ) {
		    cudaGetDeviceProperties( &deviceProp, current_device );
		    if (deviceProp.major > 0 && deviceProp.major < 9999) {
			    best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		    }
		    current_device++;
	    }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count ) {
           cudaGetDeviceProperties( &deviceProp, current_device );
           if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               sm_per_multiproc = 1;
		   } else {
               sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
           }

           int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
           if( compute_perf  > max_compute_perf ) {
               // If we find GPU with SM major > 2, search only these
               if ( best_SM_arch > 2 ) {
                   // If our device==dest_SM_arch, choose this, or else pass
                   if (deviceProp.major == best_SM_arch) {	
                       max_compute_perf  = compute_perf;
                       max_perf_device   = current_device;
                   }
               } else {
                   max_compute_perf  = compute_perf;
                   max_perf_device   = current_device;
               }
           }
           ++current_device;
	    }
	    return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    int findCudaDevice(int argc, const char **argv)
    {
        cudaDeviceProp deviceProp;
        int devID = 0;
        // If the command-line has a device number specified, use it
        if (checkCmdLineFlag(argc, argv, "device")) {
            devID = getCmdLineArgumentInt(argc, argv, "device=");
            if (devID < 0) {
                printf("Invalid command line parameters\n");
                exit(-1);
            } else {
                devID = gpuDeviceInit(devID);
                if (devID < 0) {
                   printf("exiting...\n");
                   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                   exit(-1);
                }
            }
        } else {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors( cudaSetDevice( devID ) );
            checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
            printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
        }
        return devID;
    }
// end of CUDA Helper Functions

////////////////////////////////////////////////////////////////////////////////
// Misaligned types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct{
    unsigned char r, g, b, a;
} RGBA8_misaligned;

typedef struct{
    unsigned int l, a;
} LA32_misaligned;

typedef struct{
    unsigned int r, g, b;
} RGB32_misaligned;

typedef struct{
    unsigned int r, g, b, a;
} RGBA32_misaligned;



////////////////////////////////////////////////////////////////////////////////
// Aligned types
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(4){
    unsigned char r, g, b, a;
} RGBA8;

typedef unsigned int I32;

typedef struct __align__(8){
    unsigned int l, a;
} LA32;

typedef struct __align__(16){
    unsigned int r, g, b;
} RGB32;

typedef struct __align__(16){
    unsigned int r, g, b, a;
} RGBA32;


////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __align__ option is supplied.
// "Structure of arrays" storage strategy offers best performance 
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(16){
    RGBA32 c1, c2;
} RGBA32_2;



////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}



////////////////////////////////////////////////////////////////////////////////
// Simple CUDA kernel.
// Copy is carried out on per-element basis,
// so it's not per-byte in case of padded structures.
////////////////////////////////////////////////////////////////////////////////
template<class TData> __global__ void testKernel(
    TData *d_odata,
    TData *d_idata,
    int numElements
){
    const int        tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;

    for(int pos = tid; pos < numElements; pos += numThreads)
        d_odata[pos] = d_idata[pos];
}



////////////////////////////////////////////////////////////////////////////////
// Validation routine for simple copy kernel.
// We must know "packed" size of TData (number_of_fields * sizeof(simple_type))
// and compare only these "packed" parts of the structure, 
// containig actual user data. The compiler behavior with padding bytes
// is undefined, since padding is merely a placeholder 
// and doesn't contain any user data.
////////////////////////////////////////////////////////////////////////////////
template<class TData> int testCPU(
    TData *h_odata,
    TData *h_idata,
    int numElements,
    int packedElementSize
){
    for(int pos = 0; pos < numElements; pos++){
        TData src = h_idata[pos];
        TData dst = h_odata[pos];
        for(int i = 0; i < packedElementSize; i++)
            if( ((char *)&src)[i] != ((char *)&dst)[i] ) return 0;
    }

    return 1;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Memory chunk size in bytes. Reused for test
// const int       MEM_SIZE = 50000000;
// const int NUM_ITERATIONS = 32;
const int       MEM_SIZE = 150000;
const int NUM_ITERATIONS = 2;

//GPU input and output data
unsigned char *d_idata, *d_odata;
//CPU input data and instance of GPU output data
unsigned char *h_idataCPU, *h_odataGPU;
StopWatchInterface *hTimer;



template<class TData> int runTest(int packedElementSize, int memory_size){
    const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
    const int         numElements = iDivDown(memory_size, sizeof(TData));

    //Clean output buffer before current test
    checkCudaErrors( cudaMemset(d_odata, 0, memory_size) );
    //Run test
    checkCudaErrors( cudaDeviceSynchronize() );
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    for(int i = 0; i < NUM_ITERATIONS; i++){
        testKernel<TData><<<64, 256>>>(
            (TData *)d_odata,
            (TData *)d_idata,
            numElements
        );
        getLastCudaError("testKernel() execution failed\n");
    }
    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);
    double gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;
    printf(
        "Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime,
        (double)totalMemSizeAligned / (gpuTime * 0.001 * 1073741824.0)
    );

    //Read back GPU results and run validation
    checkCudaErrors( cudaMemcpy(h_odataGPU, d_odata, memory_size, cudaMemcpyDeviceToHost) );
    int flag = testCPU(
        (TData *)h_odataGPU,
        (TData *)h_idataCPU,
        numElements,
        packedElementSize
    );
    
    printf(flag ? "\tTEST OK\n" : "\tTEST FAILURE\n" );

	return !flag;
}

int main(int argc, char **argv){
    int i, nTotalFailures = 0;

    int devID;
    cudaDeviceProp deviceProp;
    shrQAStart(argc, argv);

    // find first CUDA device
    devID = findCudaDevice(argc, (const char **)argv);
		
    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("[%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n", 
			deviceProp.name, deviceProp.multiProcessorCount,
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    // Anything that is less than 192 Cores will have a scaled down workload
    float scale_factor = max((192.0f / (ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount) ), 1.0f);

    int   MemorySize = (int)(MEM_SIZE/scale_factor) & 0xffffff00; // force multiple of 256 bytes

    printf("> Compute scaling value = %4.2f\n", scale_factor);
    printf("> Memory Size = %d\n", MemorySize);

    sdkCreateTimer(&hTimer);

    printf("Allocating memory...\n");
        h_idataCPU = (unsigned char *)malloc(MemorySize);
        h_odataGPU = (unsigned char *)malloc(MemorySize);
        checkCudaErrors(cudaMalloc((void **)&d_idata, MemorySize));
        checkCudaErrors(cudaMalloc((void **)&d_odata, MemorySize));

    printf("Generating host input data array...\n");
        for(i = 0; i < MemorySize; i++)
            h_idataCPU[i] = (i & 0xFF) + 1;
    printf("Uploading input data to GPU memory...\n");
        checkCudaErrors(cudaMemcpy(d_idata, h_idataCPU, MemorySize, cudaMemcpyHostToDevice));

    printf("Testing misaligned types...\n");
        printf("uint8...\n");
        nTotalFailures += runTest<uint8>(1, MemorySize);

        printf("uint16...\n");
        nTotalFailures += runTest<uint16>(2, MemorySize);

        printf("RGBA8_misaligned...\n");
        nTotalFailures += runTest<RGBA8_misaligned>(4, MemorySize);

        printf("LA32_misaligned...\n");
        nTotalFailures += runTest<LA32_misaligned>(8, MemorySize);

        printf("RGB32_misaligned...\n");
        nTotalFailures += runTest<RGB32_misaligned>(12, MemorySize);

        printf("RGBA32_misaligned...\n");
        nTotalFailures += runTest<RGBA32_misaligned>(16, MemorySize);

    printf("Testing aligned types...\n");
        printf("RGBA8...\n");
        nTotalFailures += runTest<RGBA8>(4, MemorySize);

        printf("I32...\n");
        nTotalFailures += runTest<I32>(4, MemorySize);

        printf("LA32...\n");
        nTotalFailures += runTest<LA32>(8, MemorySize);

        printf("RGB32...\n");
        nTotalFailures += runTest<RGB32>(12, MemorySize);

        printf("RGBA32...\n");
        nTotalFailures += runTest<RGBA32>(16, MemorySize);

        printf("RGBA32_2...\n");
        nTotalFailures += runTest<RGBA32_2>(32, MemorySize);

	printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);

    printf("Shutting down...\n");
        checkCudaErrors(cudaFree(d_idata));
        checkCudaErrors(cudaFree(d_odata));
        free(h_odataGPU);
        free(h_idataCPU);

    sdkDeleteTimer(&hTimer);
    cudaDeviceReset();
	shrQAFinishExit(argc, (const char **)argv, (nTotalFailures==0 ? QA_PASSED : QA_FAILED));
}
