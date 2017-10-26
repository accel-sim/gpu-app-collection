/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

// Utility and system includes
#include <shrUtils.h>
#include <shrQATest.h>

#include <sdkHelper.h>  // helper for shared that are common to CUDA SDK samples
#include <cuda_runtime.h>

// project include
#include "histogram_common.h"

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

const int numRuns = 1;

static char *sSDKsample = "[histogram]\0";

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    StopWatchInterface *hTimer;
    int PassFailFlag = 1;
    uint byteCount = 64 * 1024; //64 * 1048576;
    uint uiSizeMult = 1;

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;


    // set logfile name and start logs
    shrQAStart(argc, argv);
    shrSetLogFileName ("histogram.txt");

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    // refer to helper_cuda.h for implementation
    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, dev) );

    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n", 
          deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = deviceProp.major * 0x10 + deviceProp.minor;

    if(version < 0x11) {
        printf("There is no device supporting a minimum of CUDA compute capability 1.1 for this SDK sample\n");
        cudaDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
    }

    sdkCreateTimer(&hTimer);

    // Optional Command-line multiplier to increase size of array to histogram
    if(checkCmdLineFlag(argc, (const char**)argv, "sizemult"))
    {
        uiSizeMult = getCmdLineArgumentInt(argc, (const char**)argv, "sizemult");
        uiSizeMult = CLAMP(uiSizeMult, 1, 10);
        byteCount *= uiSizeMult;
    }

    shrLog("Initializing data...\n");
        shrLog("...allocating CPU memory.\n");
            h_Data         = (uchar *)malloc(byteCount);
            h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
            h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

        shrLog("...generating input data\n");
            srand(2009);
            for(uint i = 0; i < byteCount; i++) 
                h_Data[i] = rand() % 256;

        shrLog("...allocating GPU memory and copying input data\n\n");
            checkCudaErrors( cudaMalloc((void **)&d_Data, byteCount  ) );
            checkCudaErrors( cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)  ) );
            checkCudaErrors( cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice) );

    {
        shrLog("Starting up 64-bin histogram...\n\n");
            initHistogram64();

        shrLog("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
            for(int iter = -1; iter < numRuns; iter++){
                //iter == -1 -- warmup iteration
                if(iter == 0){
                    checkCudaErrors( cudaDeviceSynchronize() );
                    sdkResetTimer(&hTimer);
                    sdkStartTimer(&hTimer);
                }

                histogram64(d_Histogram, d_Data, byteCount);
            }

            checkCudaErrors( cudaDeviceSynchronize() );
            sdkStopTimer(&hTimer);
            double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
        shrLog("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
        shrLogEx(LOGBOTH | MASTER, 0, "histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n", 
                (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM64_THREADBLOCK_SIZE); 

        shrLog("\nValidating GPU results...\n");
            shrLog(" ...reading back GPU results\n");
                checkCudaErrors( cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );

            shrLog(" ...histogram64CPU()\n");
               histogram64CPU(
                    h_HistogramCPU,
                    h_Data,
                    byteCount
                );

            shrLog(" ...comparing the results...\n");
                for(uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
            shrLog(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n" );

        shrLog("Shutting down 64-bin histogram...\n\n\n");
            closeHistogram64();
    }

    {
        shrLog("Initializing 256-bin histogram...\n");
            initHistogram256();

        shrLog("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
            for(int iter = -1; iter < numRuns; iter++){
                //iter == -1 -- warmup iteration
                if(iter == 0){
                    checkCudaErrors( cudaDeviceSynchronize() );
                    sdkResetTimer(&hTimer);
                    sdkStartTimer(&hTimer);
                }

                histogram256(d_Histogram, d_Data, byteCount);
            }

            checkCudaErrors( cudaDeviceSynchronize() );
            sdkStopTimer(&hTimer);
            double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
        shrLog("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
        shrLogEx(LOGBOTH | MASTER, 0, "histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n", 
                (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE); 
                
        shrLog("\nValidating GPU results...\n");
            shrLog(" ...reading back GPU results\n");
                checkCudaErrors( cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );

            shrLog(" ...histogram256CPU()\n");
                histogram256CPU(
                    h_HistogramCPU,
                    h_Data,
                    byteCount
                );

            shrLog(" ...comparing the results\n");
                for(uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
            shrLog(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n" );

        shrLog("Shutting down 256-bin histogram...\n\n\n");
            closeHistogram256();
    }

    shrLog("Shutting down...\n");
        sdkDeleteTimer(&hTimer);
        checkCudaErrors( cudaFree(d_Histogram) );
        checkCudaErrors( cudaFree(d_Data) );
        free(h_HistogramGPU);
        free(h_HistogramCPU);
        free(h_Data);

    cudaDeviceReset();
	shrLog("%s - Test Summary\n", sSDKsample);
    // pass or fail (for both 64 bit and 256 bit histograms)
    shrQAFinishExit(argc, (const char **)argv, (PassFailFlag ? QA_PASSED : QA_FAILED));
}
