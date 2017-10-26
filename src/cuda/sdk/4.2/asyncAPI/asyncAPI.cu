/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 *
 *
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are insterted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 *
*/

// includes, system
#include <stdio.h>

// includes, project
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>

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


__global__ void increment_kernel(int *g_data, int inc_value)
{ 
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   g_data[idx] = g_data[idx] + inc_value;
}

int correct_output(int *data, const int n, const int x)
{
    for(int i = 0; i < n; i++)
        if(data[i] != x)
            return 0;
    return 1;
}

int main(int argc, char *argv[])
{
    int devID;
    cudaDeviceProp deviceProps;

    shrQAStart(argc, argv);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice((const int)argc, (const char **)argv);

    // get device name 
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    int n = 16 * 64 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

    // allocate host memory
    int *a = 0;
    checkCudaErrors( cudaMallocHost((void**)&a, nbytes) );
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a=0;
    checkCudaErrors( cudaMalloc((void**)&d_a, nbytes) );
    checkCudaErrors( cudaMemset(d_a, 255, nbytes) );

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop)  );
    
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors( cudaDeviceSynchronize() );
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
        cudaEventRecord(start, 0);
        cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
        increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
        cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;
    while( cudaEventQuery(stop) == cudaErrorNotReady )
    {
        counter++;
    }
    checkCudaErrors( cudaEventElapsedTime(&gpu_time, start, stop) );

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer) );
    printf("CPU executed %d iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = (bool)correct_output(a, n, value);
	
    // release resources
    checkCudaErrors( cudaEventDestroy(start) );
    checkCudaErrors( cudaEventDestroy(stop) );
    checkCudaErrors( cudaFreeHost(a) );
    checkCudaErrors( cudaFree(d_a) );

    cudaDeviceReset();

	shrQAFinishExit(argc, (const char **)argv, (bFinalResults ? QA_PASSED : QA_FAILED));
}
