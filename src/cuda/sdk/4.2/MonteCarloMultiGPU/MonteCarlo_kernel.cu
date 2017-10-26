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


////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <curand_kernel.h>
#include "MonteCarlo_common.h"



////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA SDK sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"


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
// end of CUDA Helper Functions


////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS 512

//Preprocessed input option data
typedef struct{
    real S;
    real X;
    real MuByT;
    real VBySqrtT; 
} __TOptionData;
static __device__ __constant__ __TOptionData d_OptionData[MAX_OPTIONS];

static __device__ __TOptionValue d_CallValue[MAX_OPTIONS];

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut payoff functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
__device__ inline float endCallValue(float S, float X, float r, float MuByT, float VBySqrtT){
    float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

#define THREAD_N 256

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy. 
////////////////////////////////////////////////////////////////////////////////
static __global__ void MonteCarloOneBlockPerOption( 
    curandState* rngStates,
    int pathN )
{
    const int SUM_N = THREAD_N;
    __shared__ real s_SumCall[SUM_N];
    __shared__ real s_Sum2Call[SUM_N];

    const int optionIndex = blockIdx.x;
    const real        S = d_OptionData[optionIndex].S;
    const real        X = d_OptionData[optionIndex].X;
    const real    MuByT = d_OptionData[optionIndex].MuByT;
    const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

    // determine global thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy random number state to local memory for efficiency
    curandState localState = rngStates[tid];

    //Cycle through the entire samples array:
    //derive end stock price for each path
    //accumulate partial integrals into intermediate shared memory buffer
    for(int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x){
        __TOptionValue sumCall = {0, 0};
        for(int i = iSum; i < pathN; i += SUM_N){
            real              r = curand_normal(&localState);
            real      callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
            sumCall.Expected   += callValue;
            sumCall.Confidence += callValue * callValue;
        }
        s_SumCall[iSum]  = sumCall.Expected;
        s_Sum2Call[iSum] = sumCall.Confidence;
    }

    // store random number state back to global memory
    rngStates[tid] = localState;

    //Reduce shared memory accumulators
    //and write final result to global memory
    sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call);
    if(threadIdx.x == 0){
        __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]};
        d_CallValue[optionIndex] = t;
    }
}

static __global__ void rngSetupStates(
  curandState *rngState, 
  unsigned long long seed,
  unsigned long long offset)
{
    // determine global thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets the same seed, a different
    // sequence number. A different offset is used for
    // each device. 
    curand_init(seed, tid, offset, &rngState[tid]);
}



////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////

extern "C" void initMonteCarloGPU(TOptionPlan *plan)
{
    //Allocate internal device memory
    checkCudaErrors( cudaMallocHost( &plan->h_CallValue, sizeof(__TOptionValue)*MAX_OPTIONS) );
    //Allocate states for pseudo random number generators
    checkCudaErrors( cudaMalloc((void**) &plan->rngStates, 
                               plan->optionCount * THREAD_N * sizeof(curandState)) );
 
    // place each device pathN random numbers apart on the random number sequence
    unsigned long long offset = plan->device * plan->pathN;
    rngSetupStates<<<plan->optionCount, THREAD_N>>>(plan->rngStates, plan->seed, offset);
    getLastCudaError("rngSetupStates kernel failed.\n");
}

//Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
    for(int i = 0; i < plan->optionCount; i++){
        const double    RT = plan->optionData[i].R * plan->optionData[i].T;
        const double   sum = plan->h_CallValue[i].Expected;
        const double  sum2 = plan->h_CallValue[i].Confidence;
        const double pathN = plan->pathN;
        //Derive average from the total sum and discount by riskfree rate 
        plan->callValue[i].Expected = (float)(exp(-RT) * sum / pathN);
        //Standart deviation
        double stdDev = sqrt((pathN * sum2 - sum * sum)/ (pathN * (pathN - 1)));
        //Confidence width; in 95% of all cases theoretical value lies within these borders
        plan->callValue[i].Confidence = (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
    }
    checkCudaErrors( cudaFree(plan->rngStates) );
    checkCudaErrors( cudaFreeHost(plan->h_CallValue) );
}

//Main computations
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream){
    __TOptionData h_OptionData[MAX_OPTIONS];
    __TOptionValue *h_CallValue = plan->h_CallValue;

    if(plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS){
        printf("MonteCarloGPU(): bad option count.\n");
        return;
    }

    for(int i = 0; i < plan->optionCount; i++){
        const double           T = plan->optionData[i].T;
        const double           R = plan->optionData[i].R;
        const double           V = plan->optionData[i].V;
        const double       MuByT = (R - 0.5 * V * V) * T;
        const double    VBySqrtT = V * sqrt(T);
        h_OptionData[i].S        = (real)plan->optionData[i].S;
        h_OptionData[i].X        = (real)plan->optionData[i].X;
        h_OptionData[i].MuByT    = (real)MuByT;
        h_OptionData[i].VBySqrtT = (real)VBySqrtT;
    }

    checkCudaErrors( cudaMemcpyToSymbolAsync(
        d_OptionData,
        h_OptionData,
        plan->optionCount * sizeof(__TOptionData), 
        0, cudaMemcpyHostToDevice, stream
    ) );

    MonteCarloOneBlockPerOption<<<plan->optionCount, THREAD_N, 0, stream>>>(
            plan->rngStates,
            plan->pathN
    );
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");


    checkCudaErrors( cudaMemcpyFromSymbolAsync(
        h_CallValue,
        d_CallValue,
        plan->optionCount * sizeof(__TOptionValue), (size_t)0, cudaMemcpyDeviceToHost, stream
    ) );

    //cudaDeviceSynchronize();

}

