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

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// includes, cuda
#include <cuda_runtime_api.h>
#include <cutil_math.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include <bicubicTexture_kernel.cuh>

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

cudaArray *d_imageArray = 0;

extern "C"
void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors( cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight) ); 
    uint size = imageWidth * imageHeight * sizeof(uchar);
    checkCudaErrors( cudaMemcpyToArray(d_imageArray, 0, 0, h_data, size, cudaMemcpyHostToDevice) );
    free(h_data);

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;    // access with integer texture coordinates

    getLastCudaError("initTexture");

    // Bind the array to the texture
    checkCudaErrors( cudaBindTextureToArray(tex, d_imageArray) );

    // bind same array to 2nd texture reference with point sampling
    tex2.addressMode[0] = cudaAddressModeClamp;
    tex2.addressMode[1] = cudaAddressModeClamp;
    tex2.filterMode = cudaFilterModePoint;
    tex2.normalized = false;    // access with integer texture coordinates

    checkCudaErrors( cudaBindTextureToArray(tex2, d_imageArray) );
}

extern "C"
void freeTexture()
{
    checkCudaErrors(cudaFreeArray(d_imageArray));
}


// render image using CUDA
extern "C" 
void render(int width, int height, float tx, float ty, float scale, float cx, float cy, 
            dim3 blockSize, dim3 gridSize, int mode, uchar4 *output)
{
    // call CUDA kernel, writing results to PBO memory
    switch(mode) {
    case MODE_NEAREST:
        tex.filterMode = cudaFilterModePoint;
        d_render<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    case MODE_BILINEAR:
        tex.filterMode = cudaFilterModeLinear;
        d_render<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    case MODE_BICUBIC:
        tex.filterMode = cudaFilterModePoint;
        d_renderBicubic<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    case MODE_FAST_BICUBIC:
        tex.filterMode = cudaFilterModeLinear;
        d_renderFastBicubic<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    case MODE_CATROM:
        tex.filterMode = cudaFilterModePoint;
        d_renderCatRom<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    }
    getLastCudaError("kernel failed");
}

#endif
