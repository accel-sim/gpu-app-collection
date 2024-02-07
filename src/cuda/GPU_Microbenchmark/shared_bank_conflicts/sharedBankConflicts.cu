/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
/* Sample CUDA application for shared memory bank conflicts.
 * Transposes a N x N square matrix of float elements in
 * global memory and generates an output matrix in global memory.
 * 
 * Kernel option 1 = bank conflicts
 * Kernel option 2 = no bank conflicts
 * 
 * Code from Nsight Compute 2023.1/extras/samples/sharedBankConflicts/
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>

#define DEFAULT_KERNEL_OPTION 1
#define DEFAULT_MATRIX_SIZE   512

#define RUNTIME_API_CALL(apiFuncCall)                                                          \
   do                                                                                          \
   {                                                                                           \
       cudaError_t _status = apiFuncCall;                                                      \
       if (_status != cudaSuccess)                                                             \
       {                                                                                       \
           fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__,      \
                   __LINE__, #apiFuncCall, cudaGetErrorString(_status));                       \
           exit(EXIT_FAILURE);                                                                 \
       }                                                                                       \
   } while (0)

#define PRINT_PROGRAM_USAGE()                                                                  \
   fprintf(stderr, "Usage: %s <kernel option> <matrix size>\n"                                 \
                   "    Default kernel option: %d\n"                                           \
                   "        Use 1 for '%s' and 2 for '%s'\n"                                   \
                   "    Default matrix size: %d\n"                                             \
                   "        Matrix size should be greater than or equal to tile size: %d and"  \
                   " must be an integral multiple of tile size.\n",                            \
           argv[0], DEFAULT_KERNEL_OPTION,                                                     \
           "transposeCoalesced", "transposeNoBankConflicts",                                   \
           DEFAULT_MATRIX_SIZE, TILE_DIM)

// Each block transposes a tile of (TILE_DIM x TILE_DIM) elements
// using TILE_DIM x BLOCK_ROWS threads, 
// so that each thread transposes (TILE_DIM / BLOCK_ROWS) elements.  
// TILE_DIM must be an integral multiple of BLOCK_ROWS
#define TILE_DIM   32
#define BLOCK_ROWS 8

// Coalesced global memory transpose with shared memory bank conflicts
__global__ void transposeCoalesced(float* odata, float* idata, int width, int height)
{
   __shared__ float tile[TILE_DIM][TILE_DIM];

   int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
   int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
   int indexIn = xIndex + yIndex*width;

   xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
   yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
   int indexOut = xIndex + yIndex*height;

   for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
   {
       tile[threadIdx.y + i][threadIdx.x] = idata[indexIn + i * width];
   }

   __syncthreads();

   for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
   {
       odata[indexOut + i * height] = tile[threadIdx.x][threadIdx.y + i];
   }
}

// Coalesced global memory transpose with no shared memory bank conflicts
__global__ void transposeNoBankConflicts(float* odata, float* idata, int width, int height)
{
   __shared__ float tile[TILE_DIM][TILE_DIM + 1];

   int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
   int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
   int indexIn = xIndex + yIndex*width;

   xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
   yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
   int indexOut = xIndex + yIndex*height;

   for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
   {
       tile[threadIdx.y + i][threadIdx.x] = idata[indexIn + i * width];
   }

   __syncthreads();

   for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
   {
       odata[indexOut + i * height] = tile[threadIdx.x][threadIdx.y + i];
   }
}

void computeTransposeGold(float* gold, float* idata, const int size_x, const int size_y)
{
   for (int y = 0; y < size_y; ++y)
   {
       for (int x = 0; x < size_x; ++x)
       {
           gold[(x * size_y) + y] = idata[(y * size_x) + x];
       }
   }
}

bool compareData(const float* reference, const float* data, const unsigned int len)
{
   const float epsilon = 0.01f;

   for (unsigned int i = 0; i < len; ++i)
   {
       float diff = reference[i] - data[i];
       if ((diff > epsilon) || (diff < -epsilon))
           return false;
   }

    return true;
}

int main(int argc, char* argv[])
{
   int kernelOption = DEFAULT_KERNEL_OPTION;
   if (argc > 1)
   {
       kernelOption = atoi(argv[1]);
   }

   void (*kernel)(float*, float*, int, int);
   const char* kernelName;
   if (kernelOption == 1)
   {
       kernel = &transposeCoalesced;
       kernelName = "transposeCoalesced";
   }
   else if (kernelOption == 2)
   {
       kernel = &transposeNoBankConflicts;
       kernelName = "transposeNoBankConflicts";
   }
   else
   {
       fprintf(stderr, "** Invalid kernel option: %s\n", argv[1]);
       PRINT_PROGRAM_USAGE();
       exit(EXIT_FAILURE);
   }

   int matrixSize = DEFAULT_MATRIX_SIZE;
   if (argc > 2)
   {
       matrixSize = atoi(argv[2]);
   }

   if ((matrixSize < TILE_DIM) || (matrixSize % TILE_DIM != 0))
   {
       fprintf(stderr, "** Invalid matrix size: %s\n", argv[2]);
       PRINT_PROGRAM_USAGE();
       exit(EXIT_FAILURE);
   }

   // size of memory required to store the matrix
   size_t memSize = sizeof(float) * matrixSize * matrixSize;

   // allocate host memory
   float* h_idata = (float*)malloc(memSize);
   float* h_odata = (float*)malloc(memSize);
   float* transposeGold = (float*)malloc(memSize);

   // allocate device memory
   float *d_idata, *d_odata;
   RUNTIME_API_CALL(cudaMalloc((void**)&d_idata, memSize));
   RUNTIME_API_CALL(cudaMalloc((void**)&d_odata, memSize));

   // initialize host data
   for (int i = 0; i < (matrixSize * matrixSize); ++i)
   {
       h_idata[i] = (float)i;
   }

   // copy host data to device
   RUNTIME_API_CALL(cudaMemcpy(d_idata, h_idata, memSize, cudaMemcpyHostToDevice));

   printf("\nmatrix size: %dx%d (%dx%d tiles), kernel name: '%s', "
          "tile size: %dx%d, block size: %dx%d\n",
          matrixSize, matrixSize,
          matrixSize/TILE_DIM, matrixSize/TILE_DIM,
          kernelName,
          TILE_DIM, TILE_DIM,
          TILE_DIM, BLOCK_ROWS);

   // execution configuration parameters
   dim3 grid(matrixSize / TILE_DIM, matrixSize / TILE_DIM);
   dim3 threads(TILE_DIM, BLOCK_ROWS);

   kernel<<<grid, threads>>>(d_odata, d_idata, matrixSize, matrixSize);
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to launch '%s' kernel (error code %s)!\n", 
               kernelName, cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
   RUNTIME_API_CALL(cudaMemcpy(h_odata, d_odata, memSize, cudaMemcpyDeviceToHost));

   // Compute reference transpose solution
   computeTransposeGold(transposeGold, h_idata, matrixSize, matrixSize);

   bool res = compareData(transposeGold, h_odata, matrixSize * matrixSize);
   if (res == false)
   {
       fprintf(stderr, "** '%s' kernel FAILED\n", kernelName);
       exit(EXIT_FAILURE);
   }

   printf("Done\n");
   return 0;
}
