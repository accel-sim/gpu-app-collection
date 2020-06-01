#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>
#include <stdio.h>

using namespace std; 

static const int BLOCK_SIZE = 128; 
static const int WARP_SIZE = 32;
static const double MAX_RELATIVE_ERROR = .02;
static const int TEMP_BUFFER_SIZE = 1024;  

enum kernelType{CSR_SCALAR, CSR_VECTOR, ELLPACKR};

texture<float, 1> vecTex;  // vector textures
texture<int2, 1>  vecTexD;

// Texture Readers (used so kernels can be templated)
struct texReaderSP {
   __device__ __forceinline__ float operator()(const int idx) const
   {
       return tex1Dfetch(vecTex, idx);
   }
};

struct texReaderDP {
   __device__ __forceinline__ double operator()(const int idx) const
   {
       int2 v = tex1Dfetch(vecTexD, idx);
#if (__CUDA_ARCH__ < 130)
       // Devices before arch 130 don't support DP, and having the
       // __hiloint2double() intrinsic will cause compilation to fail.
       // This return statement added as a workaround -- it will compile,
       // but since the arch doesn't support DP, it will never be called
       return 0;
#else
       return __hiloint2double(v.y, v.x);
#endif
   }
};
void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (int i = 0; i < dim; i++)
    {
        rowDelimiters[i] = nnzAssigned;
        for (int j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j);
            int needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n);
}
template <typename floatType>
void fill(floatType *A, const int n, const float maxi)
{
    for (int j = 0; j < n; j++) 
    {
        A[j] = ((floatType) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}


// Forward declarations for kernels
template <typename fpType, typename texReader>
__global__ void 
spmv_csr_scalar_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, fpType * __restrict__ out,
const fpType * __restrict__ vec);

template <typename fpType, typename texReader>
__global__ void 
spmv_csr_vector_kernel(const fpType * __restrict__ val,
             	       const int    * __restrict__ cols,
		               const int    * __restrict__ rowDelimiters,
                       const int dim, fpType * __restrict__ out);

template <typename fpType>
__global__ void
zero(fpType * __restrict__ a, const int size);



// ****************************************************************************
// Function: spmvCpu
//
// Purpose: 
//   Runs sparse matrix vector multiplication on the CPU 
//
// Arguements: 
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of A
//   rowDelimiters: array of size dim+1 holding indices to rows of A; 
//                  last element is the index one past the last
//                  element of A
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: input - buffer of size dim
//        output - result from the spmv calculation 
// 
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing directly
//   out indirectly through a pointer
// ****************************************************************************
template <typename floatType>
void spmvCpu(const floatType *val, const int *cols, const int *rowDelimiters, 
	     const floatType *vec, int dim, floatType *out) 
{
    for (int i=0; i<dim; i++) 
    {
        floatType t = 0; 
        for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
        {
            int col = cols[j]; 
            t += val[j] * vec[col];
        }    
        out[i] = t; 
    }
}


template <typename floatType>
bool verifyResults(const floatType *cpuResults, const floatType *gpuResults,
                   const int size, const int pass = -1) 
{
    bool passed = true; 
    FILE *fcpu = fopen("cpu_results.txt", "w");
    FILE *fgpu = fopen("gpu_results.txt", "w");
    for (int i = 0; i < size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i]
            > MAX_RELATIVE_ERROR)
        {
            cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
                " dev: " << gpuResults[i] << endl;
            passed = false;
        }
        fprintf(fcpu, "%f\n", cpuResults[i]);
        fprintf(fgpu, "%f\n", gpuResults[i]);
    }
    fclose(fcpu);
    fclose(fgpu);
    if (pass != -1) 
    {
        cout << "Pass "<<pass<<": ";
    }
    if (passed) 
    {
        cout << "Passed" << endl;
    }
    else 
    {
        cout << "---FAILED---" << endl;
    }
    return passed;
}

template <typename floatType, typename texReader>
void csrTest(floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded)
{
      // Device data structures
      floatType *d_val, *d_vec, *d_out;
      int *d_cols, *d_rowDelimiters;

      // Allocate device memory
      cudaMalloc(&d_val,  numNonZeroes * sizeof(floatType));
      cudaMalloc(&d_cols, numNonZeroes * sizeof(int));
      cudaMalloc(&d_vec,  numRows * sizeof(floatType));
      cudaMalloc(&d_out,  numRows * sizeof(floatType));
      cudaMalloc(&d_rowDelimiters, (numRows+1) * sizeof(int));

      // Transfer data to device
      cudaMemcpy(d_val, h_val,   numNonZeroes * sizeof(floatType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_cols, h_cols, numNonZeroes * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_vec, h_vec, numRows * sizeof(floatType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_rowDelimiters, h_rowDelimiters,(numRows+1) * sizeof(int), cudaMemcpyHostToDevice);

      // Bind texture for position
      string suffix;
      if (sizeof(floatType) == sizeof(float))
      {
          cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
          cudaBindTexture(0, vecTex, d_vec, channelDesc, numRows * sizeof(float));
          suffix = "-SP";
      }
      else {
          cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
          cudaBindTexture(0, vecTexD, d_vec, channelDesc, numRows * sizeof(int2));
          suffix = "-DP";
      }

      // Setup thread configuration
      int nBlocksScalar = (int) ceil((floatType) numRows / BLOCK_SIZE);
      int nBlocksVector = (int) ceil(numRows /
                  (floatType)(BLOCK_SIZE / WARP_SIZE));
      int passes = 1;
      int iters  = 1;

      // Results description info
      char atts[TEMP_BUFFER_SIZE];
     // sprintf(atts, "%d_elements_%d_rows", numNonZeroes, numRows);
      string prefix = "";
      prefix += (padded) ? "Padded_" : "";
      cout << "CSR Scalar Kernel\n";
      for (int k=0; k<passes; k++)
      {
          // Run Scalar Kernel
          for (int j = 0; j < iters; j++)
          {
              spmv_csr_scalar_kernel<floatType, texReader>
              <<<nBlocksScalar, BLOCK_SIZE>>>
              (d_val, d_cols, d_rowDelimiters, numRows, d_out, d_vec);
          }
          // Transfer data back to host
          cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),cudaMemcpyDeviceToHost);
          // Compare reference solution to GPU result
          cout << "CSR Scalar Kernel Finished\n";
          if (! verifyResults(refOut, h_out, numRows, k))
          {
              return;  // If results don't match, don't report performance
          }
      }
      cout << "CSR Scalar Done\n";
      zero<floatType><<<nBlocksScalar, BLOCK_SIZE>>>(d_out, numRows);
      cudaThreadSynchronize();

      //cout << "CSR Vector Kernel\n";
      //for (int k=0; k<passes; k++)
      //{
      //    // Run Vector Kernel
      //    cudaEventRecord(start, 0);
      //    for (int j = 0; j < iters; j++)
      //    {
      //        //spmv_csr_vector_kernel<floatType, texReader> <<<nBlocksVector, BLOCK_SIZE>>> (d_val, d_cols, d_rowDelimiters, numRows, d_out);
      //    }
      //    cudaEventRecord(stop, 0);
      //    cudaEventSynchronize(stop);
      //    float vectorKernelTime;
      //    cudaEventElapsedTime(&vectorKernelTime, start, stop);
      //    cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),cudaMemcpyDeviceToHost);
      //    cudaThreadSynchronize();
      //     Compare reference solution to GPU result
      //    if (! verifyResults(refOut, h_out, numRows, k))
      //    {
      //        return;  // If results don't match, don't report performance
      //    }
      //    vectorKernelTime = (vectorKernelTime / (float)iters) * 1.e-3;
      //    string testName = prefix+"CSR-Vector"+suffix;
      //    double totalTransfer = iTransferTime + oTransferTime;
      //}
      // Free device memory
      cudaFree(d_rowDelimiters);
      cudaFree(d_vec);
      cudaFree(d_out);
      cudaFree(d_val);
      cudaFree(d_cols);
      cudaUnbindTexture(vecTexD);
      cudaUnbindTexture(vecTex);
}

template <typename floatType, typename texReader>
void RunTest(int nRows=0) 
{
    // Host data structures
    // Array of values in the sparse matrix
    floatType *h_val;
    // Array of column indices for each value in h_val
    int *h_cols;
    // Array of indices to the start of each row in h_Val
    int *h_rowDelimiters;
    // Dense vector and space for dev/cpu reference solution
    floatType *h_vec, *h_out, *refOut;
    // nItems = number of non zero elems
    int nItems, numRows;

    // This benchmark either reads in a matrix market input file or
    // generates a random matrix

        numRows = nRows;
        nItems = numRows * numRows / 100; // 1% of entries will be non-zero
        cudaMallocHost(&h_val, nItems * sizeof(floatType)); 
        cudaMallocHost(&h_cols, nItems * sizeof(int)); 
        cudaMallocHost(&h_rowDelimiters, (numRows + 1) * sizeof(int)); 
        fill(h_val, nItems, 10); 
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows);

    // Set up remaining host data
    cudaMallocHost(&h_vec, numRows * sizeof(floatType)); 
    refOut = new floatType[numRows];
    fill(h_vec, numRows, 10);

    // Set up the padded data structures
    int PAD_FACTOR = 16;
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    cudaMallocHost(&h_out, paddedSize * sizeof(floatType));

    // Compute reference solution
    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);

    // Test CSR kernels on normal data
    cout << "CSR Test\n";
    csrTest<floatType, texReader>(h_val, h_cols,
            h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false);


    delete[] refOut; 
    cudaFreeHost(h_val); 
    cudaFreeHost(h_cols); 
    cudaFreeHost(h_rowDelimiters);
    cudaFreeHost(h_vec); 
    cudaFreeHost(h_out);   
}

int main()
{
   
    int probSizes[4] = {1024, 8192, 12288, 16384};
    int sizeClass = 16384;

    cout <<"Single precision tests:\n";
    RunTest<float, texReaderSP>(sizeClass);
return 0;
    
}

template <typename fpType, typename texReader>
__global__ void 
spmv_csr_scalar_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, fpType * __restrict__ out,
 			const fpType    * __restrict__ vec)
{
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    texReader vecTexReader;

    if (myRow < dim) 
    {
        fpType t = 0.0f;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++)
        {
            int col = cols[j]; 
            t += val[j] * vecTexReader(col);
	      //t += val[j] * vec[col];
        }
        out[myRow] = t; 
    }
}
/*
template <typename fpType, typename texReader>
__global__ void 
spmv_csr_vector_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, fpType * __restrict__ out)
{
    // Thread ID in block
    int t = threadIdx.x; 
    // Thread ID within warp
    int id = t & (warpSize-1);
    int warpsPerBlock = blockDim.x / warpSize;
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / warpSize);
    // Texture reader for the dense vector
    texReader vecTexReader;

    __shared__ volatile fpType partialSums[BLOCK_SIZE];

    if (myRow < dim) 
    {
        int warpStart = rowDelimiters[myRow];
        int warpEnd = rowDelimiters[myRow+1];
        fpType mySum = 0;
        for (int j = warpStart + id; j < warpEnd; j += warpSize)
        {
            int col = cols[j];
            mySum += val[j] * vecTexReader(col);
        }
        partialSums[t] = mySum;

        // Reduce partial sums
        if (id < 16) partialSums[t] += partialSums[t+16];
        if (id <  8) partialSums[t] += partialSums[t+ 8];
        if (id <  4) partialSums[t] += partialSums[t+ 4];
        if (id <  2) partialSums[t] += partialSums[t+ 2];
        if (id <  1) partialSums[t] += partialSums[t+ 1];

        // Write result 
        if (id == 0)
        {
            out[myRow] = partialSums[t];
        }
    }
}*/

template <typename fpType>
__global__ void
zero(fpType * __restrict__ a, const int size)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < size) a[t] = 0;
}
