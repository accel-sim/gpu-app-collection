#include <stdio.h>
#include <iostream>
#include <math.h>
 
__global__
void stream_triad(int n, int scalar, float *x, float *y, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = scalar * x[i] + y[i];
}
 
int main(int argc, char **argv)
{
  int N = 1<<20;
  int scalar = 3;
  float *x, *y, *z;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&z, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
    z[i] = 0.0f;
  }

  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  stream_triad<<<numBlocks, blockSize>>>(N, scalar, x, y, z);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 5.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i]-5.0f));

  FILE *fp;

  fp = fopen("result_stream_triad.txt","a+");

  fprintf(fp, "StreamTriad: Max error = %f\n", maxError);

  fclose(fp);

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  return 0;
}

