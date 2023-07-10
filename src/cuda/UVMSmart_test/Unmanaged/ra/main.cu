#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N (1<<20)

__global__ void kernel(float* input, float* output, float* table, size_t size)
{
	int x_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (x_id > size || x_id % 100 != 0)
    		return;

	float in_f = input[x_id];
	int in_i = (int)(floor(in_f));
	int table_index = (int)((in_f - float(in_i)) *( (float)(N) ));
	float* t = table + table_index;
	output[table_index] = t[0] * in_f;
}

int main(void)
{
  //int N = 1<<10;
  float *input, *output, *table;
  float *d_input, *d_table, *d_output;

  input = (float*)malloc(N*sizeof(float));
  output = (float*)malloc(N*sizeof(float));
  table = (float*)malloc(N*sizeof(float));


  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMalloc(&d_input, N*sizeof(float));
  cudaMalloc(&d_output, N*sizeof(float));
  cudaMalloc(&d_table, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    input[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //printf("-%lf-\n", input[i]);
    table[i] = ((float)(i));
  }

  cudaMemcpy(d_input, input, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_table, table, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, N*sizeof(float)); 

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_table, N);

  cudaMemcpy(output, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
    if(output[i] != 0) {
  	printf("-%d %lf-\n", i, output[i]);
    }

  return 0;
}
