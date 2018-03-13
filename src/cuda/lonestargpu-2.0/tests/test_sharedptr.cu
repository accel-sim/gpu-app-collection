/* -*- mode: C++ -*- */

#include <cuda.h>
#include "sharedptr.h"
#include <assert.h>

__global__ void test_share_kernel(int *p)
{
  assert(p[0] == 1234);
}

__global__ void test_share_kernel_wr(int *p)
{
  assert(p[0] == 1234);
  p[1] = 5678;
}

__global__ void test_share_kernel_wro(int *p)
{
  p[0] = 0xABCD;
}

void test_share_rd()
{
  Shared<int> p(128);
  int *x = p.cpu_wr_ptr();

  x[0] = 1234;
  test_share_kernel<<<1, 1>>>(p.gpu_rd_ptr());
}

void test_share_wr()
{
  Shared<int> p(128);
  int *x = p.cpu_wr_ptr();

  x[0] = 1234;
  test_share_kernel<<<1, 1>>>(p.gpu_wr_ptr());
}

void test_share_wrwr() // write, read, write, read
{
  Shared<int> p(128);
  int *x = p.cpu_wr_ptr();

  x[0] = 1234;
  test_share_kernel_wr<<<1, 1>>>(p.gpu_wr_ptr());

  x = p.cpu_rd_ptr();
  assert(x[1] == 5678);
}

void test_share_rwr()
{
  Shared<int> p(128);
  int *x = p.cpu_rd_ptr();

  test_share_kernel_wro<<<1, 1>>>(p.gpu_wr_ptr());

  x = p.cpu_rd_ptr();
  assert(x[0] == 0xABCD);
}

void test_share_wwovr() // write-overwrite
{
  Shared<int> p(128);
  int *x = p.cpu_wr_ptr();

  x[0] = 1234;

  test_share_kernel_wro<<<1, 1>>>(p.gpu_wr_ptr(true));

  x = p.cpu_rd_ptr();
  assert(x[0] == 0xABCD);
}


void test_alloc_rd()
{
  Shared<int> p(128);

  int *x;

  x = p.cpu_rd_ptr();
  assert(x != NULL);

  x = p.gpu_rd_ptr();
  assert(x != NULL);
}

void test_alloc_wr()
{
  Shared<int> p(128);

  int *x;

  x = p.cpu_wr_ptr();
  assert(x != NULL);

  x = p.gpu_wr_ptr();
  assert(x != NULL);
}

int main(void)
{
  test_alloc_rd(); // 0 transfers
  test_alloc_wr(); // 1 transfer -> CPU-to-GPU

  test_share_rd(); // 1 transfer -> C2G 
  test_share_wr(); // 1 transfer -> C2G

  test_share_wrwr(); // 2 transfers -> C2G, G2C
  test_share_rwr(); // 1 transfer -> G2C
  test_share_wwovr(); // 1 transfer -> G2C

  cudaDeviceSynchronize(); // to ensure CUDA profiler writes correct log
}
