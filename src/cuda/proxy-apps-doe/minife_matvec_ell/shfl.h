#include <stdio.h>

//extern "C" __device__ void _Z_intrinsic_pseudo_syncwarp();


#if defined(__CUDA_ARCH__) & (__CUDA_ARCH__ < 300)

#define MAX_BLOCK 256
template <class T>
__device__ inline T __shfl_down(T var, const unsigned int delta, const unsigned int width=32) {
  __shared__ /*volatile*/ T shfl_array[MAX_BLOCK];
  unsigned int x=threadIdx.x;
  unsigned int y=threadIdx.y;
  unsigned int xd=blockDim.x;
  

  int loc=y*xd+x;
  shfl_array[loc]=var;
  //warp sync here...
  __syncthreads();
  if(x+delta<width) {
    unsigned int srcLoc=loc+delta;
    var=shfl_array[srcLoc];
  }
  //warp sync here...
  __syncthreads();
  return var;
}

#else

__device__ inline
double __shfl_down(double var, unsigned int srcLane, int width=32) {

  int2 a=*reinterpret_cast<int2*>(&var);
  a.x=__shfl_down(a.x,srcLane,width);
  a.y=__shfl_down(a.y,srcLane,width);
  return *reinterpret_cast<double*>(&a);
}
#endif
