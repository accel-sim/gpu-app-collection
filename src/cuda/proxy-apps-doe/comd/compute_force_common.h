/*
  Common functions and macros
*/

#pragma once

#include "interface.h"

#ifdef ECX_TARGET
// Einstein target: warp = 4 threads, temporal SIMT
#define WARP_SIZE	4
extern "C" __device__ void _Z_intrinsic_pseudo_syncwarp();
#else
// GPU target: warp = 32 threads, warp-synchronous SIMT
#define WARP_SIZE	32
__device__ void _Z_intrinsic_pseudo_syncwarp() {}
#endif

template<int cta_size>
__device__ __forceinline__
double __shfl_xor(double var, int laneMask, volatile double *smem = NULL)
{
#if __CUDA_ARCH__ >= 300
  int lo = __shfl_xor( __double2loint(var), laneMask );
  int hi = __shfl_xor( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
#else
  smem[threadIdx.x] = var;
  _Z_intrinsic_pseudo_syncwarp();
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
#endif
}

template<int cta_size>
__device__ __forceinline__
float __shfl_xor(float var, int laneMask, volatile float *smem = NULL) 
{
#if __CUDA_ARCH__ >= 300
  return __shfl_xor(var, laneMask);
#else
  smem[threadIdx.x] = var;
  _Z_intrinsic_pseudo_syncwarp();
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
#endif
}

// optimized version of DP rsqrt(a) provided by Norbert Juffa
__device__
double fast_rsqrt(double a)
{
  double x, e, t;
  float f;
  asm ("cvt.rn.f32.f64       %0, %1;" : "=f"(f) : "d"(a));
  asm ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(f));
  asm ("cvt.f64.f32          %0, %1;" : "=d"(x) : "f"(f));
  t = __dmul_rn (x, x);
  e = __fma_rn (a, -t, 1.0);
  t = __fma_rn (0.375, e, 0.5);
  e = __dmul_rn (e, x);
  x = __fma_rn (t, e, x);
  return x;
}

__device__
float fast_rsqrt(float a)
{
  return rsqrtf(a);
}

// optimized version of sqrt(a)
// improves EAM kernel perf by 5-10% in double precision on GK110 
// on ECX regular sqrt is still better performance-wise
template<class real>
__device__
real sqrt_opt(real a)
{
#ifdef ECX_TARGET
  return sqrt(a);
#elif 0
  return a * rsqrt(a);
#elif 1
  return a * fast_rsqrt(a);
#endif
}

#ifndef uint
typedef unsigned int uint;
#endif

// insert the first numBits of y into x starting at bit
__device__ uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}


__device__
static bool is_halo(sim_t &sim, int ibox)
{
        if ((ibox % sim.nx) >= sim.nx-2) return true;
        if ((ibox / sim.nx) % sim.ny >= sim.ny-2) return true;
        if ((ibox / sim.nx) / sim.ny >= sim.nz-2) return true;
        return false;
}

__device__
static int get_mirror_cell_id(sim_t &sim, int ibox)
{
			// get 3 coords
		int ix = (ibox % sim.nx);
		int iy = (ibox / sim.nx) % sim.ny;
        int iz = (ibox / sim.nx / sim.ny) % sim.nz;

			// mirror
		if (ix == sim.nx-2) ix = sim.nx-3;
		if (ix == sim.nx-1) ix = 0;
        if (iy == sim.ny-2) iy = sim.ny-3;
		if (iy == sim.ny-1) iy = 0;
        if (iz == sim.nz-2) iz = sim.nz-3;
		if (iz == sim.nz-1) iz = 0;
        
			// recompute index
		return ix + iy * sim.nx + iz * sim.nx * sim.ny;
}
