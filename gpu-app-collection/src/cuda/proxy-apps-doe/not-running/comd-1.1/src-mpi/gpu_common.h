#ifdef ECX_TARGET
// Einstein target: warp = 4 threads, temporal SIMT
#define WARP_SIZE       4
extern "C" __device__ void _Z_intrinsic_pseudo_syncwarp();
#else
// GPU target: warp = 32 threads, warp-synchronous SIMT
#define WARP_SIZE       32
__device__ void _Z_intrinsic_pseudo_syncwarp() {}
#endif

#define THREAD_ATOM_CTA         128
#define WARP_ATOM_CTA		128
#define CTA_CELL_CTA		128

// NOTE: the following is tuned for GK110
#ifdef DOUBLE
#define THREAD_ATOM_ACTIVE_CTAS 	10	// 62%
#define WARP_ATOM_ACTIVE_CTAS 		12	// 75%
#define CTA_CELL_ACTIVE_CTAS 		10	// 62%
#else
// 100% occupancy for SP
#define THREAD_ATOM_ACTIVE_CTAS 	16
#define WARP_ATOM_ACTIVE_CTAS 		16
#define CTA_CELL_ACTIVE_CTAS 		16
#endif

/// Interpolate a table to determine f(r) and its derivative f'(r).
///
/// The forces on the particle are much more sensitive to the derivative
/// of the potential than on the potential itself.  It is therefore
/// absolutely essential that the interpolated derivatives are smooth
/// and continuous.  This function uses simple quadratic interpolation
/// to find f(r).  Since quadric interpolants don't have smooth
/// derivatives, f'(r) is computed using a 4 point finite difference
/// stencil.
///
/// Interpolation is used heavily by the EAM force routine so this
/// function is a potential performance hot spot.  Feel free to
/// reimplement this function (and initInterpolationObject if necessay)
/// with any higher performing implementation of interpolation, as long
/// as the alternate implmentation that has the required smoothness
/// properties.  Cubic splines are one common alternate choice.
///
/// \param [in] table Interpolation table.
/// \param [in] r Point where function value is needed.
/// \param [out] f The interpolated value of f(r).
/// \param [out] df The interpolatedi value of df(r)/dr.
__inline__ __device__
void interpolate(InterpolationObjectGpu table, real_t r, real_t &f, real_t &df)
{
   const real_t* tt = table.values; // alias

   // check boundaries
   r = max(r, table.x0);
   r = min(r, table.xn);

   // compute index
   r = (r - table.x0) * table.invDx;
   int ii = (int)floor(r);

   // reset r to fractional distance
   r = r - ii;

    // using LDG on Kepler only
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350  
    real_t v0 = __ldg(tt + ii);
    real_t v1 = __ldg(tt + ii + 1);
    real_t v2 = __ldg(tt + ii + 2);
    real_t v3 = __ldg(tt + ii + 3);
#else
    real_t v0 = tt[ii];
    real_t v1 = tt[ii + 1];
    real_t v2 = tt[ii + 2];
    real_t v3 = tt[ii + 3];
#endif

   real_t g1 = v2 - v0;
   real_t g2 = v3 - v1;

   f = v1 + 0.5 * r * (g1 + r * (v2 + v0 - 2.0 * v1));
   df = 0.5 * (g1 + r * (g2 - g1)) * table.invDx;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
// emulate shuffles through shared memory for old devices (SLOW)
__device__ __forceinline__
double __shfl_xor(double var, int laneMask, double *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
}

__device__ __forceinline__
float __shfl_xor(float var, int laneMask, float *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
}

__device__ __forceinline__
int __shfl_up(int var, unsigned int delta, int width, int *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / width;
  const int lane_id = threadIdx.x % width;
  return lane_id >= delta ? smem[width * warp_id + (lane_id - delta)] : var;
}

__device__ __forceinline__
double __shfl(double var, int laneMask, volatile double *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
float __shfl(float var, int laneMask, volatile float *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
int __shfl(int var, int laneMask, volatile int *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}
#else	// >= SM 3.0
__device__ __forceinline__
double __shfl_xor(double var, int laneMask)
{
  int lo = __shfl_xor( __double2loint(var), laneMask );
  int hi = __shfl_xor( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
}

__device__ __forceinline__
double __shfl(double var, int laneMask)
{
  int lo = __shfl( __double2loint(var), laneMask );
  int hi = __shfl( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
}
#endif

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

__device__ __forceinline__
void warp_reduce(real_t &x, real_t *smem)
{
  int lane_id = threadIdx.x % WARP_SIZE;
  smem[threadIdx.x] = x;
  // technically we also need warp sync here
  if (lane_id < 16) smem[threadIdx.x] += smem[threadIdx.x + 16];
  if (lane_id < 8) smem[threadIdx.x] += smem[threadIdx.x + 8];
  if (lane_id < 4) smem[threadIdx.x] += smem[threadIdx.x + 4];
  if (lane_id < 2) smem[threadIdx.x] += smem[threadIdx.x + 2];
  if (lane_id < 1) smem[threadIdx.x] += smem[threadIdx.x + 1];
  x = smem[threadIdx.x];
}

template<int step>
__device__ __forceinline__
void warp_reduce(real_t &ifx, real_t &ify, real_t &ifz, real_t &ie, real_t &irho)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  // warp reduction
  for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
    ifx += __shfl_xor(ifx, i);
    ify += __shfl_xor(ify, i);
    ifz += __shfl_xor(ifz, i);
    if (step == 1) {
      ie += __shfl_xor(ie, i);
      irho += __shfl_xor(irho, i);
    }
  }
#else
  // reduction using shared memory
  __shared__ real_t smem[WARP_ATOM_CTA];
  warp_reduce(ifx, smem);
  warp_reduce(ify, smem);
  warp_reduce(ifz, smem);
  if (step == 1) {
    warp_reduce(ie, smem);
    warp_reduce(irho, smem);
  }
#endif
}

// emulate atomic add for doubles
__device__ inline void atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;
  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = atomicCAS((unsigned long long*)address, oldval, newval)) != oldval)
  {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
}

