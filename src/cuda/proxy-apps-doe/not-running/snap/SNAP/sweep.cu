#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define MAX_CMOM 16
#define MAX_NG 40
#define MAX_NANG 64
#define G_OFF (2<<15)

long tot_gpu_mem = 0;
long tot_host_mem = 0;

#define CUDA_MALLOC( x, y) {\
    *(x) = NULL; \
    cudaError err = cudaMalloc(x,y); \
    if (*(x)  == NULL || err != cudaSuccess ) {  \
        printf("CUDA_MALLOC error\n"); \
        exit(1); \
    } \
    tot_gpu_mem += y; \
    printf("CUDA_MALLOC total %ld KB - %ld KB\n", tot_gpu_mem / 1024, y/ 1024); \
}

#define CUDA_MALLOC_HOST( x, y) {\
    *(x) = NULL; \
    cudaError err = cudaMallocHost(x,y); \
    if (*(x)  == NULL || err != cudaSuccess ) {  \
        printf("CUDA_MALLOC_HOST error\n"); \
        exit(1); \
    } \
    tot_host_mem += y; \
    printf("CUDA_MALLOC_HOST total %ld KB - %ld KB\n", tot_host_mem / 1024, y/ 1024); \
}



#define ECX

//#define REMOVE_UNBALANCE
//#define REMOVE_SYNC_AFTER_DIAG
//#define REMOVE_REDUCE

//#define ONLY_ONE_DIAG

#ifdef ECX
#define SMEM_SIZE                     (48 * 1024)  /* limit imposed by sm_20 nvcc compilation flow in EMC (not enforced at simulation time by ECX) */
#define SMEM_REDUCE
//WARNING: these parameters should be extracted from the knobs in mdf file 
#define NUM_SM                          4  /* knob num_sm * num_tpc */
#define NUM_LANES_PER_SM                16 /* knob num_lanes */
#define WARP_SIZE                       4  /* knob warp_size */
#define SIZE_MRF_PER_LANE               24 /* knob uarch_rf_size_kb */
#define MAX_NUM_CONC_CTX_PER_SM         8  /* knob num_concurrent_ctas */
#define MAX_NUM_CTX_PER_LANE            32 /* knob num_contexts */

#define GPU_BATCH_SIZE 1
#else

#define WARP_SIZE 32
#define KERNEL_REGISTER 63
#define SMEM_SIZE 16384
#define GPU_BATCH_SIZE 1
#define SMEM_REDUCE

#endif

#define BLOCK_SIZE MAX_NANG

// theoretically REGISTERS_QUANTIZED_PER_WARP should come from emc, for now we count them in the simulator at least once for a given code
#define REGISTERS_QUANTIZED_PER_WARP  320

int grid_size;
int gpu_batch_size;

#define CUDA_SAFE_CALL(call) {                                    \
  cudaError err = call;                                           \
  if( cudaSuccess != err) {                                       \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
          __FILE__, __LINE__, cudaGetErrorString( err) );         \
    fflush(stderr);                                               \
    exit(EXIT_FAILURE);                                           \
  }                                                               \
}

class CTimer
{
protected:
  timeval start,end;
public:
  void Start() {gettimeofday(&start, NULL);}
  double GetET() {
    gettimeofday(&end,NULL);
    double et=(end.tv_sec+end.tv_usec*0.000001)-(start.tv_sec+start.tv_usec*0.000001);
    return et;
  }
};

#ifdef ECX
const int ichunk = 4; 
const int nx = 4;     
const int ny = 12;    
const int nz = 12;    
#else
const int ichunk = 8;
const int nx = 32;
const int ny = 32;
const int nz = 32;
#endif

const int nmom = 3;
const int cmom = nmom*nmom;
const int ng = MAX_NG;
const int nang = MAX_NANG;
const int fixup = 1;
const int ndimen = 3;
const int noct = 8;
const int src_opt = 1;
const double tolr = 1e-12;

static int *d_diag_len; 
  #define diag_len(i) diag_len[i-1]
static int *d_diag_ic;
  #define diag_ic(i) diag_ic[i-1]
static int *d_diag_j;
  #define diag_j(i) diag_j[i-1]
static int *d_diag_k;
  #define diag_k(i) diag_k[i-1]
static int *d_diag_count;
  #define diag_count(i) diag_count[i-1]
static int ndiag;
static double *d_vdelt;
  #define vdelt(g) vdelt[g-1]
static double *d_w;
  #define w(a) w[a-1]
static double *d_t_xs;
  #define t_xs(i,j,k,g) t_xs[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
static double *d_hj;
  #define hj(a) hj[a-1]
static double *d_hk;
  #define hk(a) hk[a-1]
static double *d_mu;
  #define mu(a) mu[a-1]
static double *d_psii;
  #define psii(i,j,k,g) psii[(i-1)+(j-1)*nang+(k-1)*nang*ny+(g-1)*nang*ny*nz]
static double *d_psij;
  #define psij(i,j,k,g) psij[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*nz]
static double *d_psik;
  #define psik(i,j,k,g) psik[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*ny]
static double *d_ptr_in;
  #define ptr_in(a,i,j,k,o,g) \
    ptr_in[(a-1)+(i-1)*nang+(j-1)*nang*nx+(k-1)*nang*nx*ny+(o-1)*nang*nx*ny*nz+(g-1)*nang*nx*ny*nz*noct]
static double *d_ptr_out;
  #define ptr_out(a,i,j,k,o,g) \
    ptr_out[(a-1)+(i-1)*nang+(j-1)*nang*nx+(k-1)*nang*nx*ny+(o-1)*nang*nx*ny*nz+(g-1)*nang*nx*ny*nz*noct]
static double *h_jb_in, *d_jb_in;
  #define jb_in(i,j,k,g) jb_in[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*nz]
static double *h_jb_out, *d_jb_out;
  #define jb_out(i,j,k,g) jb_out[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*nz]
static double *h_kb_in, *d_kb_in;
  #define kb_in(i,j,k,g) kb_in[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*ny]
static double *h_kb_out, *d_kb_out;
  #define kb_out(i,j,k,g) kb_out[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*ny]
static double *d_qtot;
  #define qtot(c,i,j,k,g) qtot[(c-1)+(i-1)*cmom+(j-1)*cmom*nx+(k-1)*cmom*nx*ny+(g-1)*cmom*nx*ny*nz]
static double *d_flux;
  #define flux(i,j,k,g) flux[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
static double *d_fluxm;
  #define fluxm(c,i,j,k,g) \
    fluxm[(c-1)+(i-1)*(cmom-1)+(j-1)*(cmom-1)*nx+(k-1)*(cmom-1)*nx*ny+(g-1)*(cmom-1)*nx*ny*nz]
static double *d_ec;
  #define ec(a,c,o) ec[(a-1)+(c-1)*nang+(o-1)*nang*cmom]
static double *d_dinv;
  #define dinv(a,i,j,k,g) dinv[(a-1)+(i-1)*nang+(j-1)*nang*nx+(k-1)*nang*nx*ny+(g-1)*nang*nx*ny*nz]
static double *d_qim;
  #define qim(a,i,j,k,o,g) \
    qim[(a-1)+(i-1)*nang+(j-1)*nang*nx+(k-1)*nang*nx*ny+(o-1)*nang*nx*ny*nz+(g-1)*nang*nx*ny*nz*noct]
static int *d_dogrp;
  #define dogrp(i) dogrp[i-1]





#ifdef SMEM_REDUCE

__forceinline__ __device__
double breduce(double a, double *buf)
{
#ifndef REMOVE_REDUCE
  buf[threadIdx.x] = a;
  __syncthreads();
  for (int i = BLOCK_SIZE/2; i >= 1; i = i >> 1) {
    if (threadIdx.x < i)
      buf[threadIdx.x] += buf[threadIdx.x + i];
    __syncthreads();
  }
#endif 
  return buf[0];
}

#else

__forceinline__ __device__ 
double __shfl_down_double(double a, unsigned int delta)
{
  return __hiloint2double(__shfl_down(__double2hiint(a), delta), 
                          __shfl_down(__double2loint(a), delta));
}

__forceinline__ __device__
double breduce(double a, volatile double *buf, int nwarp, int niter)
{
#ifndef REMOVE_REDUCE
  int wid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;
  double b;
  #pragma unroll 5
  for (int i = WARP_SIZE/2; i >= 1; i = i >> 1) {
    b = __shfl_down_double(a, i);
    a += b;
  }
  if (laneid == 0)
    buf[wid] = a;
  __syncthreads();
  if (wid == 0) {
    if (laneid < nwarp)
      a = buf[laneid];
    else
      a = 0;
    for (int i = niter; i >= 1; i = i >> 1) {
      b = __shfl_down_double(a, i);
      a += b;
    }
  }
#endif
  return a;
}
#endif

#ifndef REMOVE_SYNC_AFTER_DIAG
//
// inter-block sw sync
__device__ volatile int *mutexin, *mutexout;
int *d_mutexin, *d_mutexout;
__forceinline__ __device__
void __barrier(int val, int g)
{
  __syncthreads();
  __threadfence();

  if (threadIdx.x == 0) {
    mutexin[(g-1)*gridDim.x + blockIdx.x] = val;
  }

  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      while (mutexin[(g-1)*gridDim.x + i] != val) { }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      mutexout[(g-1)*gridDim.x + i] = val;
    }
  }

  if (threadIdx.x == 0) {
    while(mutexout[(g-1)*gridDim.x + blockIdx.x] != val) { }
  }

  __syncthreads();
}

__forceinline__ __device__
void __barrier2(int & sense, int * pcount, int * psense) {
    __syncthreads();
    if (threadIdx.x == 0 ) {
        sense = !sense;
        if (atomicSub( pcount, 1) == 1) {
            atomicAdd( pcount, gridDim.x);
            atomicExch(psense, sense);
        }
        else
            while ( atomicAdd(psense,0) != sense);
    }
    __syncthreads();
}



#endif


__global__ void 
dim3_sweep_kernel_miniKBA_block(
  int *diag_len, int *diag_ic, int *diag_j, int *diag_k, int *diag_count, 
  int ndiag, int grpidx, int *dogrp, int ich, int id, int oct, 
  int ichunk, int ndimen, int noct, 
  int nang, int nx, int ny, int nz, int jst, int kst, 
  int cmom, int src_opt, int ng, int fixup,
  int jlo, int klo, int jhi, int khi, int jd, int kd,
  int firsty, int firstz, int lasty, int lastz, 
  const double* vdelt, const double* __restrict__ w, double *t_xs,
  double tolr, double hi, double *hj, double *hk, double *mu,
  double *psii, double *psij, double *psik, 
  double *jb_in, double *kb_in, double *jb_out, double *kb_out,
  double *ptr_in, double *ptr_out,
  const double* __restrict__ qtot, const double* __restrict__ ec, 
  double *dinv, double *qim,
  double *flux, double *fluxm)
{

#ifndef REMOVE_SYNC_AFTER_DIAG
  int goalval = gridDim.x;
#endif


#if (defined ECX || __CUDA_ARCH__ <= 200)
  __shared__ double buf[BLOCK_SIZE];
#else
  extern __shared__ double buf[];
#endif  
  int niter = (int)ceil((float)blockDim.x/WARP_SIZE/2);
  int nwarp = (int)ceil((float)blockDim.x/WARP_SIZE);

  int a = threadIdx.x + 1;

  // 1 kernel per group
  grpidx = blockIdx.y;
  int g = dogrp(grpidx + 1);
  __shared__ double s_vdelt;
  if (threadIdx.x == 0) 
    s_vdelt = vdelt(g);
  __syncthreads();
//_______________________________________________________________________
//
//   Set up the sweep order in the i-direction.
//_______________________________________________________________________

  int ist = -1;
  if (id == 2) ist = 1;

  for (int d = 1; d <= ndiag; d++) {

#ifdef REMOVE_UNBALANCE
  if ( diag_len(d) >= gridDim.x ) { 
#endif
    // 1 block per cell, iterates over the diagonal plane
    for (int n = blockIdx.x + 1; n <= diag_len(d); n += gridDim.x) {
      int ic = diag_ic(diag_count(d)+n);
      int i, j, k;
      
      if (ist < 0) 
        i = ich*ichunk - ic + 1;
      else
        i = (ich-1)*ichunk + ic;
      
      if (i > nx) return;
          
      j = diag_j(diag_count(d)+n);
      if (jst < 0) j = ny -j + 1;

      k = diag_k(diag_count(d)+n);
      if (kst < 0) k = nz - k + 1;
//_______________________________________________________________________
//
//       Left/right boundary conditions, always vacuum.
//_______________________________________________________________________
      if ( i == nx && ist == -1) {
        psii(a,j,k,g) = 0.0;
      } else if (i == 1 && ist == 1) {
        psii(a,j,k,g) = 0.0;
      }
//_______________________________________________________________________
//
//       Top/bottom boundary conditions. Vacuum at global boundaries, but
//       set to some incoming flux from neighboring proc.
//_______________________________________________________________________

      if (j == jlo) {
        if (jd == 1 && lasty) {
          psij(a,ic,k,g) = 0.0;
        } else if (jd == 2 && firsty) {
          psij(a,ic,k,g) = 0.0;
        } else {
          psij(a,ic,k,g) = jb_in(a,ic,k,g);
        }
      }

//_______________________________________________________________________
//
//       Front/back boundary conditions. Vacuum at global boundaries, but
//       set to some incoming flux from neighboring proc.
//_______________________________________________________________________
  
      if (k == klo) {
        if ((kd == 1 && lastz) || ndimen < 3) {
          psik(a,ic,j,g) = 0.0;
        } else if(kd == 2 && firstz) {
          psik(a,ic,j,g) = 0.0;
        } else {
          psik(a,ic,j,g) = kb_in(a,ic,j,g);
        }
      }      
//_______________________________________________________________________
//
//       Clear the flux arrays
//_______________________________________________________________________
      if (oct == 1) {
        if (threadIdx.x == 0)
          flux(i,j,k,g) = 0.0;
        for (int c = threadIdx.x + 1; c <= cmom-1; c += blockDim.x)
          fluxm(c,i,j,k,g) = 0.0;
      }

      double pc, psi;
      double psii_a = psii(a,j,k,g);
      double psij_a = psij(a,ic,k,g);
      double psik_a = psik(a,ic,j,g);
      double ptr_in_a = ptr_in(a,i,j,k,oct,g);        
//_______________________________________________________________________
//
//       Compute the angular source
//_______________________________________________________________________
      psi = qtot(1,i,j,k,g);
      if (src_opt == 3) {
        psi += qim(a,i,j,k,oct,g);
      } 
      for (int l = 2; l <= cmom; l++) {
        psi += ec(a,l,oct)*qtot(l,i,j,k,g);
      }
//_______________________________________________________________________
//
//       Compute the numerator for the update formula
//_______________________________________________________________________
      pc = psi + psii_a*mu(a)*hi + psij_a*hj(a) + 
        psik_a*hk(a);       
      if (s_vdelt != 0) {
        pc += s_vdelt*ptr_in_a;
      }
//_______________________________________________________________________
//
//       Compute the solution of the center. Use DD for edges. Use fixup
//       if requested.
//_______________________________________________________________________
      if (fixup == 0) {
        psi = pc*dinv(a,i,j,k,g);
        psii(a,j,k,g) = 2.0*psi - psii_a;
        psij(a,ic,k,g) = 2.0*psi - psij_a;
        if (ndimen == 3)
          psik(a,ic,j,g) = 2.0*psi - psik_a;
        if (s_vdelt != 0.0) 
          ptr_out(a,i,j,k,oct,g) = 2.0*psi - ptr_in_a;
      } else {
        double sum_hv = 4;
        double hv1 = 1, hv2 = 1, hv3 = 1, hv4 = 1;
        double fxhv1, fxhv2, fxhv3, fxhv4;
        pc = pc * dinv(a,i,j,k,g);
        //
        // fixup loop
        //
        while (1) {
          fxhv1 = 2.0*pc - psii_a;
          fxhv2 = 2.0*pc - psij_a;
          if (ndimen == 3)
            fxhv3 = 2.0*pc - psik_a;
          if (s_vdelt != 0.0) 
            fxhv4 = 2.0*pc - ptr_in_a;
          if (fxhv1 < 0) hv1 = 0;
          if (fxhv2 < 0) hv2 = 0; 
          if (fxhv3 < 0) hv3 = 0; 
          if (fxhv4 < 0) hv4 = 0; 
//_______________________________________________________________________
//
//           Exit loop when angle is fixed up
//_______________________________________________________________________     
          double sum_hv_new = hv1 + hv2 + hv3 + hv4;
          if (sum_hv != sum_hv_new+100) break;
          sum_hv = sum_hv_new;
          
//_______________________________________________________________________ 
//                                                                        
//           Recompute balance equation numerator and denominator and get 
//           new cell average flux  
//_______________________________________________________________________       
          pc = psii_a*mu(a)*hi*(1.0+hv1) +
            psij_a*hj(a)*(1.0+hv2) +
            psik_a*hk(a)*(1.0+hv3);
          if (s_vdelt != 0.0)
            pc += s_vdelt*ptr_in_a*(1.0+hv4);
          pc = psi + 0.5*pc;
          double den = t_xs(i,j,k,g) + mu(a)*hi*hv1 + 
            hj(a)*hv2 + hk(a)*hv3 + s_vdelt*hv4;

          if (den > tolr)
            pc = pc/den;
          else
            pc = 0.0;
        } 
//_______________________________________________________________________
//
//            Fixup done, compute edges
//_______________________________________________________________________
        psi = pc;
        psii(a,j,k,g) = fxhv1 * hv1;
        psij(a,ic,k,g) = fxhv2 * hv2;
        if (ndimen == 3)
          psik(a,ic,j,g) = fxhv3 * hv3;
        if (s_vdelt != 0) {
          ptr_out(a,i,j,k,oct,g) = fxhv4 * hv4;
        }
      }
//_______________________________________________________________________
//
//       Reduction to compute the flux moments
//_______________________________________________________________________

      double fluxtmp = (a <= nang) ? w(a)*psi : 0;
#ifdef SMEM_REDUCE
      double sum = breduce(fluxtmp, buf);
#else
      double sum = breduce(fluxtmp, buf, nwarp, niter);
#endif
      if (threadIdx.x == 0)
        flux(i,j,k,g) += sum;

      for (int l = 1; l <= cmom-1; l++) {
        double fluxtmp = (a <= nang) ? ec(a,l+1,oct)*w(a)*psi : 0;
#ifdef SMEM_REDUCE
        double sum = breduce(fluxtmp, buf);
#else
        double sum = breduce(fluxtmp, buf, nwarp, niter);
#endif
        if (threadIdx.x == 0)   
          fluxm(l,i,j,k,g) += sum;
      }
 
//_______________________________________________________________________
//
//       Save edge fluxes (dummy if checks for unused non-vacuum BCs)
//_______________________________________________________________________
      int ibb = 0;
      if (j == jhi) {
        if (jd == 2 && lasty) ;
        else if (jd == 1 && firsty) {
          if (ibb == 1) ;
        } else {
          jb_out(a,ic,k,g) = psij(a,ic,k,g);
        }
      }
  
      int ibf = 0;
      if (k == khi) {
        if (kd == 2 && lastz) ;
        else if (kd == 1 && firstz) {
          if (ibf == 1) ;
        } else {
          kb_out(a,ic,j,g) = psik(a,ic,j,g);
        } 
      }
#ifdef REMOVE_UNBALANCE
      n = diag_len(d);
#endif    
    } // end diagonal plane loop
#ifndef REMOVE_SYNC_AFTER_DIAG
    if (gridDim.x > 1) {
      __barrier(goalval, g);
      goalval += gridDim.x;
    } else {
      __syncthreads();
    }
#endif

#ifdef ONLY_ONE_DIAG
  d = ndiag;
#endif

#ifdef REMOVE_UNBALANCE
  }
#endif
  }
}


int main()
{
  assert(ng <= MAX_NG && nang <= MAX_NANG);
#ifndef ECX
  CUDA_SAFE_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );
  CUDA_SAFE_CALL( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
  CUDA_SAFE_CALL( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
#endif  

  ndiag = ichunk + ny + nz - 2;
  int *diag_len = (int*)malloc(ndiag*sizeof(int));
  assert(diag_len != NULL);
  memset(diag_len, 0, ndiag*sizeof(int));
  for (int k = 1; k <= nz; k++) {
    for (int j = 1; j <= ny; j++) {
      for (int i = 1; i <= ichunk; i++) {
        int nn = i + j + k - 2;
        diag_len[nn - 1] += 1;
      }
    }
  }

  int *diag_count = (int*)malloc(ndiag*sizeof(int));
  assert(diag_count != NULL);
  diag_count[0] = 0;
  int tot_size = 0;
  for (int d = 0; d < ndiag; d++) {
    tot_size += diag_len[d];
    printf("SNAP: %d %d\n", d, diag_len[d]);
    if (d > 0) diag_count[d] = diag_count[d-1] + diag_len[d-1];
  }
  int *diag_ic = (int*)malloc(tot_size*sizeof(int));
  assert(diag_ic != NULL);
  int *diag_j  = (int*)malloc(tot_size*sizeof(int));
  assert(diag_j != NULL);
  int *diag_k  = (int*)malloc(tot_size*sizeof(int));
  assert(diag_k != NULL);
  int *indx    = (int*)malloc(ndiag*sizeof(int));
  assert(indx != NULL);
  
  memset(indx, 0, ndiag*sizeof(int));
  for (int k = 1; k <= nz; k++) {
    for (int j = 1; j <= ny; j++) {
      for (int i = 1; i <= ichunk; i++) {
        int nn = i + j + k - 2;
        indx[nn-1]++;
        int ing = indx[nn-1];
        diag_ic[diag_count[nn-1]+ing-1] = i;
        diag_j [diag_count[nn-1]+ing-1] = j;
        diag_k [diag_count[nn-1]+ing-1] = k;
      }
    }
  }

  CUDA_MALLOC(&d_diag_len, ndiag*sizeof(int));
  CUDA_MALLOC(&d_diag_count, ndiag*sizeof(int));
  CUDA_MALLOC(&d_diag_ic, tot_size*sizeof(int));
  CUDA_MALLOC(&d_diag_j, tot_size*sizeof(int));
  CUDA_MALLOC(&d_diag_k, tot_size*sizeof(int));
  cudaMemcpy(d_diag_len, diag_len, ndiag*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_count, diag_count, ndiag*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_ic, diag_ic, tot_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_j, diag_j, tot_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_k, diag_k, tot_size*sizeof(int), cudaMemcpyHostToDevice);
  CUDA_SAFE_CALL( cudaGetLastError() );

//   for (int d = 0; d < ndiag; d++) {
//     printf("%d, %d, ", diag_len[d], diag_count[d]);
//     for (int i = 0; i < diag_len[d]; i++) {
//       printf("(%d,%d,%d) ", diag_ic[diag_count[d]+i], diag_j[diag_count[d]+i], diag_k[diag_count[d]+i]);
//     }
//     printf("\n");
//   }

  // compute max number of diagonal cells
  int max_diag_len = 0;
  for (int i = 0; i < ndiag; i++) {
    if (diag_len[i] > max_diag_len)
      max_diag_len = diag_len[i];
  }
  free(diag_len);
  free(diag_count);
  free(diag_ic);
  free(diag_j);
  free(diag_k);
  free(indx);
  printf("SNAP: max_diag_len = %d\n", max_diag_len);

  int *dogrp = (int*)malloc(ng*sizeof(int));
  assert(dogrp != NULL);
  for (int i = 0; i < ng; i++)
    dogrp[i] = i + 1;
  CUDA_MALLOC(&d_dogrp, ng*sizeof(int));
  cudaMemcpy(d_dogrp, dogrp, ng*sizeof(int), cudaMemcpyHostToDevice);
  free(dogrp);

  //
  // geom module
  //
  CUDA_MALLOC(&d_dinv, (nang)*(nx)*(ny)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_hj, nang*sizeof(double));
  CUDA_MALLOC(&d_hk, nang*sizeof(double));
  CUDA_SAFE_CALL( cudaGetLastError() );
  //
  // sn module
  //
  CUDA_MALLOC(&d_ec, (nang)*(cmom)*(noct)*sizeof(double));
  CUDA_MALLOC(&d_mu, nang*sizeof(double));
  CUDA_MALLOC(&d_w, nang*sizeof(double));
  CUDA_SAFE_CALL( cudaGetLastError() );
  //                                            
  // data module
  //
  CUDA_MALLOC(&d_vdelt, ng*sizeof(double));
  CUDA_SAFE_CALL( cudaGetLastError() );
  //
  // solvar module
  //
  CUDA_MALLOC(&d_psii, (nang)*(ny)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_psij, (nang)*(ichunk)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_psik, (nang)*(ichunk)*(ny)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_ptr_in, (nang)*(nx)*(ny)*(nz)*(noct)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_ptr_out, (nang)*(nx)*(ny)*(nz)*(noct)*(ng)*sizeof(double));
  cudaMemset(d_ptr_in, 0, 
             (nang)*(nx)*(ny)*(nz)*(noct)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_t_xs, (nx)*(ny)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_qtot, (cmom)*(nx)*(ny)*(nz)*(ng)*sizeof(double));
  cudaMemset(d_qtot, 0, (cmom)*(nx)*(ny)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC(&d_flux, (nx)*(ny)*(nz)*(ng)*sizeof(double));
  if (cmom > 1) {
    CUDA_MALLOC(&d_fluxm, (cmom-1)*(nx)*(ny)*(nz)*(ng)*sizeof(double));
  }
  
  CUDA_MALLOC_HOST(&h_jb_in, (nang)*(ichunk)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC_HOST(&h_kb_in, (nang)*(ichunk)*(ny)*(ng)*sizeof(double));
  CUDA_MALLOC_HOST(&h_jb_out, (nang)*(ichunk)*(nz)*(ng)*sizeof(double));
  CUDA_MALLOC_HOST(&h_kb_out, (nang)*(ichunk)*(ny)*(ng)*sizeof(double));

  cudaHostGetDevicePointer(&d_jb_in, h_jb_in, 0);
  cudaHostGetDevicePointer(&d_kb_in, h_kb_in, 0);
  cudaHostGetDevicePointer(&d_jb_out, h_jb_out, 0);
  cudaHostGetDevicePointer(&d_kb_out, h_kb_out, 0);
  CUDA_SAFE_CALL( cudaGetLastError() );

  //
  // Compute parameters for sweep kernel
  //
  gpu_batch_size = GPU_BATCH_SIZE;

#ifdef WARP_VERSION
  int max_active_block_smem_limit = SMEM_SIZE / ((4*nang)*sizeof(double));
#else
  int max_active_block_smem_limit = SMEM_SIZE / (sizeof(double)*(BLOCK_SIZE/WARP_SIZE));
#endif

  int numsm_per_kernel;
  int max_active_blocks_sm;
#ifdef ECX 
  numsm_per_kernel = NUM_SM;
  int num_reg = ( SIZE_MRF_PER_LANE * 1024 * NUM_LANES_PER_SM ) / ( 8 /* size of a DP register in bytes */ );
  int max_active_block_reg_limit = num_reg / ((BLOCK_SIZE / WARP_SIZE) * REGISTERS_QUANTIZED_PER_WARP);

  max_active_blocks_sm = min (max_active_block_smem_limit, MAX_NUM_CONC_CTX_PER_SM );
  max_active_blocks_sm = min (max_active_blocks_sm,  max_active_block_reg_limit);
  max_active_blocks_sm = min (max_active_blocks_sm, ( MAX_NUM_CTX_PER_LANE * NUM_LANES_PER_SM) / BLOCK_SIZE);
#else
  int numsm;
  cudaDeviceGetAttribute(&numsm,  cudaDevAttrMultiProcessorCount,  0);  
  numsm_per_kernel = numsm / gpu_batch_size;
  int regcnt;
  cudaDeviceGetAttribute(&regcnt, cudaDevAttrMaxRegistersPerBlock, 0);
  int max_active_block_reg_limit = regcnt / (KERNEL_REGISTER*BLOCK_SIZE);
  max_active_blocks_sm = min(max_active_block_reg_limit, max_active_block_smem_limit);
#endif  
  grid_size = numsm_per_kernel * max_active_blocks_sm;

#ifdef REMOVE_UNBALANCE
  printf("SNAP: REMOVE_UNBALANCE\n");
#endif
#ifdef REMOVE_SYNC_AFTER_DIAG
  printf("SNAP: REMOVE_SYNC_AFTER_DIAG\n");
#endif
#ifdef REMOVE_REDUCE
  printf("SNAP: REMOVE_REDUCE\n");
#endif
#ifdef ONLY_ONE_DIAG
  printf("SNAP: ONLY_ONE_DIAG\n");
#endif

#ifdef ECX
  printf("SNAP: NUM_SM                   %d\n", NUM_SM);
  printf("SNAP: NUM_LANES_PER_SM         %d\n", NUM_LANES_PER_SM);
  printf("SNAP: WARP_SIZE                %d\n", WARP_SIZE);
  printf("SNAP: SIZE_MRF_PER_LANE        %d\n", SIZE_MRF_PER_LANE);
  printf("SNAP: MAX_NUM_CONC_CTX_PER_SM  %d\n", MAX_NUM_CONC_CTX_PER_SM);
  printf("SNAP: MAX_NUM_CTX_PER_LANE     %d\n", MAX_NUM_CTX_PER_LANE);
#endif

  printf("SNAP: max_diag_len = %d\n", max_diag_len);
  printf("SNAP: numsm_per_kernel = %d, max_active_blocks_sm = %d\n",
         numsm_per_kernel, max_active_blocks_sm);
  printf("SNAP: grid_size = %d, block_size = %d, gpu_batch_size = %d\n", 
         grid_size, BLOCK_SIZE, gpu_batch_size);

#ifndef REMOVE_SYNC_AFTER_DIAG
  CUDA_MALLOC(&d_mutexin, grid_size*(nang)*sizeof(int));
  CUDA_MALLOC(&d_mutexout, grid_size*(nang)*sizeof(int));
  cudaMemcpyToSymbol(mutexin, &d_mutexin, sizeof(int*));
  cudaMemcpyToSymbol(mutexout, &d_mutexout, sizeof(int*));
#endif
  CUDA_SAFE_CALL( cudaGetLastError() );

  CTimer timer;

  int jd = 1, kd = 1;
  int nc = nx / ichunk;
  int iop = 1;
  int id = 1 + (iop - 1) / nc;
  int oct = id + 2*(jd - 1) + 4*(kd-1);
  int ich = nc - iop + 1;
  int jst = -1, kst = -1;
  int jlo = ny, klo = nz;
  int jhi = 1, khi = 1;
  int firsty = -1, firstz = -1, lasty = -1, lastz = -1;
  double hi = 0;

  timer.Start();
  dim3 grid(grid_size, 1);

#ifndef ECX
  int nsmem = sizeof(double)*BLOCK_SIZE;
  dim3_sweep_kernel_miniKBA_block<<<grid, nang, nsmem, 0>>>(
#else
    dim3_sweep_kernel_miniKBA_block<<<grid, nang>>>(
#endif
    d_diag_len, d_diag_ic, d_diag_j, d_diag_k, d_diag_count, ndiag,
    0, d_dogrp, ich, id, oct, ichunk, ndimen, noct,
    nang, nx, ny, nz, jst, kst, cmom, src_opt, ng, fixup,
    jlo, klo, jhi, khi, jd, kd,
    firsty, firstz, lasty, lastz, 
    d_vdelt, d_w, d_t_xs,
    tolr, hi, d_hj, d_hk, d_mu,
    d_psii, d_psij, d_psik,
    d_jb_in, d_kb_in, d_jb_out, d_kb_out,
    d_ptr_in, d_ptr_out,
    d_qtot, d_ec, d_dinv, d_qim,
    d_flux, d_fluxm);

  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
  double et = timer.GetET() * 1e3;
  printf("SNAP: kernel time = %f ms, throughput = %f MODF/s - MDOF = %d\n", 
         et, (double)ichunk*ny*nz*nang*gpu_batch_size*1e-3/et,
         ichunk*ny*nz*nang*gpu_batch_size);

  cudaFree(d_diag_len);
  cudaFree(d_diag_count);
  cudaFree(d_diag_ic);
  cudaFree(d_diag_j);
  cudaFree(d_diag_k);
  cudaFree(d_dogrp);
  cudaFree(d_dinv);
  cudaFree(d_hj);
  cudaFree(d_hk);
  cudaFree(d_ec);
  cudaFree(d_mu);
  cudaFree(d_w);
  cudaFree(d_vdelt);
  cudaFree(d_psii);
  cudaFree(d_psij);
  cudaFree(d_psik);
  cudaFree(d_ptr_in);
  cudaFree(d_ptr_out);
  cudaFree(d_t_xs);
  cudaFree(d_qtot);
  cudaFree(d_flux);
  cudaFree(d_fluxm);
  cudaFreeHost(h_jb_in);
  cudaFreeHost(h_kb_in);
  cudaFreeHost(h_jb_out);
  cudaFreeHost(h_kb_out);

  CUDA_SAFE_CALL( cudaGetLastError() );
  cudaDeviceReset();
}
