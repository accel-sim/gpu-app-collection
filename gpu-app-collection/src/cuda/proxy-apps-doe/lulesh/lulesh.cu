/*

                 Copyright (c) 2010.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 1.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define LULESH_SHOW_PROGRESS 1

enum { VolumeError = -1, QStopError = -2 } ;

/****************************************************/
/* Allow flexibility for arithmetic representations */
/****************************************************/

/* Could also support fixed point and interval arithmetic types */
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  /* 10 bytes on x86 */

typedef int    Index_t ; /* array subscript and loop index */
typedef real8  Real_t ;  /* floating point representation */
typedef int    Int_t ;   /* integer representation */

__host__ __device__ inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
__host__ __device__ inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
__host__            inline real10 SQRT(real10 arg) { return sqrtl(arg) ; }

__host__ __device__ inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
__host__ __device__ inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
__host__            inline real10 CBRT(real10 arg) { return cbrtl(arg) ; }

__host__ __device__ inline real4  FABS(real4  arg) { return fabsf(arg) ; }
__host__ __device__ inline real8  FABS(real8  arg) { return fabs(arg) ; }
__host__            inline real10 FABS(real10 arg) { return fabsl(arg) ; }

__host__ __device__ inline real4  FMAX(real4  arg1,real4  arg2) { return fmaxf(arg1,arg2) ; }
__host__ __device__ inline real8  FMAX(real8  arg1,real8  arg2) { return fmax(arg1,arg2) ; }
__host__            inline real10 FMAX(real10 arg1,real10 arg2) { return fmaxl(arg1,arg2) ; }

#define CUDA_SAFE_CALL( call) do {                                           \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)
    
#define CUDA(call) CUDA_SAFE_CALL(call)

#ifdef CUDA_SYNC_ALL
#define CUDA_DEBUGSYNC CUDA(cudaThreadSynchronize())
#else
#define CUDA_DEBUGSYNC
#endif

#define BLOCKSIZE 256

/* Given a number of bytes, nbytes, and a byte alignment, align, (e.g., 2,
 * 4, 8, or 16), return the smallest integer that is larger than nbytes and
 * a multiple of align.
 */
#define PAD_DIV(nbytes, align)  (((nbytes) + (align) - 1) / (align))
#define PAD(nbytes, align)  (PAD_DIV((nbytes),(align)) * (align))

   /* More general version of reduceInPlacePOT (this works for arbitrary
    * numThreadsPerBlock <= 1024). Again, conditionals on
    * numThreadsPerBlock are evaluated at compile time.
    */
template <class T, int numThreadsPerBlock>
__device__ void
reduceSum(T *sresult, const int threadID)
{
    /* If number of threads is not a power of two, first add the ones
       after the last power of two into the beginning. At most one of
       these conditionals will be true for a given NPOT block size. */
    if (numThreadsPerBlock > 512 && numThreadsPerBlock <= 1024)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-512)
            sresult[threadID] += sresult[threadID + 512];
    }
    
    if (numThreadsPerBlock > 256 && numThreadsPerBlock < 512)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-256)
            sresult[threadID] += sresult[threadID + 256];
    }
    
    if (numThreadsPerBlock > 128 && numThreadsPerBlock < 256)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-128)
            sresult[threadID] += sresult[threadID + 128];
    }
    
    if (numThreadsPerBlock > 64 && numThreadsPerBlock < 128)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-64)
            sresult[threadID] += sresult[threadID + 64];
    }
    
    if (numThreadsPerBlock > 32 && numThreadsPerBlock < 64)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-32)
            sresult[threadID] += sresult[threadID + 32];
    }
    
    if (numThreadsPerBlock > 16 && numThreadsPerBlock < 32)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-16)
            sresult[threadID] += sresult[threadID + 16];
    }
    
    if (numThreadsPerBlock > 8 && numThreadsPerBlock < 16)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-8)
            sresult[threadID] += sresult[threadID + 8];
    }
    
    if (numThreadsPerBlock > 4 && numThreadsPerBlock < 8)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-4)
            sresult[threadID] += sresult[threadID + 4];
    }
    
    if (numThreadsPerBlock > 2 && numThreadsPerBlock < 4)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-2)
            sresult[threadID] += sresult[threadID + 2];
    }
    
    if (numThreadsPerBlock >= 512) {
        __syncthreads();
        if (threadID < 256)
            sresult[threadID] += sresult[threadID + 256];
    }
    
    if (numThreadsPerBlock >= 256) {
        __syncthreads();
        if (threadID < 128)
            sresult[threadID] += sresult[threadID + 128];
    }
    if (numThreadsPerBlock >= 128) {
        __syncthreads();
        if (threadID < 64)
            sresult[threadID] += sresult[threadID + 64];
    }
    __syncthreads();
#ifdef _DEVICEEMU
    if (numThreadsPerBlock >= 64) {
        __syncthreads();
        if (threadID < 32)
            sresult[threadID] += sresult[threadID + 32];
    }
    if (numThreadsPerBlock >= 32) {
        __syncthreads();
        if (threadID < 16)
            sresult[threadID] += sresult[threadID + 16];
    }
    if (numThreadsPerBlock >= 16) {
        __syncthreads();
        if (threadID < 8)
            sresult[threadID] += sresult[threadID + 8];
    }
    if (numThreadsPerBlock >= 8) {
        __syncthreads();
        if (threadID < 4)
            sresult[threadID] += sresult[threadID + 4];
    }
    if (numThreadsPerBlock >= 4) {
        __syncthreads();
        if (threadID < 2)
            sresult[threadID] += sresult[threadID + 2];
    }
    if (numThreadsPerBlock >= 2) {
        __syncthreads();
        if (threadID < 1)
            sresult[threadID] += sresult[threadID + 1];
    }
#else
    if (threadID < 32) {
        volatile T *vol = sresult;
        if (numThreadsPerBlock >= 64) vol[threadID] += vol[threadID + 32];
        if (numThreadsPerBlock >= 32) vol[threadID] += vol[threadID + 16];
        if (numThreadsPerBlock >= 16) vol[threadID] += vol[threadID + 8];
        if (numThreadsPerBlock >= 8) vol[threadID] += vol[threadID + 4];
        if (numThreadsPerBlock >= 4) vol[threadID] += vol[threadID + 2];
        if (numThreadsPerBlock >= 2) vol[threadID] += vol[threadID + 1];
    }
#endif
    __syncthreads();
}

#define MINEQ(a,b) (a)=(((a)<(b))?(a):(b))

template <class T, int numThreadsPerBlock>
__device__ void
reduceMin(T *sresult, const int threadID)
{
    /* If number of threads is not a power of two, first add the ones
       after the last power of two into the beginning. At most one of
       these conditionals will be true for a given NPOT block size. */
    if (numThreadsPerBlock > 512 && numThreadsPerBlock <= 1024)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-512)
            MINEQ(sresult[threadID],sresult[threadID + 512]);
    }
    
    if (numThreadsPerBlock > 256 && numThreadsPerBlock < 512)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-256)
            MINEQ(sresult[threadID],sresult[threadID + 256]);
    }
    
    if (numThreadsPerBlock > 128 && numThreadsPerBlock < 256)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-128)
            MINEQ(sresult[threadID],sresult[threadID + 128]);
    }
    
    if (numThreadsPerBlock > 64 && numThreadsPerBlock < 128)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-64)
            MINEQ(sresult[threadID],sresult[threadID + 64]);
    }
    
    if (numThreadsPerBlock > 32 && numThreadsPerBlock < 64)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-32)
            MINEQ(sresult[threadID],sresult[threadID + 32]);
    }
    
    if (numThreadsPerBlock > 16 && numThreadsPerBlock < 32)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-16)
            MINEQ(sresult[threadID],sresult[threadID + 16]);
    }
    
    if (numThreadsPerBlock > 8 && numThreadsPerBlock < 16)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-8)
            MINEQ(sresult[threadID],sresult[threadID + 8]);
    }
    
    if (numThreadsPerBlock > 4 && numThreadsPerBlock < 8)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-4)
            MINEQ(sresult[threadID],sresult[threadID + 4]);
    }
    
    if (numThreadsPerBlock > 2 && numThreadsPerBlock < 4)
    {
        __syncthreads();
        if (threadID < numThreadsPerBlock-2)
            MINEQ(sresult[threadID],sresult[threadID + 2]);
    }
    
    if (numThreadsPerBlock >= 512) {
        __syncthreads();
        if (threadID < 256)
            MINEQ(sresult[threadID],sresult[threadID + 256]);
    }
    
    if (numThreadsPerBlock >= 256) {
        __syncthreads();
        if (threadID < 128)
            MINEQ(sresult[threadID],sresult[threadID + 128]);
    }
    if (numThreadsPerBlock >= 128) {
        __syncthreads();
        if (threadID < 64)
            MINEQ(sresult[threadID],sresult[threadID + 64]);
    }
    __syncthreads();
#ifdef _DEVICEEMU
    if (numThreadsPerBlock >= 64) {
        __syncthreads();
        if (threadID < 32)
            MINEQ(sresult[threadID],sresult[threadID + 32]);
    }
    if (numThreadsPerBlock >= 32) {
        __syncthreads();
        if (threadID < 16)
            MINEQ(sresult[threadID],sresult[threadID + 16]);
    }
    if (numThreadsPerBlock >= 16) {
        __syncthreads();
        if (threadID < 8)
            MINEQ(sresult[threadID],sresult[threadID + 8]);
    }
    if (numThreadsPerBlock >= 8) {
        __syncthreads();
        if (threadID < 4)
            MINEQ(sresult[threadID],sresult[threadID + 4]);
    }
    if (numThreadsPerBlock >= 4) {
        __syncthreads();
        if (threadID < 2)
            MINEQ(sresult[threadID],sresult[threadID + 2]);
    }
    if (numThreadsPerBlock >= 2) {
        __syncthreads();
        if (threadID < 1)
            MINEQ(sresult[threadID],sresult[threadID + 1]);
    }
#else
    if (threadID < 32) {
        volatile T *vol = sresult;
        if (numThreadsPerBlock >= 64) MINEQ(vol[threadID],vol[threadID + 32]);
        if (numThreadsPerBlock >= 32) MINEQ(vol[threadID],vol[threadID + 16]);
        if (numThreadsPerBlock >= 16) MINEQ(vol[threadID],vol[threadID + 8]);
        if (numThreadsPerBlock >= 8)  MINEQ(vol[threadID],vol[threadID + 4]);
        if (numThreadsPerBlock >= 4)  MINEQ(vol[threadID],vol[threadID + 2]);
        if (numThreadsPerBlock >= 2)  MINEQ(vol[threadID],vol[threadID + 1]);
    }
#endif
    __syncthreads();
}

void cuda_init()
{
    int deviceCount, dev;
    cudaDeviceProp cuda_deviceProp;
    char *s;
    
    CUDA( cudaGetDeviceCount(&deviceCount) );
    if (deviceCount == 0) {
        fprintf(stderr, "cuda_init(): no devices supporting CUDA.\n");
        exit(1);
    }
    if (s=getenv("CUDA_DEVICE")) dev=atoi(s);
    else dev=0;
    if ((dev < 0) || (dev > deviceCount-1)) {
        fprintf(stderr, "cuda_init(): requested device (%d) out of range [%d,%d]\n",
                dev, 0, deviceCount-1);
        exit(1);
    }
    CUDA( cudaGetDeviceProperties(&cuda_deviceProp, dev) );
    if (cuda_deviceProp.major < 1) {
        fprintf(stderr, "cuda_init(): device %d does not support CUDA.\n", dev);
        exit(1);
    }
    fprintf(stderr, "setting CUDA device %d\n",dev);
    CUDA( cudaSetDevice(dev) );
}

/************************************************************/
/* Allow for flexible data layout experiments by separating */
/* array interface from underlying implementation.          */
/************************************************************/

struct Mesh {

/* This first implementation allows for runnable code */
/* and is not meant to be optimal. Final implementation */
/* should separate declaration and allocation phases */
/* so that allocation can be scheduled in a cache conscious */
/* manner. */
    
    friend struct MeshGPU;
    
public:

   /**************/
   /* Allocation */
   /**************/

   void AllocateNodalPersistent(size_t size)
   {
      m_x.resize(size) ;
      m_y.resize(size) ;
      m_z.resize(size) ;

      m_xd.resize(size, Real_t(0.)) ;
      m_yd.resize(size, Real_t(0.)) ;
      m_zd.resize(size, Real_t(0.)) ;

      m_xdd.resize(size, Real_t(0.)) ;
      m_ydd.resize(size, Real_t(0.)) ;
      m_zdd.resize(size, Real_t(0.)) ;

      m_fx.resize(size) ;
      m_fy.resize(size) ;
      m_fz.resize(size) ;

      m_nodalMass.resize(size, Real_t(0.)) ;
   }

   void AllocateElemPersistent(size_t size)
   {
      m_matElemlist.resize(size) ;
      m_nodelist.resize(8*size) ;

      m_lxim.resize(size) ;
      m_lxip.resize(size) ;
      m_letam.resize(size) ;
      m_letap.resize(size) ;
      m_lzetam.resize(size) ;
      m_lzetap.resize(size) ;

      m_elemBC.resize(size) ;

      m_e.resize(size, Real_t(0.)) ;

      m_p.resize(size, Real_t(0.)) ;
      m_q.resize(size) ;
      m_ql.resize(size) ;
      m_qq.resize(size) ;

      m_v.resize(size, 1.0) ;
      m_volo.resize(size) ;
      m_delv.resize(size) ;
      m_vdov.resize(size) ;

      m_arealg.resize(size) ;
   
      m_ss.resize(size) ;

      m_elemMass.resize(size) ;
   }

   /* Temporaries should not be initialized in bulk but */
   /* this is a runnable placeholder for now */
   void AllocateElemTemporary(size_t size)
   {
      m_dxx.resize(size) ;
      m_dyy.resize(size) ;
      m_dzz.resize(size) ;

      m_delv_xi.resize(size) ;
      m_delv_eta.resize(size) ;
      m_delv_zeta.resize(size) ;

      m_delx_xi.resize(size) ;
      m_delx_eta.resize(size) ;
      m_delx_zeta.resize(size) ;

      m_vnew.resize(size) ;
   }

   void AllocateNodesets(size_t size)
   {
      m_symmX.resize(size) ;
      m_symmY.resize(size) ;
      m_symmZ.resize(size) ;
   }

   void AllocateNodeElemIndexes()
   {
     Index_t i,j,nidx;
       
       /* set up node-centered indexing of elements */
       m_nodeElemCount.resize(m_numNode);
       for (i=0;i<m_numNode;i++) m_nodeElemCount[i]=0;
       m_nodeElemCornerList.resize(m_numNode*8);
       for (i=0;i<m_numElem;i++) {
           for (j=0;j<8;j++) {
               nidx=nodelist(i,j);
               m_nodeElemCornerList[nidx+m_numNode*m_nodeElemCount[nidx]++] = i+m_numElem*j;
	       if (m_nodeElemCount[nidx]>8) {
		 fprintf(stderr, "Node degree is higher than 8!\n"); 
		 exit(1);
	       }
           }
       }
   }
    
   /**********/
   /* Access */
   /**********/

   /* Node-centered */

   Real_t& x(Index_t idx)    { return m_x[idx] ; }
   Real_t& y(Index_t idx)    { return m_y[idx] ; }
   Real_t& z(Index_t idx)    { return m_z[idx] ; }

   Real_t& xd(Index_t idx)   { return m_xd[idx] ; }
   Real_t& yd(Index_t idx)   { return m_yd[idx] ; }
   Real_t& zd(Index_t idx)   { return m_zd[idx] ; }

   Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }
   Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }
   Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }

   Real_t& fx(Index_t idx)   { return m_fx[idx] ; }
   Real_t& fy(Index_t idx)   { return m_fy[idx] ; }
   Real_t& fz(Index_t idx)   { return m_fz[idx] ; }

   Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }

   Index_t&  symmX(Index_t idx) { return m_symmX[idx] ; }
   Index_t&  symmY(Index_t idx) { return m_symmY[idx] ; }
   Index_t&  symmZ(Index_t idx) { return m_symmZ[idx] ; }
    
   /* Element-centered */

   Index_t&  matElemlist(Index_t idx) { return m_matElemlist[idx] ; }
   Index_t&  nodelist(Index_t idx,Index_t nidx)    { return m_nodelist[idx+nidx*m_numElem] ; }

   Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }
   Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }
   Index_t&  letam(Index_t idx) { return m_letam[idx] ; }
   Index_t&  letap(Index_t idx) { return m_letap[idx] ; }
   Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }
   Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }

   Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }

   Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }
   Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }
   Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }

   Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }
   Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }
   Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }

   Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }
   Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }
   Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }

   Real_t& e(Index_t idx)          { return m_e[idx] ; }

   Real_t& p(Index_t idx)          { return m_p[idx] ; }
   Real_t& q(Index_t idx)          { return m_q[idx] ; }
   Real_t& ql(Index_t idx)         { return m_ql[idx] ; }
   Real_t& qq(Index_t idx)         { return m_qq[idx] ; }

   Real_t& v(Index_t idx)          { return m_v[idx] ; }
   Real_t& volo(Index_t idx)       { return m_volo[idx] ; }
   Real_t& vnew(Index_t idx)       { return m_vnew[idx] ; }
   Real_t& delv(Index_t idx)       { return m_delv[idx] ; }
   Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }

   Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }
   
   Real_t& ss(Index_t idx)         { return m_ss[idx] ; }

   Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }

   /* Params */

   Real_t& dtfixed()              { return m_dtfixed ; }
   Real_t& time()                 { return m_time ; }
   Real_t& deltatime()            { return m_deltatime ; }
   Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
   Real_t& deltatimemultub()      { return m_deltatimemultub ; }
   Real_t& stoptime()             { return m_stoptime ; }

   Real_t& u_cut()                { return m_u_cut ; }
   Real_t& hgcoef()               { return m_hgcoef ; }
   Real_t& qstop()                { return m_qstop ; }
   Real_t& monoq_max_slope()      { return m_monoq_max_slope ; }
   Real_t& monoq_limiter_mult()   { return m_monoq_limiter_mult ; }
   Real_t& e_cut()                { return m_e_cut ; }
   Real_t& p_cut()                { return m_p_cut ; }
   Real_t& ss4o3()                { return m_ss4o3 ; }
   Real_t& q_cut()                { return m_q_cut ; }
   Real_t& v_cut()                { return m_v_cut ; }
   Real_t& qlc_monoq()            { return m_qlc_monoq ; }
   Real_t& qqc_monoq()            { return m_qqc_monoq ; }
   Real_t& qqc()                  { return m_qqc ; }
   Real_t& eosvmax()              { return m_eosvmax ; }
   Real_t& eosvmin()              { return m_eosvmin ; }
   Real_t& pmin()                 { return m_pmin ; }
   Real_t& emin()                 { return m_emin ; }
   Real_t& dvovmax()              { return m_dvovmax ; }
   Real_t& refdens()              { return m_refdens ; }

   Real_t& dtcourant()            { return m_dtcourant ; }
   Real_t& dthydro()              { return m_dthydro ; }
   Real_t& dtmax()                { return m_dtmax ; }

   Int_t&  cycle()                { return m_cycle ; }

   Index_t&  sizeX()              { return m_sizeX ; }
   Index_t&  sizeY()              { return m_sizeY ; }
   Index_t&  sizeZ()              { return m_sizeZ ; }
   Index_t&  numElem()            { return m_numElem ; }
   Index_t&  numNode()            { return m_numNode ; }
    
    
//private:

   /******************/
   /* Implementation */
   /******************/

   /* Node-centered */

   std::vector<Real_t> m_x ;  /* coordinates */
   std::vector<Real_t> m_y ;
   std::vector<Real_t> m_z ;

   std::vector<Real_t> m_xd ; /* velocities */
   std::vector<Real_t> m_yd ;
   std::vector<Real_t> m_zd ;

   std::vector<Real_t> m_xdd ; /* accelerations */
   std::vector<Real_t> m_ydd ;
   std::vector<Real_t> m_zdd ;

   std::vector<Real_t> m_fx ;  /* forces */
   std::vector<Real_t> m_fy ;
   std::vector<Real_t> m_fz ;

   std::vector<Real_t> m_nodalMass ;  /* mass */

   std::vector<Index_t> m_symmX ;  /* symmetry plane nodesets */
   std::vector<Index_t> m_symmY ;
   std::vector<Index_t> m_symmZ ;
    
   std::vector<Int_t> m_nodeElemCount ;
   std::vector<Index_t> m_nodeElemCornerList ;
    
   /* Element-centered */

   std::vector<Index_t>  m_matElemlist ;  /* material indexset */
   std::vector<Index_t>  m_nodelist ;     /* elemToNode connectivity */

   std::vector<Index_t>  m_lxim ;  /* element connectivity across each face */
   std::vector<Index_t>  m_lxip ;
   std::vector<Index_t>  m_letam ;
   std::vector<Index_t>  m_letap ;
   std::vector<Index_t>  m_lzetam ;
   std::vector<Index_t>  m_lzetap ;

   std::vector<Int_t>    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   std::vector<Real_t> m_dxx ;  /* principal strains -- temporary */
   std::vector<Real_t> m_dyy ;
   std::vector<Real_t> m_dzz ;

   std::vector<Real_t> m_delv_xi ;    /* velocity gradient -- temporary */
   std::vector<Real_t> m_delv_eta ;
   std::vector<Real_t> m_delv_zeta ;

   std::vector<Real_t> m_delx_xi ;    /* coordinate gradient -- temporary */
   std::vector<Real_t> m_delx_eta ;
   std::vector<Real_t> m_delx_zeta ;
   
   std::vector<Real_t> m_e ;   /* energy */

   std::vector<Real_t> m_p ;   /* pressure */
   std::vector<Real_t> m_q ;   /* q */
   std::vector<Real_t> m_ql ;  /* linear term for q */
   std::vector<Real_t> m_qq ;  /* quadratic term for q */

   std::vector<Real_t> m_v ;     /* relative volume */
   std::vector<Real_t> m_volo ;  /* reference volume */
   std::vector<Real_t> m_vnew ;  /* new relative volume -- temporary */
   std::vector<Real_t> m_delv ;  /* m_vnew - m_v */
   std::vector<Real_t> m_vdov ;  /* volume derivative over volume */

   std::vector<Real_t> m_arealg ;  /* characteristic length of an element */
   
   std::vector<Real_t> m_ss ;      /* "sound speed" */

   std::vector<Real_t> m_elemMass ;  /* mass */

   /* Parameters */

   Real_t  m_dtfixed ;           /* fixed time increment */
   Real_t  m_time ;              /* current time */
   Real_t  m_deltatime ;         /* variable time increment */
   Real_t  m_deltatimemultlb ;
   Real_t  m_deltatimemultub ;
   Real_t  m_stoptime ;          /* end time for simulation */

   Real_t  m_u_cut ;             /* velocity tolerance */
   Real_t  m_hgcoef ;            /* hourglass control */
   Real_t  m_qstop ;             /* excessive q indicator */
   Real_t  m_monoq_max_slope ;
   Real_t  m_monoq_limiter_mult ;
   Real_t  m_e_cut ;             /* energy tolerance */
   Real_t  m_p_cut ;             /* pressure tolerance */
   Real_t  m_ss4o3 ;
   Real_t  m_q_cut ;             /* q tolerance */
   Real_t  m_v_cut ;             /* relative volume tolerance */
   Real_t  m_qlc_monoq ;         /* linear term coef for q */
   Real_t  m_qqc_monoq ;         /* quadratic term coef for q */
   Real_t  m_qqc ;
   Real_t  m_eosvmax ;
   Real_t  m_eosvmin ;
   Real_t  m_pmin ;              /* pressure floor */
   Real_t  m_emin ;              /* energy floor */
   Real_t  m_dvovmax ;           /* maximum allowable volume change */
   Real_t  m_refdens ;           /* reference density */

   Real_t  m_dtcourant ;         /* courant constraint */
   Real_t  m_dthydro ;           /* volume change constraint */
   Real_t  m_dtmax ;             /* maximum allowable time increment */

   Int_t   m_cycle ;             /* iteration count for simulation */

   Index_t   m_sizeX ;           /* X,Y,Z extent of this block */
   Index_t   m_sizeY ;
   Index_t   m_sizeZ ;

   Index_t   m_numElem ;         /* Elements/Nodes in this domain */
   Index_t   m_numNode ;
} mesh ;

template <typename T>
T *Allocate(size_t size)
{
   return static_cast<T *>(malloc(sizeof(T)*size)) ;
}

template <typename T>
void Release(T **ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}


#define GPU_STALE 0
#define CPU_STALE 1
#define ALL_FRESH 2

template<typename T>
void freshenGPU(std::vector<T>&cpu,T **gpu,int& stale) {
    if (stale!=GPU_STALE) return;
    if (!(*gpu)) {CUDA( cudaMalloc(gpu,sizeof(T)*cpu.size()) );}
    CUDA( cudaMemcpy(*gpu,&cpu[0],sizeof(T)*cpu.size(),cudaMemcpyHostToDevice) );
    stale=ALL_FRESH;
}

template<typename T>
void freshenCPU(std::vector<T>&cpu,T *gpu,int& stale) {
    if (stale!=CPU_STALE) return;
    if (!gpu) {fprintf(stderr,"freshenCPU(): NULL GPU data!\n");exit(1);}
    CUDA( cudaMemcpy(&cpu[0],gpu,sizeof(T)*cpu.size(),cudaMemcpyDeviceToHost) );
    stale=ALL_FRESH;
}

// freshen helpers
#define FC(var) freshenCPU(mesh.m_ ## var , meshGPU.m_ ## var ,meshGPU.m_ ## var ## _stale ); // freshen CPU
#define FG(var) freshenGPU(mesh.m_ ## var , &meshGPU.m_ ## var ,meshGPU.m_ ## var ## _stale ); // freshen GPU
// stale helpers
#define SC(var) meshGPU.m_ ## var ## _stale = CPU_STALE; // stale CPU
#define SG(var) meshGPU.m_ ## var ## _stale = GPU_STALE; // stale GPU
    
struct MeshGPU {
    Mesh *m_mesh;
    
   /******************/
   /* Implementation */
   /******************/

   /* Node-centered */

   Real_t *m_x ;  /* coordinates */
   Real_t *m_y ;
   Real_t *m_z ;

   Real_t *m_xd ; /* velocities */
   Real_t *m_yd ;
   Real_t *m_zd ;

   Real_t *m_xdd ; /* accelerations */
   Real_t *m_ydd ;
   Real_t *m_zdd ;

   Real_t *m_fx ;  /* forces */
   Real_t *m_fy ;
   Real_t *m_fz ;

   Real_t *m_nodalMass ;  /* mass */

   Index_t *m_symmX ;  /* symmetry plane nodesets */
   Index_t *m_symmY ;
   Index_t *m_symmZ ;
    
   Int_t   *m_nodeElemCount ;
   Index_t *m_nodeElemCornerList ;
    
   /* Element-centered */

   Index_t * m_matElemlist ;  /* material indexset */
   Index_t * m_nodelist ;     /* elemToNode connectivity */

   Index_t * m_lxim ;  /* element connectivity across each face */
   Index_t * m_lxip ;
   Index_t * m_letam ;
   Index_t * m_letap ;
   Index_t * m_lzetam ;
   Index_t * m_lzetap ;

   Int_t *   m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   Real_t *m_dxx ;  /* principal strains -- temporary */
   Real_t *m_dyy ;
   Real_t *m_dzz ;

   Real_t *m_delv_xi ;    /* velocity gradient -- temporary */
   Real_t *m_delv_eta ;
   Real_t *m_delv_zeta ;

   Real_t *m_delx_xi ;    /* coordinate gradient -- temporary */
   Real_t *m_delx_eta ;
   Real_t *m_delx_zeta ;
   
   Real_t *m_e ;   /* energy */

   Real_t *m_p ;   /* pressure */
   Real_t *m_q ;   /* q */
   Real_t *m_ql ;  /* linear term for q */
   Real_t *m_qq ;  /* quadratic term for q */

   Real_t *m_v ;     /* relative volume */
   Real_t *m_volo ;  /* reference volume */
   Real_t *m_vnew ;  /* new relative volume -- temporary */
   Real_t *m_delv ;  /* m_vnew - m_v */
   Real_t *m_vdov ;  /* volume derivative over volume */

   Real_t *m_arealg ;  /* characteristic length of an element */
   
   Real_t *m_ss ;      /* "sound speed" */

   Real_t *m_elemMass ;  /* mass */
    
   /* Stale flags */
    int m_x_stale,m_y_stale,m_z_stale;
    int m_xd_stale,m_yd_stale,m_zd_stale;
    int m_xdd_stale,m_ydd_stale,m_zdd_stale;
    int m_fx_stale,m_fy_stale,m_fz_stale;
    int m_nodalMass_stale;
    int m_symmX_stale,m_symmY_stale,m_symmZ_stale;
    int m_nodeElemCount_stale,m_nodeElemCornerList_stale;
    int m_matElemlist_stale,m_nodelist_stale;
    int m_lxim_stale,m_lxip_stale,m_letam_stale,m_letap_stale,m_lzetam_stale,m_lzetap_stale;
    int m_elemBC_stale;
    int m_dxx_stale,m_dyy_stale,m_dzz_stale;
    int m_delv_xi_stale,m_delv_eta_stale,m_delv_zeta_stale;
    int m_delx_xi_stale,m_delx_eta_stale,m_delx_zeta_stale;
    int m_e_stale;
    int m_p_stale,m_q_stale,m_ql_stale,m_qq_stale;
    int m_v_stale,m_volo_stale,m_vnew_stale,m_delv_stale,m_vdov_stale;
    int m_arealg_stale;
    int m_ss_stale;
    int m_elemMass_stale;
    
    void init(Mesh *mesh) {
        m_mesh=mesh;
        m_x=m_y=m_z=NULL;
        m_xd=m_yd=m_zd=NULL;
        m_xdd=m_ydd=m_zdd=NULL;
        m_fx=m_fy=m_fz=NULL;
        m_nodalMass=NULL;
        m_symmX=m_symmY=m_symmZ=NULL;
        m_nodeElemCount=m_nodeElemCornerList=NULL;
        m_matElemlist=m_nodelist=NULL;
        m_lxim=m_lxip=m_letam=m_letap=m_lzetam=m_lzetap=NULL;
        m_elemBC=NULL;
        m_dxx=m_dyy=m_dzz=NULL;
        m_delv_xi=m_delv_eta=m_delv_zeta=NULL;
        m_delx_xi=m_delx_eta=m_delx_zeta=NULL;
        m_e=NULL;
        m_p=m_q=m_ql=m_qq=NULL;
        m_v=m_volo=m_vnew=m_delv=m_vdov=NULL;
        m_arealg=NULL;
        m_ss=NULL;
        m_elemMass=NULL;
        m_x_stale=m_y_stale=m_z_stale=
            m_xd_stale=m_yd_stale=m_zd_stale=
            m_xdd_stale=m_ydd_stale=m_zdd_stale=
            m_fx_stale=m_fy_stale=m_fz_stale=
            m_nodalMass_stale=
            m_symmX_stale=m_symmY_stale=m_symmZ_stale=
            m_nodeElemCount_stale=m_nodeElemCornerList_stale=
            m_matElemlist_stale=m_nodelist_stale=
            m_lxim_stale=m_lxip_stale=m_letam_stale=m_letap_stale=m_lzetam_stale=m_lzetap_stale=
            m_elemBC_stale=
            m_dxx_stale=m_dyy_stale=m_dzz_stale=
            m_delv_xi_stale=m_delv_eta_stale=m_delv_zeta_stale=
            m_delx_xi_stale=m_delx_eta_stale=m_delx_zeta_stale=
            m_e_stale=
            m_p_stale=m_q_stale=m_ql_stale=m_qq_stale=
            m_v_stale=m_volo_stale=m_vnew_stale=m_delv_stale=m_vdov_stale=
            m_arealg_stale=
            m_ss_stale=
            m_elemMass_stale=
            GPU_STALE;
    }
    void freshenGPU() {
#define F(var) ::freshenGPU(m_mesh->m_ ## var , &m_ ## var ,m_ ## var ## _stale);
        F(x); F(y); F(z);
        F(xd); F(yd); F(zd);
        F(xdd); F(ydd); F(zdd);
        F(fx); F(fy); F(fz);
        F(nodalMass);
        F(symmX); F(symmY); F(symmZ);
        F(nodeElemCount); F(nodeElemCornerList);
        F(matElemlist); F(nodelist);
        F(lxim); F(lxip); F(letam); F(letap); F(lzetam); F(lzetap);
        F(elemBC);
        F(dxx); F(dyy); F(dzz);
        F(delv_xi); F(delv_eta); F(delv_zeta);
        F(delx_xi); F(delx_eta); F(delx_zeta);
        F(e);
        F(p); F(q); F(ql); F(qq);
        F(v); F(volo); F(vnew); F(delv); F(vdov);
        F(arealg);
        F(ss);
        F(elemMass);
#undef F
    }
    void freshenCPU() {
#define F(var) ::freshenCPU(m_mesh->m_ ## var , m_ ## var ,m_ ## var ## _stale);
        F(x); F(y); F(z);
        F(xd); F(yd); F(zd);
        F(xdd); F(ydd); F(zdd);
        F(fx); F(fy); F(fz);
        F(nodalMass);
        F(symmX); F(symmY); F(symmZ);
        F(nodeElemCount); F(nodeElemCornerList);
        F(matElemlist); F(nodelist);
        F(lxim); F(lxip); F(letam); F(letap); F(lzetam); F(lzetap);
        F(elemBC);
        F(dxx); F(dyy); F(dzz);
        F(delv_xi); F(delv_eta); F(delv_zeta);
        F(delx_xi); F(delx_eta); F(delx_zeta);
        F(e);
        F(p); F(q); F(ql); F(qq);
        F(v); F(volo); F(vnew); F(delv); F(vdov);
        F(arealg);
        F(ss);
        F(elemMass);
#undef F
    }
} meshGPU;



/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x003
#define XI_M_SYMM   0x001
#define XI_M_FREE   0x002

#define XI_P        0x00c
#define XI_P_SYMM   0x004
#define XI_P_FREE   0x008

#define ETA_M       0x030
#define ETA_M_SYMM  0x010
#define ETA_M_FREE  0x020

#define ETA_P       0x0c0
#define ETA_P_SYMM  0x040
#define ETA_P_FREE  0x080

#define ZETA_M      0x300
#define ZETA_M_SYMM 0x100
#define ZETA_M_FREE 0x200

#define ZETA_P      0xc00
#define ZETA_P_SYMM 0x400
#define ZETA_P_FREE 0x800


static inline
void TimeIncrement()
{
   Real_t targetdt = mesh.stoptime() - mesh.time() ;

   if ((mesh.dtfixed() <= Real_t(0.0)) && (mesh.cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = mesh.deltatime() ;

      /* This will require a reduction in parallel */
      Real_t newdt = Real_t(1.0e+20) ;
      if (mesh.dtcourant() < newdt) {
         newdt = mesh.dtcourant() / Real_t(2.0) ;
      }
      if (mesh.dthydro() < newdt) {
         newdt = mesh.dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < mesh.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > mesh.deltatimemultub()) {
            newdt = olddt*mesh.deltatimemultub() ;
         }
      }

      if (newdt > mesh.dtmax()) {
         newdt = mesh.dtmax() ;
      }
      mesh.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > mesh.deltatime()) &&
       (targetdt < (Real_t(4.0) * mesh.deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * mesh.deltatime() / Real_t(3.0) ;
   }

   if (targetdt < mesh.deltatime()) {
      mesh.deltatime() = targetdt ;
   }

   mesh.time() += mesh.deltatime() ;

   ++mesh.cycle() ;
}

__global__
void InitStressTermsForElems_kernel(
    int numElem,Real_t *sigxx, Real_t *sigyy, Real_t *sigzz, Real_t *p, Real_t *q)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numElem)
        sigxx[i] = sigyy[i] = sigzz[i] =  - p[i] - q[i] ;
}

static inline
void InitStressTermsForElems_gpu(Index_t numElem, 
                                 Real_t *sigxx, Real_t *sigyy, Real_t *sigzz)
{
    dim3 dimBlock(BLOCKSIZE,1,1);
    dim3 dimGrid(PAD_DIV(numElem,dimBlock.x),1,1);
    //cudaFuncSetCacheConfig(InitStressTermsForElems_kernel,cudaFuncCachePreferL1); // set as default for all kernels after this one
    InitStressTermsForElems_kernel<<<dimGrid, dimBlock>>>
        (numElem,sigxx,sigyy,sigzz,meshGPU.m_p,meshGPU.m_q);
    CUDA_DEBUGSYNC;
}

static inline
void InitStressTermsForElems_cpu(Index_t numElem, 
                                 Real_t *sigxx, Real_t *sigyy, Real_t *sigzz)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //
   for (Index_t i = 0 ; i < numElem ; ++i){
      sigxx[i] =  sigyy[i] = sigzz[i] =  - mesh.p(i) - mesh.q(i) ;
   }
}

static inline
void InitStressTermsForElems(Index_t numElem, 
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                             int useCPU)
{
    if (useCPU) {
        FC(p); FC(q);
        InitStressTermsForElems_cpu(numElem,sigxx,sigyy,sigzz);
    }
    else {
        FG(p); FG(q);    
        InitStressTermsForElems_gpu(numElem,sigxx,sigyy,sigzz);
    }
}

__host__ __device__
static inline
void CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

__host__ __device__
static inline
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

__host__ __device__
static inline
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

__host__ __device__
static inline
void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t* const fx,
                                  Real_t* const fy,
                                  Real_t* const fz,
				  int stride)
{
  Real_t pfx0 = B[0][0] ;   Real_t pfx1 = B[0][1] ;
  Real_t pfx2 = B[0][2] ;   Real_t pfx3 = B[0][3] ;
  Real_t pfx4 = B[0][4] ;   Real_t pfx5 = B[0][5] ;
  Real_t pfx6 = B[0][6] ;   Real_t pfx7 = B[0][7] ;

  Real_t pfy0 = B[1][0] ;   Real_t pfy1 = B[1][1] ;
  Real_t pfy2 = B[1][2] ;   Real_t pfy3 = B[1][3] ;
  Real_t pfy4 = B[1][4] ;   Real_t pfy5 = B[1][5] ;
  Real_t pfy6 = B[1][6] ;   Real_t pfy7 = B[1][7] ;

  Real_t pfz0 = B[2][0] ;   Real_t pfz1 = B[2][1] ;
  Real_t pfz2 = B[2][2] ;   Real_t pfz3 = B[2][3] ;
  Real_t pfz4 = B[2][4] ;   Real_t pfz5 = B[2][5] ;
  Real_t pfz6 = B[2][6] ;   Real_t pfz7 = B[2][7] ;

  fx[0*stride] = -( stress_xx * pfx0 );
  fx[1*stride] = -( stress_xx * pfx1 );
  fx[2*stride] = -( stress_xx * pfx2 );
  fx[3*stride] = -( stress_xx * pfx3 );
  fx[4*stride] = -( stress_xx * pfx4 );
  fx[5*stride] = -( stress_xx * pfx5 );
  fx[6*stride] = -( stress_xx * pfx6 );
  fx[7*stride] = -( stress_xx * pfx7 );

  fy[0*stride] = -( stress_yy * pfy0  );
  fy[1*stride] = -( stress_yy * pfy1  );
  fy[2*stride] = -( stress_yy * pfy2  );
  fy[3*stride] = -( stress_yy * pfy3  );
  fy[4*stride] = -( stress_yy * pfy4  );
  fy[5*stride] = -( stress_yy * pfy5  );
  fy[6*stride] = -( stress_yy * pfy6  );
  fy[7*stride] = -( stress_yy * pfy7  );

  fz[0*stride] = -( stress_zz * pfz0 );
  fz[1*stride] = -( stress_zz * pfz1 );
  fz[2*stride] = -( stress_zz * pfz2 );
  fz[3*stride] = -( stress_zz * pfz3 );
  fz[4*stride] = -( stress_zz * pfz4 );
  fz[5*stride] = -( stress_zz * pfz5 );
  fz[6*stride] = -( stress_zz * pfz6 );
  fz[7*stride] = -( stress_zz * pfz7 );
}

__global__
void IntegrateStressForElems_kernel( Index_t numElem, Index_t *nodelist,
                                     Real_t *x, Real_t *y, Real_t *z,
                                     Real_t *fx_elem, Real_t *fy_elem, Real_t *fz_elem,
                                     Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                                     Real_t *determ)
{
  Real_t B[3][8] ;// shape function derivatives
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;

  int k=blockDim.x*blockIdx.x + threadIdx.x;
  if (k<numElem) {
    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*numElem];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode]; 
   }

    /* Volume calculation involves extra work for numerical consistency. */
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                         x_local, y_local, z_local );

    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                 &fx_elem[k], &fy_elem[k], &fz_elem[k], numElem ) ;
  }
}

__global__
void AddNodeForcesFromElems_kernel( Index_t numNode,
                                    Int_t *nodeElemCount, Index_t *nodeElemCornerList,
                                    Real_t *fx_elem, Real_t *fy_elem, Real_t *fz_elem,
                                    Real_t *fx_node, Real_t *fy_node, Real_t *fz_node)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numNode) {
        Int_t count=nodeElemCount[i];
        Real_t fx,fy,fz;
        fx=fy=fz=Real_t(0.0);
        for (int j=0;j<count;j++) {
            Index_t elem=nodeElemCornerList[i+numNode*j];
            fx+=fx_elem[elem]; fy+=fy_elem[elem]; fz+=fz_elem[elem];
        }
        fx_node[i]=fx; fy_node[i]=fy; fz_node[i]=fz;
    }
}

__global__
void AddNodeForcesFromElems2_kernel( Index_t numNode,
                                    Int_t *nodeElemCount, Index_t *nodeElemCornerList,
                                    Real_t *fx_elem, Real_t *fy_elem, Real_t *fz_elem,
                                    Real_t *fx_node, Real_t *fy_node, Real_t *fz_node)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numNode) {
        Int_t count=nodeElemCount[i];
        Real_t fx,fy,fz;
        fx=fy=fz=Real_t(0.0);
        for (int j=0;j<count;j++) {
            Index_t elem=nodeElemCornerList[i+numNode*j];
            fx+=fx_elem[elem]; fy+=fy_elem[elem]; fz+=fz_elem[elem];
        }
        fx_node[i]+=fx; fy_node[i]+=fy; fz_node[i]+=fz;
    }
}

static inline
void IntegrateStressForElems_gpu( Index_t numElem,
                                  Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                                  Real_t *determ, int& badvol)
{
    Real_t *fx_elem,*fy_elem,*fz_elem;

    CUDA( cudaMalloc(&fx_elem,numElem*8*sizeof(Real_t)) );
    CUDA( cudaMalloc(&fy_elem,numElem*8*sizeof(Real_t)) );
    CUDA( cudaMalloc(&fz_elem,numElem*8*sizeof(Real_t)) );
    
    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(numElem,dimBlock.x),1,1);
    IntegrateStressForElems_kernel<<<dimGrid,dimBlock>>>
        (numElem, meshGPU.m_nodelist, meshGPU.m_x, meshGPU.m_y, meshGPU.m_z,
         fx_elem, fy_elem, fz_elem, sigxx, sigyy, sigzz, determ);
    CUDA_DEBUGSYNC;

    dimGrid=dim3(PAD_DIV(mesh.numNode(),dimBlock.x),1,1);
    AddNodeForcesFromElems_kernel<<<dimGrid,dimBlock>>>
        (mesh.numNode(),meshGPU.m_nodeElemCount,meshGPU.m_nodeElemCornerList,
         fx_elem,fy_elem,fz_elem,meshGPU.m_fx,meshGPU.m_fy,meshGPU.m_fz);
    CUDA_DEBUGSYNC;

    CUDA( cudaFree(fx_elem) );
    CUDA( cudaFree(fy_elem) );
    CUDA( cudaFree(fz_elem) );
    
    // JDC -- need a reduction step to check for non-positive element volumes
    badvol=0; 
}

static inline
void IntegrateStressForElems_cpu( Index_t numElem,
                                  Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                                  Real_t *determ, int& badvol)
{
  Real_t B[3][8] ;// shape function derivatives
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ; 
  Real_t fx_local[8] ;
  Real_t fy_local[8] ;
  Real_t fz_local[8] ;

  // loop over all elements
  for( Index_t k=0 ; k<numElem ; ++k )
  {
    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = mesh.nodelist(k,lnode);
      x_local[lnode] = mesh.x(gnode);
      y_local[lnode] = mesh.y(gnode);
      z_local[lnode] = mesh.z(gnode);
    }

    /* Volume calculation involves extra work for numerical consistency. */
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                          x_local, y_local, z_local );

    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
				 fx_local, fy_local, fz_local, 1 ) ;

    // copy nodal force contributions to global force arrray.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = mesh.nodelist(k,lnode);
      mesh.fx(gnode) += fx_local[lnode];
      mesh.fy(gnode) += fy_local[lnode];
      mesh.fz(gnode) += fz_local[lnode];
    }
  }

  badvol=0;
  for ( Index_t k=0 ; k<numElem ; ++k ) {
      if (determ[k] <= Real_t(0.0)) {
          badvol=1;
      }
  }
}

static inline
void IntegrateStressForElems( Index_t numElem,
                              Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                              Real_t *determ, int& badvol, int useCPU)
{
    if (useCPU) {
        FC(nodelist); FC(x); FC(y); FC(z);
        IntegrateStressForElems_cpu(numElem,sigxx,sigyy,sigzz,determ,badvol);
        SG(fx); SG(fy); SG(fz);
    }
    else {
        FG(nodelist); FG(nodeElemCount); FG(nodeElemCornerList);
        FG(x); FG(y); FG(z);
        IntegrateStressForElems_gpu(numElem,sigxx,sigyy,sigzz,determ,badvol);
        SC(fx); SC(fy); SC(fz);
    }
    
}


static inline
void CollectDomainNodesToElemNodes(const Index_t elemNum,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8])
{
   Index_t nd0i = mesh.nodelist(elemNum,0) ;
   Index_t nd1i = mesh.nodelist(elemNum,1) ;
   Index_t nd2i = mesh.nodelist(elemNum,2) ;
   Index_t nd3i = mesh.nodelist(elemNum,3) ;
   Index_t nd4i = mesh.nodelist(elemNum,4) ;
   Index_t nd5i = mesh.nodelist(elemNum,5) ;
   Index_t nd6i = mesh.nodelist(elemNum,6) ;
   Index_t nd7i = mesh.nodelist(elemNum,7) ;

   elemX[0] = mesh.x(nd0i);
   elemX[1] = mesh.x(nd1i);
   elemX[2] = mesh.x(nd2i);
   elemX[3] = mesh.x(nd3i);
   elemX[4] = mesh.x(nd4i);
   elemX[5] = mesh.x(nd5i);
   elemX[6] = mesh.x(nd6i);
   elemX[7] = mesh.x(nd7i);

   elemY[0] = mesh.y(nd0i);
   elemY[1] = mesh.y(nd1i);
   elemY[2] = mesh.y(nd2i);
   elemY[3] = mesh.y(nd3i);
   elemY[4] = mesh.y(nd4i);
   elemY[5] = mesh.y(nd5i);
   elemY[6] = mesh.y(nd6i);
   elemY[7] = mesh.y(nd7i);

   elemZ[0] = mesh.z(nd0i);
   elemZ[1] = mesh.z(nd1i);
   elemZ[2] = mesh.z(nd2i);
   elemZ[3] = mesh.z(nd3i);
   elemZ[4] = mesh.z(nd4i);
   elemZ[5] = mesh.z(nd5i);
   elemZ[6] = mesh.z(nd6i);
   elemZ[7] = mesh.z(nd7i);

}


__host__ 
static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

#if 0
__device__ 
static inline
void VOLUDER(const Real_t a0, const Real_t a1, const Real_t a2,
             const Real_t a3, const Real_t a4, const Real_t a5,
             const Real_t b0, const Real_t b1, const Real_t b2,
             const Real_t b3, const Real_t b4, const Real_t b5,
             Real_t& dvdc)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   dvdc=
      (a1 + a2) * (b0 + b1) - (a0 + a1) * (b1 + b2) +
      (a0 + a4) * (b3 + b4) - (a3 + a4) * (b0 + b4) -
      (a2 + a5) * (b3 + b5) + (a3 + a5) * (b2 + b5);
   dvdc *= twelfth;
}
#else
// Even though the above version is inlined, it seems to prohibit some kind of compiler optimization.
// This macro version uses many fewer registers and avoids spill-over into local memory.
#define VOLUDER(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)		\
{									\
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;			\
									\
   dvdc= 								\
     ((a1) + (a2)) * ((b0) + (b1)) - ((a0) + (a1)) * ((b1) + (b2)) +	\
     ((a0) + (a4)) * ((b3) + (b4)) - ((a3) + (a4)) * ((b0) + (b4)) -	\
     ((a2) + (a5)) * ((b3) + (b5)) + ((a3) + (a5)) * ((b2) + (b5));	\
   dvdc *= twelfth;							\
}
#endif

__host__
static inline
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

__device__
static inline
void CalcElemVolumeDerivative(Real_t& dvdx,
                              Real_t& dvdy,
                              Real_t& dvdz,
                              const Real_t x,
                              const Real_t y,
                              const Real_t z,
			      unsigned int node)
{
  __shared__ Real_t array1[256],array2[256];
  volatile Real_t *va1;
  volatile Real_t *va2;

  unsigned int idx,elem;
  unsigned int ind0,ind1,ind2,ind3,ind4,ind5;

  switch(node) {
  case 0:
    {ind0=1; ind1=2; ind2=3; ind3=4; ind4=5; ind5=7;
    break;}
  case 1:
    {ind0=2; ind1=3; ind2=0; ind3=5; ind4=6; ind5=4;
    break;}
  case 2:
    {ind0=3; ind1=0; ind2=1; ind3=6; ind4=7; ind5=5;
    break;}
  case 3:
    {ind0=0; ind1=1; ind2=2; ind3=7; ind4=4; ind5=6;
    break;}
  case 4:
    {ind0=7; ind1=6; ind2=5; ind3=0; ind4=3; ind5=1;
    break;}
  case 5:
    {ind0=4; ind1=7; ind2=6; ind3=1; ind4=0; ind5=2;
    break;}
  case 6:
    {ind0=5; ind1=4; ind2=7; ind3=2; ind4=1; ind5=3;
    break;}
  case 7:
    {ind0=6; ind1=5; ind2=4; ind3=3; ind4=2; ind5=0;
    break;}
  default:
    {ind0=ind1=ind2=ind3=ind4=ind5=0xFFFFFFFF;
    break;}
  }
  
  idx=threadIdx.x;
  elem=idx /*& 0x1F*/ - node*32;

  va1=&array1[0];
  va2=&array2[0];

  // load y and z
  __syncthreads();
  va1[idx]=y; va2[idx]=z;
  __syncthreads();
  VOLUDER(va1[ind0*32+elem],va1[ind1*32+elem],va1[ind2*32+elem],
	  va1[ind3*32+elem],va1[ind4*32+elem],va1[ind5*32+elem],
	  va2[ind0*32+elem],va2[ind1*32+elem],va2[ind2*32+elem],
	  va2[ind3*32+elem],va2[ind4*32+elem],va2[ind5*32+elem],
	  dvdx);

  // load x
  __syncthreads();
  va1[idx]=x;
  __syncthreads();
  VOLUDER(va2[ind0*32+elem],va2[ind1*32+elem],va2[ind2*32+elem],
	  va2[ind3*32+elem],va2[ind4*32+elem],va2[ind5*32+elem],
	  va1[ind0*32+elem],va1[ind1*32+elem],va1[ind2*32+elem],
	  va1[ind3*32+elem],va1[ind4*32+elem],va1[ind5*32+elem],
	  dvdy);
  __syncthreads();

  // load y
  __syncthreads();
  va2[idx]=y;
  __syncthreads();
  VOLUDER(va1[ind0*32+elem],va1[ind1*32+elem],va1[ind2*32+elem],
	  va1[ind3*32+elem],va1[ind4*32+elem],va1[ind5*32+elem],
	  va2[ind0*32+elem],va2[ind1*32+elem],va2[ind2*32+elem],
	  va2[ind3*32+elem],va2[ind4*32+elem],va2[ind5*32+elem],
	  dvdz);
  __syncthreads();
}

__host__
static inline
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
                              Real_t *hourgam1, Real_t *hourgam2, Real_t *hourgam3,
                              Real_t *hourgam4, Real_t *hourgam5, Real_t *hourgam6,
                              Real_t *hourgam7, Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Index_t i00=0;
   Index_t i01=1;
   Index_t i02=2;
   Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}


__shared__ Real_t shm_array[32*8];

__device__
static inline
Real_t SumOverNodes(Real_t val) {
    // Sum up 8 node values for each element
    // Assumes 256 threads: 32 elements, 8 nodes per element.
    // NOTE: we could probably avoid some of the __syncthreads() if we map 8 nodes 
    //       of an element to the same warp.
    unsigned int tid=threadIdx.x;

#if 1
#if 0
    unsigned int node=tid>>5;
    unsigned int elem=tid-(node<<5);
#elif 1
    unsigned int node=tid/32;
    unsigned int elem=tid-(node*32);
#else
    unsigned int elem=tid & 0x1F;
#endif
    __syncthreads();
    shm_array[tid]=val;
    __syncthreads();
    if (tid<128) shm_array[tid]+=shm_array[tid+128];
    __syncthreads();
    if (tid<64)  shm_array[tid]+=shm_array[tid+64];
    __syncthreads();
    if (tid<32)  shm_array[tid]+=shm_array[tid+32];
    __syncthreads();
    Real_t ret=shm_array[elem];
    __syncthreads();
    return ret;
#else
#if 0
    unsigned int node=tid>>5;
    unsigned int elem=tid-(node<<5);
#else
    unsigned int node=tid/32;
    unsigned int elem=tid-(node*32);
#endif
    unsigned int idx=elem*8+node;
    __syncthreads();
    shm_array[idx]=val;
    __syncthreads();
    if (node<4) shm_array[idx]+=shm_array[idx+4];
    if (node<2) shm_array[idx]+=shm_array[idx+2];
    if (node<1) shm_array[idx]+=shm_array[idx+1];
    __syncthreads();
    return shm_array[elem*8];
#endif
}

__device__
static inline
void CalcElemFBHourglassForce(Real_t xd,Real_t yd,Real_t zd,
                              Real_t *hourgam,Real_t coefficient,
                              Real_t &hgfx, Real_t &hgfy, Real_t &hgfz)
{
    hgfx=0;
    for (int i=0;i<4;i++) {
        Real_t h;
        h=hourgam[i]*xd;
        h=SumOverNodes(h);
        hgfx+=hourgam[i]*h;
    }
    hgfx *= coefficient;

    hgfy=0;
    for (int i=0;i<4;i++) {
        Real_t h;
        h=hourgam[i]*yd;
        h=SumOverNodes(h);
        hgfy+=hourgam[i]*h;
    }
    hgfy *= coefficient;

    hgfz=0;
    for (int i=0;i<4;i++) {
        Real_t h;
        h=hourgam[i]*zd;
        h=SumOverNodes(h);
        hgfz+=hourgam[i]*h;
    }
    hgfz *= coefficient;
}

__global__
void CalcFBHourglassForceForElems_kernel(
    Real_t *determ,
    Real_t *x8n,      Real_t *y8n,      Real_t *z8n,
    Real_t *dvdx,     Real_t *dvdy,     Real_t *dvdz,
    Real_t hourg,
    Index_t numElem, Index_t *nodelist,
    Real_t *ss, Real_t *elemMass,
    Real_t *xd, Real_t *yd, Real_t *zd,
    Real_t *fx_elem, Real_t *fy_elem, Real_t *fz_elem)
{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/

    Real_t hgfx, hgfy, hgfz;
    
    Real_t coefficient;
    
    Real_t hourgam[4];
    Real_t xd1, yd1, zd1;
    
/*************************************************/
/*    compute the hourglass modes */
    
    const Real_t posf = Real_t( 1.);
    const Real_t negf = Real_t(-1.);

    // Assume we will launch 256 threads, which we map to 32 elements, each
    // with 8 per-node threads. Organize so each warp of 32 consecutive
    // threads operates on the same node of different elements.
    

    // THESE ARE ALL GIVING ME DIFFERENT ANSWERS IN CUDA 4.0 !!?!!?!!
    unsigned int tid=threadIdx.x;
    unsigned int bid=blockIdx.x;
#if 0
    unsigned int node=tid>>5;
    unsigned int elem=bid<<5 + (tid - (node<<5));
#elif 1
    unsigned int node=tid/32;
    unsigned int elem=bid*32 + (tid-node*32);
#elif 0
    unsigned int node=tid/32;;
    unsigned int elem=bid*32 + (tid & 0x1F);
#elif 0
    unsigned int node=tid/32;
    unsigned int elem=bid<<5 + (tid & 0x1F);
#elif 0
    unsigned int node=tid>>5;
    unsigned int elem=bid*32 + (tid & 0x1F);
#else
    unsigned int node=tid>>5;
    unsigned int elem=bid<<5 + (tid & 0x1F);
#endif

    if (elem>=numElem) elem=numElem-1; // don't return -- need thread to participate in sync operations

    //if (elem<0) elem=0; // debugging test

    Real_t volinv=Real_t(1.0)/determ[elem];
    Real_t ss1, mass1, volume13 ;

    Real_t xn,yn,zn,dvdxn,dvdyn,dvdzn;
    Real_t hourmodx, hourmody, hourmodz;


#if 1
    xn=x8n[elem+numElem*node]; yn=y8n[elem+numElem*node]; zn=z8n[elem+numElem*node];
    dvdxn=dvdx[elem+numElem*node]; dvdyn=dvdy[elem+numElem*node]; dvdzn=dvdz[elem+numElem*node]; 
#else
    xn=yn=zn=posf; dvdxn=dvdyn=dvdzn=negf;
#endif

#if 1
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==2 || node==3 || node==4 || node==5) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[0] = negf;
    }
    else hourgam[0] = posf;
    hourmodx = SumOverNodes(hourmodx);
    hourmody = SumOverNodes(hourmody);
    hourmodz = SumOverNodes(hourmodz);
    hourgam[0] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==1 || node==2 || node==4 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[1] = negf;
    }
    else hourgam[1] = posf;
    hourmodx = SumOverNodes(hourmodx);
    hourmody = SumOverNodes(hourmody);
    hourmodz = SumOverNodes(hourmodz);
    hourgam[1] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==1 || node==3 || node==5 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[2] = negf;
    }
    else hourgam[2] = posf;
    hourmodx = SumOverNodes(hourmodx);
    hourmody = SumOverNodes(hourmody);
    hourmodz = SumOverNodes(hourmodz);
    hourgam[2] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==0 || node==2 || node==5 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[3] = negf;
    }
    else hourgam[3] = posf;
    hourmodx = SumOverNodes(hourmodx);
    hourmody = SumOverNodes(hourmody);
    hourmodz = SumOverNodes(hourmodz);
    hourgam[3] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);
    
    
    /* compute forces */
    /* store forces into h arrays (force arrays) */
    
    ss1=ss[elem];
    mass1=elemMass[elem];
    volume13=CBRT(determ[elem]);
    
    Index_t ni = nodelist[elem+numElem*node];
    xd1=xd[ni]; yd1=yd[ni]; zd1=zd[ni];
    
    coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;
    
    CalcElemFBHourglassForce(xd1,yd1,zd1,hourgam,coefficient,hgfx,hgfy,hgfz);
#else
    hgfx=xn+dvdxn; hgfy=yn+dvdyn; hgfz=zn+dvdzn;
#endif
#if 1
    fx_elem[elem+numElem*node]=hgfx; fy_elem[elem+numElem*node]=hgfy; fz_elem[elem+numElem*node]=hgfz;
#else
    fx_elem[0]=hgfx; fy_elem[0]=hgfy; fz_elem[0]=hgfz;
#endif
}


static inline
void CalcFBHourglassForceForElems_cpu(Real_t *determ,
            Real_t *x8n,      Real_t *y8n,      Real_t *z8n,
            Real_t *dvdx,     Real_t *dvdy,     Real_t *dvdz,
            Real_t hourg)
{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/

   Index_t numElem = mesh.numElem() ;

   Real_t hgfx[8], hgfy[8], hgfz[8] ;

   Real_t coefficient;

   Real_t  gamma[4][8];
   Real_t hourgam0[4], hourgam1[4], hourgam2[4], hourgam3[4] ;
   Real_t hourgam4[4], hourgam5[4], hourgam6[4], hourgam7[4];
   Real_t xd1[8], yd1[8], zd1[8] ;

   gamma[0][0] = Real_t( 1.);
   gamma[0][1] = Real_t( 1.);
   gamma[0][2] = Real_t(-1.);
   gamma[0][3] = Real_t(-1.);
   gamma[0][4] = Real_t(-1.);
   gamma[0][5] = Real_t(-1.);
   gamma[0][6] = Real_t( 1.);
   gamma[0][7] = Real_t( 1.);
   gamma[1][0] = Real_t( 1.);
   gamma[1][1] = Real_t(-1.);
   gamma[1][2] = Real_t(-1.);
   gamma[1][3] = Real_t( 1.);
   gamma[1][4] = Real_t(-1.);
   gamma[1][5] = Real_t( 1.);
   gamma[1][6] = Real_t( 1.);
   gamma[1][7] = Real_t(-1.);
   gamma[2][0] = Real_t( 1.);
   gamma[2][1] = Real_t(-1.);
   gamma[2][2] = Real_t( 1.);
   gamma[2][3] = Real_t(-1.);
   gamma[2][4] = Real_t( 1.);
   gamma[2][5] = Real_t(-1.);
   gamma[2][6] = Real_t( 1.);
   gamma[2][7] = Real_t(-1.);
   gamma[3][0] = Real_t(-1.);
   gamma[3][1] = Real_t( 1.);
   gamma[3][2] = Real_t(-1.);
   gamma[3][3] = Real_t( 1.);
   gamma[3][4] = Real_t( 1.);
   gamma[3][5] = Real_t(-1.);
   gamma[3][6] = Real_t( 1.);
   gamma[3][7] = Real_t(-1.);

/*************************************************/
/*    compute the hourglass modes */


   for(Index_t i2=0;i2<numElem;++i2){
      Index_t i3=8*i2;
      Real_t volinv=Real_t(1.0)/determ[i2];
      Real_t ss1, mass1, volume13 ;
      for(Index_t i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         Real_t hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam0[i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam1[i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam2[i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam3[i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam4[i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam5[i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam6[i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam7[i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=mesh.ss(i2);
      mass1=mesh.elemMass(i2);
      volume13=CBRT(determ[i2]);

      Index_t n0si2 = mesh.nodelist(i2,0);
      Index_t n1si2 = mesh.nodelist(i2,1);
      Index_t n2si2 = mesh.nodelist(i2,2);
      Index_t n3si2 = mesh.nodelist(i2,3);
      Index_t n4si2 = mesh.nodelist(i2,4);
      Index_t n5si2 = mesh.nodelist(i2,5);
      Index_t n6si2 = mesh.nodelist(i2,6);
      Index_t n7si2 = mesh.nodelist(i2,7);

      xd1[0] = mesh.xd(n0si2);
      xd1[1] = mesh.xd(n1si2);
      xd1[2] = mesh.xd(n2si2);
      xd1[3] = mesh.xd(n3si2);
      xd1[4] = mesh.xd(n4si2);
      xd1[5] = mesh.xd(n5si2);
      xd1[6] = mesh.xd(n6si2);
      xd1[7] = mesh.xd(n7si2);

      yd1[0] = mesh.yd(n0si2);
      yd1[1] = mesh.yd(n1si2);
      yd1[2] = mesh.yd(n2si2);
      yd1[3] = mesh.yd(n3si2);
      yd1[4] = mesh.yd(n4si2);
      yd1[5] = mesh.yd(n5si2);
      yd1[6] = mesh.yd(n6si2);
      yd1[7] = mesh.yd(n7si2);

      zd1[0] = mesh.zd(n0si2);
      zd1[1] = mesh.zd(n1si2);
      zd1[2] = mesh.zd(n2si2);
      zd1[3] = mesh.zd(n3si2);
      zd1[4] = mesh.zd(n4si2);
      zd1[5] = mesh.zd(n5si2);
      zd1[6] = mesh.zd(n6si2);
      zd1[7] = mesh.zd(n7si2);

      coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1,
                      hourgam0,hourgam1,hourgam2,hourgam3,
                      hourgam4,hourgam5,hourgam6,hourgam7,
                      coefficient, hgfx, hgfy, hgfz);

      mesh.fx(n0si2) += hgfx[0];
      mesh.fy(n0si2) += hgfy[0];
      mesh.fz(n0si2) += hgfz[0];

      mesh.fx(n1si2) += hgfx[1];
      mesh.fy(n1si2) += hgfy[1];
      mesh.fz(n1si2) += hgfz[1];

      mesh.fx(n2si2) += hgfx[2];
      mesh.fy(n2si2) += hgfy[2];
      mesh.fz(n2si2) += hgfz[2];

      mesh.fx(n3si2) += hgfx[3];
      mesh.fy(n3si2) += hgfy[3];
      mesh.fz(n3si2) += hgfz[3];

      mesh.fx(n4si2) += hgfx[4];
      mesh.fy(n4si2) += hgfy[4];
      mesh.fz(n4si2) += hgfz[4];

      mesh.fx(n5si2) += hgfx[5];
      mesh.fy(n5si2) += hgfy[5];
      mesh.fz(n5si2) += hgfz[5];

      mesh.fx(n6si2) += hgfx[6];
      mesh.fy(n6si2) += hgfy[6];
      mesh.fz(n6si2) += hgfz[6];

      mesh.fx(n7si2) += hgfx[7];
      mesh.fy(n7si2) += hgfy[7];
      mesh.fz(n7si2) += hgfz[7];
   }
}

static inline
void CalcFBHourglassForceForElems_gpu(Real_t *determ,
            Real_t *x8n,      Real_t *y8n,      Real_t *z8n,
            Real_t *dvdx,     Real_t *dvdy,     Real_t *dvdz,
            Real_t hourg)
{
    Index_t numElem = mesh.numElem();
    Real_t *fx_elem,*fy_elem,*fz_elem;
    
    CUDA( cudaMalloc(&fx_elem,numElem*8*sizeof(Real_t)) );
    CUDA( cudaMalloc(&fy_elem,numElem*8*sizeof(Real_t)) );
    CUDA( cudaMalloc(&fz_elem,numElem*8*sizeof(Real_t)) );
    
    dim3 dimBlock=dim3(256,1,1);
    dim3 dimGrid=dim3(PAD_DIV(numElem*8,dimBlock.x),1,1);
    CalcFBHourglassForceForElems_kernel<<<dimGrid,dimBlock>>>(
        determ,x8n,y8n,z8n,dvdx,dvdy,dvdz,hourg,
        numElem,meshGPU.m_nodelist,
        meshGPU.m_ss,meshGPU.m_elemMass,
        meshGPU.m_xd,meshGPU.m_yd,meshGPU.m_zd,
        fx_elem,fy_elem,fz_elem);
    CUDA_DEBUGSYNC;
    
    dimGrid=dim3(PAD_DIV(mesh.numNode(),dimBlock.x),1,1);
    AddNodeForcesFromElems2_kernel<<<dimGrid,dimBlock>>>
        (mesh.numNode(),meshGPU.m_nodeElemCount,meshGPU.m_nodeElemCornerList,
         fx_elem,fy_elem,fz_elem,meshGPU.m_fx,meshGPU.m_fy,meshGPU.m_fz);
    CUDA_DEBUGSYNC;

    CUDA( cudaFree(fx_elem) );
    CUDA( cudaFree(fy_elem) );
    CUDA( cudaFree(fz_elem) );
}


__global__
void CalcHourglassControlForElems_kernel(Int_t numElem,Index_t *nodelist,
                                         Real_t *x,Real_t *y,Real_t *z,
                                         Real_t *determ,Real_t *volo,Real_t *v,
                                         Real_t *dvdx,Real_t *dvdy,Real_t *dvdz,
                                         Real_t *x8n,Real_t *y8n,Real_t *z8n)
{
    Real_t  x1,y1,z1;
    Real_t pfx,pfy,pfz;
  
    // THESE ARE ALL GIVING ME DIFFERENT ANSWERS IN CUDA 4.0 !!?!!?!!
    unsigned int tid=threadIdx.x;
    unsigned int bid=blockIdx.x;
#if 0
    unsigned int node=tid>>5;
    unsigned int elem=bid<<5 + (tid - (node<<5));
#elif 1
    unsigned int node=tid/32;
    unsigned int elem=bid*32 + (tid-node*32);
#elif 0
    unsigned int node=tid/32;;
    unsigned int elem=bid*32 + (tid & 0x1F);
#elif 0
    unsigned int node=tid/32;
    unsigned int elem=bid<<5 + (tid & 0x1F);
#elif 0
    unsigned int node=tid>>5;
    unsigned int elem=bid*32 + (tid & 0x1F);
#else
    unsigned int node=tid>>5;
    unsigned int elem=bid<<5 + (tid & 0x1F);
#endif
    
    if (elem>=numElem) elem=numElem-1; // don't return -- need thread to participate in sync operations

    Index_t idx=elem+numElem*node;

    Index_t ni = nodelist[idx];
    x1=x[ni]; y1=y[ni]; z1=z[ni];
    
    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1, node);
    
    /* load into temporary storage for FB Hour Glass control */
      
    dvdx[idx] = pfx;
    dvdy[idx] = pfy;
    dvdz[idx] = pfz;
    x8n[idx]  = x1;
    y8n[idx]  = y1;
    z8n[idx]  = z1;
    
    //if (node==0)
      determ[elem] = volo[elem] * v[elem];
    
#if 0 // JDC
      /* Do a check for negative volumes */
    if ( mesh.v(i) <= Real_t(0.0) ) {
      exit(VolumeError) ;
    }
#endif
}


static inline
void CalcHourglassControlForElems_gpu(Real_t determ[], Real_t hgcoef)
{
   Index_t numElem = mesh.numElem() ;
   Index_t numElem8 = numElem * 8 ;
   Real_t *dvdx,*dvdy,*dvdz;
   Real_t *x8n,*y8n,*z8n;

   CUDA( cudaMalloc(&dvdx,sizeof(Real_t)*numElem8) );
   CUDA( cudaMalloc(&dvdy,sizeof(Real_t)*numElem8) );
   CUDA( cudaMalloc(&dvdz,sizeof(Real_t)*numElem8) );
   CUDA( cudaMalloc(&x8n,sizeof(Real_t)*numElem8) );
   CUDA( cudaMalloc(&y8n,sizeof(Real_t)*numElem8) );
   CUDA( cudaMalloc(&z8n,sizeof(Real_t)*numElem8) );

   dim3 dimBlock=dim3(256,1,1);
   dim3 dimGrid=dim3(PAD_DIV(numElem*8,dimBlock.x),1,1);
   CalcHourglassControlForElems_kernel<<<dimGrid,dimBlock>>>
       (numElem, meshGPU.m_nodelist,
        meshGPU.m_x,meshGPU.m_y,meshGPU.m_z,
        determ,meshGPU.m_volo,meshGPU.m_v,
        dvdx,dvdy,dvdz,x8n,y8n,z8n);
   CUDA_DEBUGSYNC;
   
   // JDC -- need a reduction to check for negative volumes

   if ( hgcoef > Real_t(0.) ) {
       CalcFBHourglassForceForElems_gpu(determ,x8n,y8n,z8n,dvdx,dvdy,dvdz,hgcoef) ;
   }
   
   CUDA( cudaFree(dvdx) );
   CUDA( cudaFree(dvdy) );
   CUDA( cudaFree(dvdz) );
   CUDA( cudaFree(x8n) );
   CUDA( cudaFree(y8n) );
   CUDA( cudaFree(z8n) );
   
   return ;
}


static inline
void CalcHourglassControlForElems_cpu(Real_t determ[], Real_t hgcoef)
{
   Index_t i, ii, jj ;
   Real_t  x1[8],  y1[8],  z1[8] ;
   Real_t pfx[8], pfy[8], pfz[8] ;
   Index_t numElem = mesh.numElem() ;
   Index_t numElem8 = numElem * 8 ;
   Real_t *dvdx = Allocate<Real_t>(numElem8) ;
   Real_t *dvdy = Allocate<Real_t>(numElem8) ;
   Real_t *dvdz = Allocate<Real_t>(numElem8) ;
   Real_t *x8n  = Allocate<Real_t>(numElem8) ;
   Real_t *y8n  = Allocate<Real_t>(numElem8) ;
   Real_t *z8n  = Allocate<Real_t>(numElem8) ;

   /* start loop over elements */
   for (i=0 ; i<numElem ; ++i){

      CollectDomainNodesToElemNodes(i, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(ii=0;ii<8;++ii){
         jj=8*i+ii;

         dvdx[jj] = pfx[ii];
         dvdy[jj] = pfy[ii];
         dvdz[jj] = pfz[ii];
         x8n[jj]  = x1[ii];
         y8n[jj]  = y1[ii];
         z8n[jj]  = z1[ii];
      }

      determ[i] = mesh.volo(i) * mesh.v(i);

      /* Do a check for negative volumes */
      if ( mesh.v(i) <= Real_t(0.0) ) {
         exit(VolumeError) ;
      }
   }

   if ( hgcoef > Real_t(0.) ) {
      CalcFBHourglassForceForElems_cpu(determ,x8n,y8n,z8n,dvdx,dvdy,dvdz,hgcoef) ;
   }

   Release(&z8n) ;
   Release(&y8n) ;
   Release(&x8n) ;
   Release(&dvdz) ;
   Release(&dvdy) ;
   Release(&dvdx) ;

   return ;
}


static inline
void CalcHourglassControlForElems(Real_t determ[], Real_t hgcoef, int useCPU)
{
    if (useCPU) {
        FC(x); FC(y); FC(z); FC(xd); FC(yd); FC(zd);
        FC(nodelist); FC(ss); FC(elemMass);
        FC(xd); FC(yd); FC(zd);
        FC(fx); FC(fy); FC(fz);
        CalcHourglassControlForElems_cpu(determ,hgcoef);
        SG(fx); SG(fy); SG(fz);
    }
    else {
        FG(x); FG(y); FG(z); FG(xd); FG(yd); FG(zd);
        FG(nodelist); FG(ss); FG(elemMass);
        FG(xd); FG(yd); FG(zd); 
        FG(fx); FG(fy); FG(fz);
        CalcHourglassControlForElems_gpu(determ,hgcoef);
        SC(fx); SC(fy); SC(fz);
    }
}


static inline
void CalcVolumeForceForElems_gpu()
{
   Index_t numElem = mesh.numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = mesh.hgcoef() ;
      Real_t *sigxx, *sigyy, *sigzz, *determ;
      int badvol;
      
      CUDA( cudaMalloc(&sigxx,numElem*sizeof(Real_t)) );
      CUDA( cudaMalloc(&sigyy,numElem*sizeof(Real_t)) );
      CUDA( cudaMalloc(&sigzz,numElem*sizeof(Real_t)) );
      CUDA( cudaMalloc(&determ,numElem*sizeof(Real_t)) );

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(numElem, sigxx, sigyy, sigzz, 0);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( numElem, sigxx, sigyy, sigzz, determ, badvol, 0) ;
      
      CUDA( cudaFree(sigxx) );
      CUDA( cudaFree(sigyy) );
      CUDA( cudaFree(sigzz) );
      
      // check for negative element volume
      if (badvol) exit(VolumeError) ;

      CalcHourglassControlForElems(determ, hgcoef, 0) ;

      CUDA( cudaFree(determ) );
   }
}


static inline
void CalcVolumeForceForElems_cpu()
{
   Index_t numElem = mesh.numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = mesh.hgcoef() ;
      Real_t *sigxx  = Allocate<Real_t>(numElem) ;
      Real_t *sigyy  = Allocate<Real_t>(numElem) ;
      Real_t *sigzz  = Allocate<Real_t>(numElem) ;
      Real_t *determ = Allocate<Real_t>(numElem) ;
      int badvol;
      
      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(numElem, sigxx, sigyy, sigzz, 1);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( numElem, sigxx, sigyy, sigzz, determ, badvol, 1) ;
      
      Release(&sigzz) ;
      Release(&sigyy) ;
      Release(&sigxx) ;
      
      // check for negative element volume
      if (badvol) exit(VolumeError);
#if 0
      for ( Index_t k=0 ; k<numElem ; ++k ) {
         if (determ[k] <= Real_t(0.0)) {
            exit(VolumeError) ;
         }
      }
#endif
      
      CalcHourglassControlForElems(determ, hgcoef, 1) ;

      Release(&determ) ;
   }
}

static inline void CalcForceForNodes_gpu()
{
  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems_gpu() ;

  /* Calculate Nodal Forces at domain boundaries */
  /* problem->commSBN->Transfer(CommSBN::forces); */
  
  
}

static inline void CalcForceForNodes_cpu()
{
  Index_t numNode = mesh.numNode() ;
  for (Index_t i=0; i<numNode; ++i) {
     mesh.fx(i) = Real_t(0.0) ;
     mesh.fy(i) = Real_t(0.0) ;
     mesh.fz(i) = Real_t(0.0) ;
  }

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems_cpu() ;

  /* Calculate Nodal Forces at domain boundaries */
  /* problem->commSBN->Transfer(CommSBN::forces); */

}

static inline void CalcForceForNodes(int useCPU)
{
    if (useCPU) {
        CalcForceForNodes_cpu();
    }
    else {
        CalcForceForNodes_gpu();
    }
}

__global__
void CalcAccelerationForNodes_kernel(int numNode,
                                     Real_t *xdd, Real_t *ydd, Real_t *zdd,
                                     Real_t *fx, Real_t *fy, Real_t *fz,
                                     Real_t *nodalMass)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numNode) {
        xdd[i]=fx[i]/nodalMass[i];
        ydd[i]=fy[i]/nodalMass[i];
        zdd[i]=fz[i]/nodalMass[i];
    }
}

static inline
void CalcAccelerationForNodes_gpu()
{
    dim3 dimBlock = dim3(BLOCKSIZE,1,1);
    dim3 dimGrid = dim3(PAD_DIV(mesh.numNode(),dimBlock.x),1,1);
    CalcAccelerationForNodes_kernel<<<dimGrid, dimBlock>>>
        (mesh.numNode(),
         meshGPU.m_xdd,meshGPU.m_ydd,meshGPU.m_zdd,
         meshGPU.m_fx,meshGPU.m_fy,meshGPU.m_fz,
         meshGPU.m_nodalMass);
    CUDA_DEBUGSYNC;
}


static inline
void CalcAccelerationForNodes_cpu()
{
   Index_t numNode = mesh.numNode() ;
   for (Index_t i = 0; i < numNode; ++i) {
      mesh.xdd(i) = mesh.fx(i) / mesh.nodalMass(i);
      mesh.ydd(i) = mesh.fy(i) / mesh.nodalMass(i);
      mesh.zdd(i) = mesh.fz(i) / mesh.nodalMass(i);
   }
}

static inline
void CalcAccelerationForNodes(int useCPU)
{
    if (useCPU) {
        FC(fx); FC(fy); FC(fz); FC(nodalMass);
        CalcAccelerationForNodes_cpu();
        SG(xdd); SG(ydd); SG(zdd);
    }
    else {
        FG(fx); FG(fy); FG(fz); FG(nodalMass);
        CalcAccelerationForNodes_gpu();
        SC(xdd); SC(ydd); SC(zdd);
    }
}

__global__
void ApplyAccelerationBoundaryConditionsForNodes_kernel(
    int numNodeBC, Real_t *xdd, Real_t *ydd, Real_t *zdd,
    Index_t *symmX, Index_t *symmY, Index_t *symmZ)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numNodeBC) {
        xdd[symmX[i]] = Real_t(0.0) ;
        ydd[symmY[i]] = Real_t(0.0) ;
        zdd[symmZ[i]] = Real_t(0.0) ;
    }
}

static inline
void ApplyAccelerationBoundaryConditionsForNodes_gpu()
{
    Index_t numNodeBC = (mesh.sizeX()+1)*(mesh.sizeX()+1) ;
    dim3 dimBlock(BLOCKSIZE,1,1);
    dim3 dimGrid(PAD_DIV(numNodeBC,dimBlock.x),1,1);
    ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (numNodeBC,
         meshGPU.m_xdd,meshGPU.m_ydd,meshGPU.m_zdd,
         meshGPU.m_symmX,meshGPU.m_symmY,meshGPU.m_symmZ);
    CUDA_DEBUGSYNC;
}

static inline
void ApplyAccelerationBoundaryConditionsForNodes_cpu()
{
  Index_t numNodeBC = (mesh.sizeX()+1)*(mesh.sizeX()+1) ;
  for(Index_t i=0 ; i<numNodeBC ; ++i)
     mesh.xdd(mesh.symmX(i)) = Real_t(0.0) ;

  for(Index_t i=0 ; i<numNodeBC ; ++i)
     mesh.ydd(mesh.symmY(i)) = Real_t(0.0) ;

  for(Index_t i=0 ; i<numNodeBC ; ++i)
     mesh.zdd(mesh.symmZ(i)) = Real_t(0.0) ;
}

static inline
void ApplyAccelerationBoundaryConditionsForNodes(int useCPU)
{
    if (useCPU) {
        FC(xdd); FC(ydd); FC(zdd); FC(symmX); FC(symmY); FC(symmZ);
        ApplyAccelerationBoundaryConditionsForNodes_cpu();
        SG(xdd); SG(ydd); SG(zdd);
    }
    else {
        FG(xdd); FG(ydd); FG(zdd); FG(symmX); FG(symmY); FG(symmZ);
        ApplyAccelerationBoundaryConditionsForNodes_gpu();
        SC(xdd); SC(ydd); SC(zdd);
    }
}


__global__
void CalcVelocityForNodes_kernel(int numNode, const Real_t dt, const Real_t u_cut,
                                 Real_t *xd, Real_t *yd, Real_t *zd,
                                 Real_t *xdd, Real_t *ydd, Real_t *zdd)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numNode) {
        Real_t xdtmp, ydtmp, zdtmp ;
        
        xdtmp = xd[i] + xdd[i] * dt ;
        if( FABS(xdtmp) < u_cut ) xdtmp = 0.0;//Real_t(0.0);
        xd[i] = xdtmp ;
        
        ydtmp = yd[i] + ydd[i] * dt ;
        if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
        yd[i] = ydtmp ;
        
        zdtmp = zd[i] + zdd[i] * dt ;
        if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
        zd[i] = zdtmp ;
    }
}

static inline
void CalcVelocityForNodes_gpu(const Real_t dt, const Real_t u_cut)
{
    dim3 dimBlock(BLOCKSIZE,1,1);
    dim3 dimGrid(PAD_DIV(mesh.numNode(),dimBlock.x),1,1);
    CalcVelocityForNodes_kernel<<<dimGrid, dimBlock>>>
        (mesh.numNode(),dt,u_cut,
         meshGPU.m_xd,meshGPU.m_yd,meshGPU.m_zd,
         meshGPU.m_xdd,meshGPU.m_ydd,meshGPU.m_zdd);
    CUDA_DEBUGSYNC;
}

static inline
void CalcVelocityForNodes_cpu(const Real_t dt, const Real_t u_cut)
{
   Index_t numNode = mesh.numNode() ;

   for ( Index_t i = 0 ; i < numNode ; ++i )
   {
     Real_t xdtmp, ydtmp, zdtmp ;

     xdtmp = mesh.xd(i) + mesh.xdd(i) * dt ;
     if( FABS(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
     mesh.xd(i) = xdtmp ;

     ydtmp = mesh.yd(i) + mesh.ydd(i) * dt ;
     if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
     mesh.yd(i) = ydtmp ;

     zdtmp = mesh.zd(i) + mesh.zdd(i) * dt ;
     if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
     mesh.zd(i) = zdtmp ;
   }
}

static inline
void CalcVelocityForNodes(const Real_t dt, const Real_t u_cut, int useCPU)
{
    if (useCPU) {
        FC(xd); FC(yd); FC(zd); FC(xdd); FC(ydd); FC(zdd);
        CalcVelocityForNodes_cpu(dt,u_cut);
        SG(xd); SG(yd); SG(zd);
    }
    else {
        FG(xd); FG(yd); FG(zd); FG(xdd); FG(ydd); FG(zdd);
        CalcVelocityForNodes_gpu(dt,u_cut);
        SC(xd); SC(yd); SC(zd);
    }
}

__global__
void CalcPositionForNodes_kernel(int numNode, Real_t dt,
                                 Real_t *x, Real_t *y, Real_t *z,
                                 Real_t *xd, Real_t *yd, Real_t *zd)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numNode) {
        x[i] += xd[i] * dt;
        y[i] += yd[i] * dt;
        z[i] += zd[i] * dt;
    }
}

static inline
void CalcPositionForNodes_gpu(const Real_t dt)
{
    dim3 dimBlock(BLOCKSIZE,1,1);
    dim3 dimGrid(PAD_DIV(mesh.numNode(),dimBlock.x),1,1);
    CalcPositionForNodes_kernel<<<dimGrid, dimBlock>>>
        (mesh.numNode(),dt,meshGPU.m_x,meshGPU.m_y,meshGPU.m_z,meshGPU.m_xd,meshGPU.m_yd,meshGPU.m_zd);
    CUDA_DEBUGSYNC;
}

static inline
void CalcPositionForNodes_cpu(const Real_t dt)
{
   Index_t numNode = mesh.numNode() ;

   for ( Index_t i = 0 ; i < numNode ; ++i )
   {
     mesh.x(i) += mesh.xd(i) * dt ;
     mesh.y(i) += mesh.yd(i) * dt ;
     mesh.z(i) += mesh.zd(i) * dt ;
   }
}

static inline
void CalcPositionForNodes(const Real_t dt,int useCPU)
{
    if (useCPU) {
        FC(x); FC(y); FC(z); FC(xd); FC(yd); FC(zd);
        CalcPositionForNodes_cpu(dt);
        SG(x); SG(y); SG(z);
    }
    else {
        FG(x); FG(y); FG(z); FG(xd); FG(yd); FG(zd);
        CalcPositionForNodes_gpu(dt);
        SC(x); SC(y); SC(z);
    }
}

static inline
void LagrangeNodal(int useCPU)
{
  const Real_t delt = mesh.deltatime() ;
  Real_t u_cut = mesh.u_cut() ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(/*0*/useCPU);

  CalcAccelerationForNodes(useCPU);

  ApplyAccelerationBoundaryConditionsForNodes(useCPU);

  CalcVelocityForNodes( delt, u_cut, useCPU ) ;

  CalcPositionForNodes( delt, useCPU );

  return;
}

__host__ __device__
static inline
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

__host__ __device__
static inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

__host__ __device__
static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

__host__ __device__
static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = FMAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

__host__ __device__
static inline
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

__global__
void CalcKinematicsForElems_kernel(
    Index_t numElem, Real_t dt,
    Index_t *nodelist,Real_t *volo,Real_t *v,
    Real_t *x,Real_t *y,Real_t *z,Real_t *xd,Real_t *yd,Real_t *zd,
    Real_t *vnew,Real_t *delv,Real_t *arealg,Real_t *dxx,Real_t *dyy,Real_t *dzz
    )
{
  Real_t B[3][8] ; /** shape function derivatives */
  Real_t D[6] ;
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  Real_t xd_local[8] ;
  Real_t yd_local[8] ;
  Real_t zd_local[8] ;
  Real_t detJ = Real_t(0.0) ;

  int k=blockDim.x*blockIdx.x + threadIdx.x;
  if (k<numElem) {

    Real_t volume ;
    Real_t relativeVolume ;

    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*numElem];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode];
    }

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / volo[k] ;
    vnew[k] = relativeVolume ;
    delv[k] = relativeVolume - v[k] ;

    // set characteristic length
    arealg[k] = CalcElemCharacteristicLength(x_local,y_local,z_local,volume);

    // get nodal velocities from global array and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*numElem];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
    }

    Real_t dt2 = Real_t(0.5) * dt;
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives(x_local,y_local,z_local,B,&detJ );

    CalcElemVelocityGradient(xd_local,yd_local,zd_local,B,detJ,D);

    // put velocity gradient quantities into their global arrays.
    dxx[k] = D[0];
    dyy[k] = D[1];
    dzz[k] = D[2];
  }
}


static inline
void CalcKinematicsForElems_gpu( Index_t numElem, Real_t dt )
{
    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(numElem,dimBlock.x),1,1);
    CalcKinematicsForElems_kernel<<<dimGrid,dimBlock>>>
        (numElem,dt,meshGPU.m_nodelist,meshGPU.m_volo,meshGPU.m_v,
         meshGPU.m_x,meshGPU.m_y,meshGPU.m_z,meshGPU.m_xd,meshGPU.m_yd,meshGPU.m_zd,
         meshGPU.m_vnew,meshGPU.m_delv,meshGPU.m_arealg,meshGPU.m_dxx,meshGPU.m_dyy,meshGPU.m_dzz);
    CUDA_DEBUGSYNC;
}


static inline
void CalcKinematicsForElems_cpu( Index_t numElem, Real_t dt )
{
  Real_t B[3][8] ; /** shape function derivatives */
  Real_t D[6] ;
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  Real_t xd_local[8] ;
  Real_t yd_local[8] ;
  Real_t zd_local[8] ;
  Real_t detJ = Real_t(0.0) ;

  // loop over all elements
  for( Index_t k=0 ; k<numElem ; ++k )
  {
    Real_t volume ;
    Real_t relativeVolume ;

    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = mesh.nodelist(k,lnode);
      x_local[lnode] = mesh.x(gnode);
      y_local[lnode] = mesh.y(gnode);
      z_local[lnode] = mesh.z(gnode);
    }

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / mesh.volo(k) ;
    mesh.vnew(k) = relativeVolume ;
    mesh.delv(k) = relativeVolume - mesh.v(k) ;

    // set characteristic length
    mesh.arealg(k) = CalcElemCharacteristicLength(x_local,
                                                  y_local,
                                                  z_local,
                                                  volume);

    // get nodal velocities from global array and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = mesh.nodelist(k,lnode);
      xd_local[lnode] = mesh.xd(gnode);
      yd_local[lnode] = mesh.yd(gnode);
      zd_local[lnode] = mesh.zd(gnode);
    }

    Real_t dt2 = Real_t(0.5) * dt;
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives( x_local,
                                      y_local,
                                      z_local,
                                      B, &detJ );

    CalcElemVelocityGradient( xd_local,
                               yd_local,
                               zd_local,
                               B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    mesh.dxx(k) = D[0];
    mesh.dyy(k) = D[1];
    mesh.dzz(k) = D[2];
  }
}


static inline
void CalcKinematicsForElems( Index_t numElem, Real_t dt, int useCPU )
{
    if (useCPU) {
        FC(nodelist); FC(volo); FC(v); FC(x); FC(y); FC(z); FC(xd); FC(yd); FC(zd);
        CalcKinematicsForElems_cpu(numElem,dt);
        SG(vnew); SG(delv); SG(arealg); SG(dxx); SG(dyy); SG(dzz);
    }
    else {
        FG(nodelist); FG(volo); FG(v); FG(x); FG(y); FG(z); FG(xd); FG(yd); FG(zd);
        CalcKinematicsForElems_gpu(numElem,dt);
        SC(vnew); SC(delv); SC(arealg); SC(dxx); SC(dyy); SC(dzz);
    }
}


__global__
void CalcLagrangeElementsPart2_kernel(
    Index_t numElem,
    Real_t *dxx,Real_t *dyy, Real_t *dzz,
    Real_t *vdov
    )
{
    int k=blockDim.x*blockIdx.x + threadIdx.x;
    if (k<numElem) {

        // calc strain rate and apply as constraint (only done in FB element)
        Real_t vdovNew = dxx[k] + dyy[k] + dzz[k] ;
        Real_t vdovthird = vdovNew/Real_t(3.0) ;
        
        // make the rate of deformation tensor deviatoric
        vdov[k] = vdovNew ;
        dxx[k] -= vdovthird ;
        dyy[k] -= vdovthird ;
        dzz[k] -= vdovthird ;
        
        // See if any volumes are negative, and take appropriate action.
        //if (mesh.vnew(k) <= Real_t(0.0))
        //{
        //    exit(VolumeError) ;
        //}
    }
}


static inline
void CalcLagrangeElementsPart2_gpu()
{
    Index_t numElem = mesh.numElem();
    
    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(numElem,dimBlock.x),1,1);
    CalcLagrangeElementsPart2_kernel<<<dimGrid,dimBlock>>>
        (numElem,
         meshGPU.m_dxx,meshGPU.m_dyy,meshGPU.m_dzz,
         meshGPU.m_vdov);
    CUDA_DEBUGSYNC;
}


static inline
void CalcLagrangeElementsPart2_cpu()
{
   Index_t numElem = mesh.numElem() ;

   // element loop to do some stuff not included in the elemlib function.
   for ( Index_t k=0 ; k<numElem ; ++k )
   {
       // calc strain rate and apply as constraint (only done in FB element)
       Real_t vdov = mesh.dxx(k) + mesh.dyy(k) + mesh.dzz(k) ;
       Real_t vdovthird = vdov/Real_t(3.0) ;
       
       // make the rate of deformation tensor deviatoric
       mesh.vdov(k) = vdov ;
       mesh.dxx(k) -= vdovthird ;
       mesh.dyy(k) -= vdovthird ;
       mesh.dzz(k) -= vdovthird ;
       
       // See if any volumes are negative, and take appropriate action.
       if (mesh.vnew(k) <= Real_t(0.0))
       {
           exit(VolumeError) ;
       }
   }
}


static inline
void CalcLagrangeElementsPart2(int useCPU)
{
    if (useCPU) {
        FC(dxx); FC(dyy); FC(dzz);
        CalcLagrangeElementsPart2_cpu();
        SG(vdov); SG(dxx); SG(dyy); SG(dzz);
    }
    else {
        FG(dxx); FG(dyy); FG(dzz);
        CalcLagrangeElementsPart2_gpu();
        SC(vdov); SC(dxx); SC(dyy); SC(dzz);
    }
}

static inline
void CalcLagrangeElements(Real_t deltatime, int useCPU)
{
   Index_t numElem = mesh.numElem() ;
   if (numElem > 0) {
       CalcKinematicsForElems(numElem, deltatime, useCPU);
       CalcLagrangeElementsPart2(useCPU);
   }
}


__global__
void CalcMonotonicQGradientsForElems_kernel(
    Index_t numElem,
    Index_t *nodelist,
    Real_t *x,Real_t *y,Real_t *z,Real_t *xd,Real_t *yd,Real_t *zd,
    Real_t *volo,Real_t *vnew,
    Real_t *delx_zeta,Real_t *delv_zeta,
    Real_t *delx_xi,Real_t *delv_xi,
    Real_t *delx_eta,Real_t *delv_eta
    )
{
#define SUM4(a,b,c,d) (a + b + c + d)
   const Real_t ptiny = Real_t(1.e-36) ;

   int i=blockDim.x*blockIdx.x + threadIdx.x;
   if (i<numElem) {
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      Index_t n0 = nodelist[i+0*numElem] ;
      Index_t n1 = nodelist[i+1*numElem] ;
      Index_t n2 = nodelist[i+2*numElem] ;
      Index_t n3 = nodelist[i+3*numElem] ;
      Index_t n4 = nodelist[i+4*numElem] ;
      Index_t n5 = nodelist[i+5*numElem] ;
      Index_t n6 = nodelist[i+6*numElem] ;
      Index_t n7 = nodelist[i+7*numElem] ;

      Real_t x0 = x[n0] ;
      Real_t x1 = x[n1] ;
      Real_t x2 = x[n2] ;
      Real_t x3 = x[n3] ;
      Real_t x4 = x[n4] ;
      Real_t x5 = x[n5] ;
      Real_t x6 = x[n6] ;
      Real_t x7 = x[n7] ;

      Real_t y0 = y[n0] ;
      Real_t y1 = y[n1] ;
      Real_t y2 = y[n2] ;
      Real_t y3 = y[n3] ;
      Real_t y4 = y[n4] ;
      Real_t y5 = y[n5] ;
      Real_t y6 = y[n6] ;
      Real_t y7 = y[n7] ;

      Real_t z0 = z[n0] ;
      Real_t z1 = z[n1] ;
      Real_t z2 = z[n2] ;
      Real_t z3 = z[n3] ;
      Real_t z4 = z[n4] ;
      Real_t z5 = z[n5] ;
      Real_t z6 = z[n6] ;
      Real_t z7 = z[n7] ;

      Real_t xv0 = xd[n0] ;
      Real_t xv1 = xd[n1] ;
      Real_t xv2 = xd[n2] ;
      Real_t xv3 = xd[n3] ;
      Real_t xv4 = xd[n4] ;
      Real_t xv5 = xd[n5] ;
      Real_t xv6 = xd[n6] ;
      Real_t xv7 = xd[n7] ;

      Real_t yv0 = yd[n0] ;
      Real_t yv1 = yd[n1] ;
      Real_t yv2 = yd[n2] ;
      Real_t yv3 = yd[n3] ;
      Real_t yv4 = yd[n4] ;
      Real_t yv5 = yd[n5] ;
      Real_t yv6 = yd[n6] ;
      Real_t yv7 = yd[n7] ;

      Real_t zv0 = zd[n0] ;
      Real_t zv1 = zd[n1] ;
      Real_t zv2 = zd[n2] ;
      Real_t zv3 = zd[n3] ;
      Real_t zv4 = zd[n4] ;
      Real_t zv5 = zd[n5] ;
      Real_t zv6 = zd[n6] ;
      Real_t zv7 = zd[n7] ;

      Real_t vol = volo[i]*vnew[i] ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
      Real_t dyj = Real_t(-0.25)*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
      Real_t dzj = Real_t(-0.25)*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

      Real_t dxi = Real_t( 0.25)*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
      Real_t dyi = Real_t( 0.25)*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
      Real_t dzi = Real_t( 0.25)*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

      Real_t dxk = Real_t( 0.25)*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
      Real_t dyk = Real_t( 0.25)*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
      Real_t dzk = Real_t( 0.25)*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      delx_zeta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
      dyv = Real_t(0.25)*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
      dzv = Real_t(0.25)*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

      delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      delx_xi[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
      dyv = Real_t(0.25)*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
      dzv = Real_t(0.25)*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

      delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      delx_eta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(-0.25)*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
      dyv = Real_t(-0.25)*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
      dzv = Real_t(-0.25)*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

      delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
   }
#undef SUM4
}


static inline
void CalcMonotonicQGradientsForElems_gpu()
{
    Index_t numElem = mesh.numElem();
    
    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(numElem,dimBlock.x),1,1);
    CalcMonotonicQGradientsForElems_kernel<<<dimGrid,dimBlock>>>
        (numElem,
         meshGPU.m_nodelist,
         meshGPU.m_x,meshGPU.m_y,meshGPU.m_z,meshGPU.m_xd,meshGPU.m_yd,meshGPU.m_zd,
         meshGPU.m_volo,meshGPU.m_vnew,
         meshGPU.m_delx_zeta,meshGPU.m_delv_zeta,
         meshGPU.m_delx_xi,meshGPU.m_delv_xi,
         meshGPU.m_delx_eta,meshGPU.m_delv_eta);
    CUDA_DEBUGSYNC;
}


static inline
void CalcMonotonicQGradientsForElems_cpu()
{
#define SUM4(a,b,c,d) (a + b + c + d)
   Index_t numElem = mesh.numElem() ;
   const Real_t ptiny = Real_t(1.e-36) ;

   for (Index_t i = 0 ; i < numElem ; ++i ) {
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      Index_t n0 = mesh.nodelist(i,0) ;
      Index_t n1 = mesh.nodelist(i,1) ;
      Index_t n2 = mesh.nodelist(i,2) ;
      Index_t n3 = mesh.nodelist(i,3) ;
      Index_t n4 = mesh.nodelist(i,4) ;
      Index_t n5 = mesh.nodelist(i,5) ;
      Index_t n6 = mesh.nodelist(i,6) ;
      Index_t n7 = mesh.nodelist(i,7) ;

      Real_t x0 = mesh.x(n0) ;
      Real_t x1 = mesh.x(n1) ;
      Real_t x2 = mesh.x(n2) ;
      Real_t x3 = mesh.x(n3) ;
      Real_t x4 = mesh.x(n4) ;
      Real_t x5 = mesh.x(n5) ;
      Real_t x6 = mesh.x(n6) ;
      Real_t x7 = mesh.x(n7) ;

      Real_t y0 = mesh.y(n0) ;
      Real_t y1 = mesh.y(n1) ;
      Real_t y2 = mesh.y(n2) ;
      Real_t y3 = mesh.y(n3) ;
      Real_t y4 = mesh.y(n4) ;
      Real_t y5 = mesh.y(n5) ;
      Real_t y6 = mesh.y(n6) ;
      Real_t y7 = mesh.y(n7) ;

      Real_t z0 = mesh.z(n0) ;
      Real_t z1 = mesh.z(n1) ;
      Real_t z2 = mesh.z(n2) ;
      Real_t z3 = mesh.z(n3) ;
      Real_t z4 = mesh.z(n4) ;
      Real_t z5 = mesh.z(n5) ;
      Real_t z6 = mesh.z(n6) ;
      Real_t z7 = mesh.z(n7) ;

      Real_t xv0 = mesh.xd(n0) ;
      Real_t xv1 = mesh.xd(n1) ;
      Real_t xv2 = mesh.xd(n2) ;
      Real_t xv3 = mesh.xd(n3) ;
      Real_t xv4 = mesh.xd(n4) ;
      Real_t xv5 = mesh.xd(n5) ;
      Real_t xv6 = mesh.xd(n6) ;
      Real_t xv7 = mesh.xd(n7) ;

      Real_t yv0 = mesh.yd(n0) ;
      Real_t yv1 = mesh.yd(n1) ;
      Real_t yv2 = mesh.yd(n2) ;
      Real_t yv3 = mesh.yd(n3) ;
      Real_t yv4 = mesh.yd(n4) ;
      Real_t yv5 = mesh.yd(n5) ;
      Real_t yv6 = mesh.yd(n6) ;
      Real_t yv7 = mesh.yd(n7) ;

      Real_t zv0 = mesh.zd(n0) ;
      Real_t zv1 = mesh.zd(n1) ;
      Real_t zv2 = mesh.zd(n2) ;
      Real_t zv3 = mesh.zd(n3) ;
      Real_t zv4 = mesh.zd(n4) ;
      Real_t zv5 = mesh.zd(n5) ;
      Real_t zv6 = mesh.zd(n6) ;
      Real_t zv7 = mesh.zd(n7) ;

      Real_t vol = mesh.volo(i)*mesh.vnew(i) ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
      Real_t dyj = Real_t(-0.25)*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
      Real_t dzj = Real_t(-0.25)*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

      Real_t dxi = Real_t( 0.25)*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
      Real_t dyi = Real_t( 0.25)*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
      Real_t dzi = Real_t( 0.25)*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

      Real_t dxk = Real_t( 0.25)*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
      Real_t dyk = Real_t( 0.25)*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
      Real_t dzk = Real_t( 0.25)*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      mesh.delx_zeta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
      dyv = Real_t(0.25)*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
      dzv = Real_t(0.25)*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

      mesh.delv_zeta(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      mesh.delx_xi(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
      dyv = Real_t(0.25)*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
      dzv = Real_t(0.25)*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

      mesh.delv_xi(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      mesh.delx_eta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(-0.25)*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
      dyv = Real_t(-0.25)*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
      dzv = Real_t(-0.25)*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

      mesh.delv_eta(i) = ax*dxv + ay*dyv + az*dzv ;
   }
#undef SUM4
}


static inline
void CalcMonotonicQGradientsForElems(int useCPU)
{
    if (useCPU) {
        FC(nodelist); FC(x); FC(y); FC(z); FC(xd); FC(yd); FC(zd); FC(volo); FC(vnew);
        CalcMonotonicQGradientsForElems_cpu();
        SG(delx_zeta); SG(delv_zeta); SG(delx_xi); SG(delv_xi); SG(delx_eta); SG(delv_eta);
    }
    else {
        FG(nodelist); FG(x); FG(y); FG(z); FG(xd); FG(yd); FG(zd); FG(volo); FG(vnew);
        CalcMonotonicQGradientsForElems_gpu();
        SC(delx_zeta); SC(delv_zeta); SC(delx_xi); SC(delv_xi); SC(delx_eta); SC(delv_eta);
    }
}


__global__
void CalcMonotonicQRegionForElems_kernel(
    Real_t qlc_monoq,
    Real_t qqc_monoq,
    Real_t monoq_limiter_mult,
    Real_t monoq_max_slope,
    Real_t ptiny,
    
    // the elementset length
    Index_t elength,
    
    Index_t *matElemlist,Index_t *elemBC,
    Index_t *lxim,Index_t *lxip,
    Index_t *letam,Index_t *letap,
    Index_t *lzetam,Index_t *lzetap,
    Real_t *delv_xi,Real_t *delv_eta,Real_t *delv_zeta,
    Real_t *delx_xi,Real_t *delx_eta,Real_t *delx_zeta,
    Real_t *vdov,Real_t *elemMass,Real_t *volo,Real_t *vnew,
    Real_t *qq,Real_t *ql
    )
{
    int ielem=blockDim.x*blockIdx.x + threadIdx.x;
    if (ielem<elength) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = matElemlist[ielem];
      Int_t bcMask = elemBC[i] ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / ( delv_xi[i] + ptiny ) ;

      switch (bcMask & XI_M) {
         case 0:         delvm = delv_xi[lxim[i]] ; break ;
         case XI_M_SYMM: delvm = delv_xi[i] ;            break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }
      switch (bcMask & XI_P) {
         case 0:         delvp = delv_xi[lxip[i]] ; break ;
         case XI_P_SYMM: delvp = delv_xi[i] ;            break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
         case 0:          delvm = delv_eta[letam[i]] ; break ;
         case ETA_M_SYMM: delvm = delv_eta[i] ;             break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }
      switch (bcMask & ETA_P) {
         case 0:          delvp = delv_eta[letap[i]] ; break ;
         case ETA_P_SYMM: delvp = delv_eta[i] ;             break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( delv_zeta[i] + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case 0:           delvm = delv_zeta[lzetam[i]] ; break ;
         case ZETA_M_SYMM: delvm = delv_zeta[i] ;              break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }
      switch (bcMask & ZETA_P) {
         case 0:           delvp = delv_zeta[lzetap[i]] ; break ;
         case ZETA_P_SYMM: delvp = delv_zeta[i] ;              break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( vdov[i] > Real_t(0.) )  {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
         Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
         Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      qq[i] = qquad ;
      ql[i] = qlin  ;
   }
}


static inline
void CalcMonotonicQRegionForElems_gpu(// parameters
                          Real_t qlc_monoq,
                          Real_t qqc_monoq,
                          Real_t monoq_limiter_mult,
                          Real_t monoq_max_slope,
                          Real_t ptiny,

                          // the elementset length
                          Index_t elength )
{
    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(elength,dimBlock.x),1,1);
    CalcMonotonicQRegionForElems_kernel<<<dimGrid,dimBlock>>>
        (qlc_monoq,qqc_monoq,monoq_limiter_mult,monoq_max_slope,ptiny,elength,
         meshGPU.m_matElemlist,meshGPU.m_elemBC,
         meshGPU.m_lxim,meshGPU.m_lxip,
         meshGPU.m_letam,meshGPU.m_letap,
         meshGPU.m_lzetam,meshGPU.m_lzetap,
         meshGPU.m_delv_xi,meshGPU.m_delv_eta,meshGPU.m_delv_zeta,
         meshGPU.m_delx_xi,meshGPU.m_delx_eta,meshGPU.m_delx_zeta,
         meshGPU.m_vdov,meshGPU.m_elemMass,meshGPU.m_volo,meshGPU.m_vnew,
         meshGPU.m_qq,meshGPU.m_ql);
    CUDA_DEBUGSYNC;
}


static inline
void CalcMonotonicQRegionForElems_cpu(// parameters
                          Real_t qlc_monoq,
                          Real_t qqc_monoq,
                          Real_t monoq_limiter_mult,
                          Real_t monoq_max_slope,
                          Real_t ptiny,

                          // the elementset length
                          Index_t elength )
{
   for ( Index_t ielem = 0 ; ielem < elength; ++ielem ) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = mesh.matElemlist(ielem);
      Int_t bcMask = mesh.elemBC(i) ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / ( mesh.delv_xi(i) + ptiny ) ;

      switch (bcMask & XI_M) {
         case 0:         delvm = mesh.delv_xi(mesh.lxim(i)) ; break ;
         case XI_M_SYMM: delvm = mesh.delv_xi(i) ;            break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }
      switch (bcMask & XI_P) {
         case 0:         delvp = mesh.delv_xi(mesh.lxip(i)) ; break ;
         case XI_P_SYMM: delvp = mesh.delv_xi(i) ;            break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( mesh.delv_eta(i) + ptiny ) ;

      switch (bcMask & ETA_M) {
         case 0:          delvm = mesh.delv_eta(mesh.letam(i)) ; break ;
         case ETA_M_SYMM: delvm = mesh.delv_eta(i) ;             break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }
      switch (bcMask & ETA_P) {
         case 0:          delvp = mesh.delv_eta(mesh.letap(i)) ; break ;
         case ETA_P_SYMM: delvp = mesh.delv_eta(i) ;             break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( mesh.delv_zeta(i) + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case 0:           delvm = mesh.delv_zeta(mesh.lzetam(i)) ; break ;
         case ZETA_M_SYMM: delvm = mesh.delv_zeta(i) ;              break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }
      switch (bcMask & ZETA_P) {
         case 0:           delvp = mesh.delv_zeta(mesh.lzetap(i)) ; break ;
         case ZETA_P_SYMM: delvp = mesh.delv_zeta(i) ;              break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( mesh.vdov(i) > Real_t(0.) )  {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = mesh.delv_xi(i)   * mesh.delx_xi(i)   ;
         Real_t delvxeta  = mesh.delv_eta(i)  * mesh.delx_eta(i)  ;
         Real_t delvxzeta = mesh.delv_zeta(i) * mesh.delx_zeta(i) ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = mesh.elemMass(i) / (mesh.volo(i) * mesh.vnew(i)) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      mesh.qq(i) = qquad ;
      mesh.ql(i) = qlin  ;
   }
}

static inline
void CalcMonotonicQRegionForElems(// parameters
                          Real_t qlc_monoq,
                          Real_t qqc_monoq,
                          Real_t monoq_limiter_mult,
                          Real_t monoq_max_slope,
                          Real_t ptiny,

                          // the elementset length
                          Index_t elength,
                          int useCPU)
{
    if (useCPU) {
        FC(matElemlist); FC(elemBC); FC(lxim); FC(lxip); FC(letam); FC(letap); FC(lzetam); FC(lzetap);
        FC(delv_xi); FC(delv_eta); FC(delv_zeta); FC(delx_xi); FC(delx_eta); FC(delx_zeta);
        FC(vdov); FC(elemMass); FC(volo); FC(vnew);
        CalcMonotonicQRegionForElems_cpu(qlc_monoq,qqc_monoq,
                                     monoq_limiter_mult,monoq_max_slope,ptiny,
                                     elength);
        SG(qq); SG(ql);
    }
    else {
        FG(matElemlist); FG(elemBC); FG(lxim); FG(lxip); FG(letam); FG(letap); FG(lzetam); FG(lzetap);
        FG(delv_xi); FG(delv_eta); FG(delv_zeta); FG(delx_xi); FG(delx_eta); FG(delx_zeta);
        FG(vdov); FG(elemMass); FG(volo); FG(vnew);
        CalcMonotonicQRegionForElems_gpu(qlc_monoq,qqc_monoq,
                                     monoq_limiter_mult,monoq_max_slope,ptiny,
                                     elength);
        SC(qq); SC(ql);
    }
}

static inline
void CalcMonotonicQForElems(int useCPU)
{  
   //
   // initialize parameters
   // 
   const Real_t ptiny        = Real_t(1.e-36) ;
   Real_t monoq_max_slope    = mesh.monoq_max_slope() ;
   Real_t monoq_limiter_mult = mesh.monoq_limiter_mult() ;

   //
   // calculate the monotonic q for pure regions
   //
   Index_t elength = mesh.numElem() ;
   if (elength > 0) {
      Real_t qlc_monoq = mesh.qlc_monoq();
      Real_t qqc_monoq = mesh.qqc_monoq();
      CalcMonotonicQRegionForElems(// parameters
                           qlc_monoq,
                           qqc_monoq,
                           monoq_limiter_mult,
                           monoq_max_slope,
                           ptiny,

                           // the elemset length
                           elength,
                           useCPU);
   }
}

static inline
void CalcQForElems(int useCPU)
{
   Real_t qstop = mesh.qstop() ;
   Index_t numElem = mesh.numElem() ;

   //
   // MONOTONIC Q option
   //

   /* Calculate velocity gradients */
   CalcMonotonicQGradientsForElems(useCPU) ;

   /* Transfer veloctiy gradients in the first order elements */
   /* problem->commElements->Transfer(CommElements::monoQ) ; */
   CalcMonotonicQForElems(useCPU) ;

   /* Don't allow excessive artificial viscosity */
   /*
   if (numElem != 0) {
      Index_t idx = -1; 
      for (Index_t i=0; i<numElem; ++i) {
         if ( mesh.q(i) > qstop ) {
            idx = i ;
            break ;
         }
      }

      if(idx >= 0) {
         exit(QStopError) ;
      }
    }
   */  
}


__global__
void CalcPressureForElems_kernel(Real_t* p_new, Real_t* bvc,
                                 Real_t* pbvc, Real_t* e_old,
                                 Real_t* compression, Real_t *vnewc,
                                 Real_t pmin,
                                 Real_t p_cut, Real_t eosvmax,
                                 Index_t length, Real_t c1s)
{
   int i=blockDim.x*blockIdx.x + threadIdx.x;
   if (i<length) {
       
      bvc[i] = c1s * (compression[i] + Real_t(1.));
      pbvc[i] = c1s;

      p_new[i] = bvc[i] * e_old[i] ;

      if    (FABS(p_new[i]) <  p_cut   )
         p_new[i] = Real_t(0.0) ;

      if    ( vnewc[i] >= eosvmax ) /* impossible condition here? */
         p_new[i] = Real_t(0.0) ;

      if    (p_new[i]       <  pmin)
         p_new[i]   = pmin ;
   }
}


static inline
void CalcPressureForElems_gpu(Real_t* p_new, Real_t* bvc,
                              Real_t* pbvc, Real_t* e_old,
                              Real_t* compression, Real_t *vnewc,
                              Real_t pmin,
                              Real_t p_cut, Real_t eosvmax,
                              Index_t length)
{
    Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(length,dimBlock.x),1,1);
    CalcPressureForElems_kernel<<<dimGrid,dimBlock>>>
        (p_new,bvc,pbvc,e_old,compression,vnewc,pmin,p_cut,eosvmax,length,c1s);
    CUDA_DEBUGSYNC;
}


static inline
void CalcPressureForElems_cpu(Real_t* p_new, Real_t* bvc,
                          Real_t* pbvc, Real_t* e_old,
                          Real_t* compression, Real_t *vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          Index_t length)
{
   Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
   for (Index_t i = 0; i < length ; ++i) {
      bvc[i] = c1s * (compression[i] + Real_t(1.));
      pbvc[i] = c1s;
   }

   for (Index_t i = 0 ; i < length ; ++i){
      p_new[i] = bvc[i] * e_old[i] ;

      if    (FABS(p_new[i]) <  p_cut   )
         p_new[i] = Real_t(0.0) ;

      if    ( vnewc[i] >= eosvmax ) /* impossible condition here? */
         p_new[i] = Real_t(0.0) ;

      if    (p_new[i]       <  pmin)
         p_new[i]   = pmin ;
   }
}

__global__
void CalcEnergyForElemsPart1_kernel(
    Index_t length,Real_t emin,
    Real_t *e_old,Real_t *delvc,Real_t *p_old,Real_t *q_old,Real_t *work,
    Real_t *e_new)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {
        e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
            + Real_t(0.5) * work[i];
        
        if (e_new[i]  < emin ) {
            e_new[i] = emin ;
        }
    }
}


__global__
void CalcEnergyForElemsPart2_kernel(
    Index_t length,Real_t rho0,Real_t e_cut,Real_t emin,
    Real_t *compHalfStep,Real_t *delvc,Real_t *pbvc,Real_t *bvc,
    Real_t *pHalfStep,Real_t *ql,Real_t *qq,Real_t *p_old,Real_t *q_old,Real_t *work,
    Real_t *e_new,
    Real_t *q_new
    )
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {

      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]) ;

      if ( delvc[i] > Real_t(0.) ) {
         q_new[i] /* = qq[i] = ql[i] */ = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

         if ( ssc <= Real_t(0.) ) {
            ssc =Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] + Real_t(0.5) * delvc[i]
         * (  Real_t(3.0)*(p_old[i]     + q_old[i])
              - Real_t(4.0)*(pHalfStep[i] + q_new[i])) ;

      e_new[i] += Real_t(0.5) * work[i];

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }
}


__global__
void CalcEnergyForElemsPart3_kernel(
    Index_t length,Real_t rho0,Real_t sixth,Real_t e_cut,Real_t emin,
    Real_t *pbvc,Real_t *vnewc,Real_t *bvc,Real_t *p_new,Real_t *ql,Real_t *qq,
    Real_t *p_old,Real_t *q_old,Real_t *pHalfStep,Real_t *q_new,Real_t *delvc,
    Real_t *e_new)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {
      Real_t q_tilde ;

      if (delvc[i] > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(0.) ) {
            ssc = Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] - (  Real_t(7.0)*(p_old[i]     + q_old[i])
                               - Real_t(8.0)*(pHalfStep[i] + q_new[i])
                               + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }
}


__global__
void CalcEnergyForElemsPart4_kernel(
    Index_t length,Real_t rho0,Real_t q_cut,
    Real_t *delvc,Real_t *pbvc,Real_t *e_new,Real_t *vnewc,Real_t *bvc,
    Real_t *p_new,Real_t *ql,Real_t *qq,
    Real_t *q_new)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {

      if ( delvc[i] <= Real_t(0.) ) {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(0.) ) {
            ssc = Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;

         if (FABS(q_new[i]) < q_cut) q_new[i] = Real_t(0.) ;
      }
   }
}

static inline
void CalcEnergyForElems_gpu(Real_t* p_new, Real_t* e_new, Real_t* q_new,
                            Real_t* bvc, Real_t* pbvc,
                            Real_t* p_old, Real_t* e_old, Real_t* q_old,
                            Real_t* compression, Real_t* compHalfStep,
                            Real_t* vnewc, Real_t* work, Real_t* delvc, Real_t pmin,
                            Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                            Real_t* qq, Real_t* ql,
                            Real_t rho0,
                            Real_t eosvmax,
                            Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
   Real_t *pHalfStep;

   dim3 dimBlock=dim3(BLOCKSIZE,1,1);
   dim3 dimGrid=dim3(PAD_DIV(length,dimBlock.x),1,1);
   
   CUDA( cudaMalloc(&pHalfStep,sizeof(Real_t)*length) );

   CalcEnergyForElemsPart1_kernel<<<dimGrid,dimBlock>>>
       (length,emin,e_old,delvc,p_old,q_old,work,e_new);
   CUDA_DEBUGSYNC;
   
   CalcPressureForElems_gpu(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax, length);

   CalcEnergyForElemsPart2_kernel<<<dimGrid,dimBlock>>>
       (length,rho0,e_cut,emin,
        compHalfStep,delvc,pbvc,bvc,pHalfStep,ql,qq,p_old,q_old,work,
        e_new,
        q_new);
   CUDA_DEBUGSYNC;
   
   CalcPressureForElems_gpu(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

   CalcEnergyForElemsPart3_kernel<<<dimGrid,dimBlock>>>
       (length,rho0,sixth,e_cut,emin,
        pbvc,vnewc,bvc,p_new,ql,qq,
        p_old,q_old,pHalfStep,q_new,delvc,
        e_new);
   CUDA_DEBUGSYNC;
   
   CalcPressureForElems_gpu(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

   CalcEnergyForElemsPart4_kernel<<<dimGrid,dimBlock>>>
       (length,rho0,q_cut,
        delvc,pbvc,e_new,vnewc,bvc,
        p_new,ql,qq,
        q_new);
   CUDA_DEBUGSYNC;
   
   CUDA( cudaFree(pHalfStep) );

   return ;
}

static inline
void CalcEnergyForElems_cpu(Real_t* p_new, Real_t* e_new, Real_t* q_new,
                            Real_t* bvc, Real_t* pbvc,
                            Real_t* p_old, Real_t* e_old, Real_t* q_old,
                            Real_t* compression, Real_t* compHalfStep,
                            Real_t* vnewc, Real_t* work, Real_t* delvc, Real_t pmin,
                            Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                            Real_t* qq, Real_t* ql,
                            Real_t rho0,
                            Real_t eosvmax,
                            Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
   Real_t *pHalfStep = Allocate<Real_t>(length) ;

   for (Index_t i = 0 ; i < length ; ++i) {
      e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
         + Real_t(0.5) * work[i];

      if (e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems_cpu(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax, length);

   for (Index_t i = 0 ; i < length ; ++i) {
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]) ;

      if ( delvc[i] > Real_t(0.) ) {
         q_new[i] /* = qq[i] = ql[i] */ = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

         if ( ssc <= Real_t(0.) ) {
            ssc =Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] + Real_t(0.5) * delvc[i]
         * (  Real_t(3.0)*(p_old[i]     + q_old[i])
              - Real_t(4.0)*(pHalfStep[i] + q_new[i])) ;
   }

   for (Index_t i = 0 ; i < length ; ++i) {

      e_new[i] += Real_t(0.5) * work[i];

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems_cpu(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

   for (Index_t i = 0 ; i < length ; ++i){
      Real_t q_tilde ;

      if (delvc[i] > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(0.) ) {
            ssc = Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] - (  Real_t(7.0)*(p_old[i]     + q_old[i])
                               - Real_t(8.0)*(pHalfStep[i] + q_new[i])
                               + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems_cpu(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

   for (Index_t i = 0 ; i < length ; ++i){

      if ( delvc[i] <= Real_t(0.) ) {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(0.) ) {
            ssc = Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;

         if (FABS(q_new[i]) < q_cut) q_new[i] = Real_t(0.) ;
      }
   }

   Release(&pHalfStep) ;

   return ;
}


__global__
void CalcSoundSpeedForElems_kernel(Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3, Index_t nz,Index_t *matElemlist,
                            Real_t *ss)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<nz) {
    
      Index_t iz = matElemlist[i];
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[i] * vnewc[i] *
                 bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= Real_t(1.111111e-36)) {
         ssTmp = Real_t(1.111111e-36);
      }
      ss[iz] = SQRT(ssTmp);
   }
}


static inline
void CalcSoundSpeedForElems_gpu(Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3, Index_t nz)
{
   dim3 dimBlock=dim3(BLOCKSIZE,1,1);
   dim3 dimGrid=dim3(PAD_DIV(nz,dimBlock.x),1,1);
   CalcSoundSpeedForElems_kernel<<<dimGrid,dimBlock>>>
       (vnewc,rho0,enewc,pnewc,pbvc,bvc,ss4o3,nz,meshGPU.m_matElemlist,meshGPU.m_ss);
   CUDA_DEBUGSYNC;
    
}

static inline
void CalcSoundSpeedForElems_cpu(Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3, Index_t nz)
{
   for (Index_t i = 0; i < nz ; ++i) {
      Index_t iz = mesh.matElemlist(i);
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[i] * vnewc[i] *
                 bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= Real_t(1.111111e-36)) {
         ssTmp = Real_t(1.111111e-36);
      }
      mesh.ss(iz) = SQRT(ssTmp);
   }
}


__global__
void EvalEOSForElemsPart1_kernel(
    Index_t length,Real_t eosvmin,Real_t eosvmax,
    Index_t *matElemlist,
    Real_t *e,Real_t *delv,Real_t *p,Real_t *q,Real_t *qq,Real_t *ql,
    Real_t *vnewc,
    Real_t *e_old,Real_t *delvc,Real_t *p_old,Real_t *q_old,
    Real_t *compression,Real_t *compHalfStep,
    Real_t *qq_old,Real_t *ql_old,Real_t *work)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {
        Index_t zidx = matElemlist[i];
        e_old[i] = e[zidx];
        delvc[i] = delv[zidx];
        p_old[i] = p[zidx];
        q_old[i] = q[zidx];

        Real_t vchalf ;
        compression[i] = Real_t(1.) / vnewc[i] - Real_t(1.);
        vchalf = vnewc[i] - delvc[i] * Real_t(.5);
        compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);

        if ( eosvmin != Real_t(0.) ) {
            if (vnewc[i] <= eosvmin) { /* impossible due to calling func? */
                compHalfStep[i] = compression[i] ;
            }
        }
        if ( eosvmax != Real_t(0.) ) {
            if (vnewc[i] >= eosvmax) { /* impossible due to calling func? */
                p_old[i]        = Real_t(0.) ;
                compression[i]  = Real_t(0.) ;
                compHalfStep[i] = Real_t(0.) ;
            }
        }

        qq_old[i] = qq[zidx] ;
        ql_old[i] = ql[zidx] ;
        work[i] = Real_t(0.) ; 
    }
}


__global__
void EvalEOSForElemsPart2_kernel(
    Index_t length,
    Index_t *matElemlist,Real_t *p_new,Real_t *e_new,Real_t *q_new,
    Real_t *p,Real_t *e,Real_t *q)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {
        Index_t zidx = matElemlist[i] ;
        p[zidx] = p_new[i];
        e[zidx] = e_new[i];
        q[zidx] = q_new[i];
    }
}


static inline
void EvalEOSForElems_gpu(Real_t *vnewc, Index_t length)
{
   Real_t  e_cut = mesh.e_cut();
   Real_t  p_cut = mesh.p_cut();
   Real_t  ss4o3 = mesh.ss4o3();
   Real_t  q_cut = mesh.q_cut();

   Real_t eosvmax = mesh.eosvmax() ;
   Real_t eosvmin = mesh.eosvmin() ;
   Real_t pmin    = mesh.pmin() ;
   Real_t emin    = mesh.emin() ;
   Real_t rho0    = mesh.refdens() ;

   Real_t *e_old,*delvc,*p_old,*q_old;
   Real_t *compression,*compHalfStep;
   Real_t *qq,*ql,*work,*p_new,*e_new,*q_new,*bvc,*pbvc;

   CUDA( cudaMalloc(&e_old,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&delvc,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&p_old,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&q_old,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&compression,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&compHalfStep,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&qq,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&ql,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&work,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&p_new,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&e_new,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&q_new,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&bvc,sizeof(Real_t)*length) );
   CUDA( cudaMalloc(&pbvc,sizeof(Real_t)*length) );

   dim3 dimBlock=dim3(BLOCKSIZE,1,1);
   dim3 dimGrid=dim3(PAD_DIV(length,dimBlock.x),1,1);

   EvalEOSForElemsPart1_kernel<<<dimGrid,dimBlock>>>
       (length,eosvmin,eosvmax,
        meshGPU.m_matElemlist,
        meshGPU.m_e,meshGPU.m_delv,meshGPU.m_p,meshGPU.m_q,meshGPU.m_qq,meshGPU.m_ql,
        vnewc,
        e_old,delvc,p_old,q_old,
        compression,compHalfStep,qq,ql,work);
   CUDA_DEBUGSYNC;

   CalcEnergyForElems_gpu(p_new, e_new, q_new, bvc, pbvc,
                 p_old, e_old,  q_old, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 qq, ql, rho0, eosvmax, length);


   EvalEOSForElemsPart2_kernel<<<dimGrid,dimBlock>>>
       (length,
        meshGPU.m_matElemlist,p_new,e_new,q_new,
        meshGPU.m_p,meshGPU.m_e,meshGPU.m_q);
   CUDA_DEBUGSYNC;

   CalcSoundSpeedForElems_gpu(vnewc, rho0, e_new, p_new,
             pbvc, bvc, ss4o3, length) ;

   CUDA( cudaFree(pbvc) );
   CUDA( cudaFree(bvc) );
   CUDA( cudaFree(q_new) );
   CUDA( cudaFree(e_new) );
   CUDA( cudaFree(p_new) );
   CUDA( cudaFree(work) );
   CUDA( cudaFree(ql) );
   CUDA( cudaFree(qq) );
   CUDA( cudaFree(compHalfStep) );
   CUDA( cudaFree(compression) );
   CUDA( cudaFree(q_old) );
   CUDA( cudaFree(p_old) );
   CUDA( cudaFree(delvc) );
   CUDA( cudaFree(e_old) );
}


static inline
void EvalEOSForElems_cpu(Real_t *vnewc, Index_t length)
{
   Real_t  e_cut = mesh.e_cut();
   Real_t  p_cut = mesh.p_cut();
   Real_t  ss4o3 = mesh.ss4o3();
   Real_t  q_cut = mesh.q_cut();

   Real_t eosvmax = mesh.eosvmax() ;
   Real_t eosvmin = mesh.eosvmin() ;
   Real_t pmin    = mesh.pmin() ;
   Real_t emin    = mesh.emin() ;
   Real_t rho0    = mesh.refdens() ;

   Real_t *e_old = Allocate<Real_t>(length) ;
   Real_t *delvc = Allocate<Real_t>(length) ;
   Real_t *p_old = Allocate<Real_t>(length) ;
   Real_t *q_old = Allocate<Real_t>(length) ;
   Real_t *compression = Allocate<Real_t>(length) ;
   Real_t *compHalfStep = Allocate<Real_t>(length) ;
   Real_t *qq = Allocate<Real_t>(length) ;
   Real_t *ql = Allocate<Real_t>(length) ;
   Real_t *work = Allocate<Real_t>(length) ;
   Real_t *p_new = Allocate<Real_t>(length) ;
   Real_t *e_new = Allocate<Real_t>(length) ;
   Real_t *q_new = Allocate<Real_t>(length) ;
   Real_t *bvc = Allocate<Real_t>(length) ;
   Real_t *pbvc = Allocate<Real_t>(length) ;

   /* compress data, minimal set */
   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      e_old[i] = mesh.e(zidx) ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      delvc[i] = mesh.delv(zidx) ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      p_old[i] = mesh.p(zidx) ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      q_old[i] = mesh.q(zidx) ;
   }

   for (Index_t i = 0; i < length ; ++i) {
      Real_t vchalf ;
      compression[i] = Real_t(1.) / vnewc[i] - Real_t(1.);
      vchalf = vnewc[i] - delvc[i] * Real_t(.5);
      compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);
   }

   /* Check for v > eosvmax or v < eosvmin */
   if ( eosvmin != Real_t(0.) ) {
      for(Index_t i=0 ; i<length ; ++i) {
         if (vnewc[i] <= eosvmin) { /* impossible due to calling func? */
            compHalfStep[i] = compression[i] ;
         }
      }
   }
   if ( eosvmax != Real_t(0.) ) {
      for(Index_t i=0 ; i<length ; ++i) {
         if (vnewc[i] >= eosvmax) { /* impossible due to calling func? */
            p_old[i]        = Real_t(0.) ;
            compression[i]  = Real_t(0.) ;
            compHalfStep[i] = Real_t(0.) ;
         }
      }
   }

   for (Index_t i = 0 ; i < length ; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      qq[i] = mesh.qq(zidx) ;
      ql[i] = mesh.ql(zidx) ;
      work[i] = Real_t(0.) ; 
   }

   CalcEnergyForElems_cpu(p_new, e_new, q_new, bvc, pbvc,
                 p_old, e_old,  q_old, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 qq, ql, rho0, eosvmax, length);


   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      mesh.p(zidx) = p_new[i] ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      mesh.e(zidx) = e_new[i] ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = mesh.matElemlist(i) ;
      mesh.q(zidx) = q_new[i] ;
   }

   CalcSoundSpeedForElems_cpu(vnewc, rho0, e_new, p_new,
             pbvc, bvc, ss4o3, length) ;

   Release(&pbvc) ;
   Release(&bvc) ;
   Release(&q_new) ;
   Release(&e_new) ;
   Release(&p_new) ;
   Release(&work) ;
   Release(&ql) ;
   Release(&qq) ;
   Release(&compHalfStep) ;
   Release(&compression) ;
   Release(&q_old) ;
   Release(&p_old) ;
   Release(&delvc) ;
   Release(&e_old) ;
}


__global__
void ApplyMaterialPropertiesForElemsPart1_kernel(
    Index_t length,Real_t eosvmin,Real_t eosvmax,
    Index_t *matElemlist,Real_t *vnew,
    Real_t *vnewc)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<length) {
        Index_t zn = matElemlist[i] ;
        vnewc[i] = vnew[zn] ;

        if (eosvmin != Real_t(0.)) {
            if (vnewc[i] < eosvmin)
                vnewc[i] = eosvmin ;
        }

        if (eosvmax != Real_t(0.)) {
            if (vnewc[i] > eosvmax)
                vnewc[i] = eosvmax ;
        }
    }
}


static inline
void ApplyMaterialPropertiesForElems_gpu()
{
  Index_t length = mesh.numElem() ;

  if (length != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = mesh.eosvmin() ;
    Real_t eosvmax = mesh.eosvmax() ;
    Real_t *vnewc;

    CUDA( cudaMalloc(&vnewc,sizeof(Real_t)*length) );

    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(length,dimBlock.x),1,1);
    ApplyMaterialPropertiesForElemsPart1_kernel<<<dimGrid,dimBlock>>>
        (length,eosvmin,eosvmax,
         meshGPU.m_matElemlist,meshGPU.m_vnew,
         vnewc);
    CUDA_DEBUGSYNC;
    
    /*
    for (Index_t i=0; i<length; ++i) {
       Index_t zn = mesh.matElemlist(i) ;
       Real_t vc = mesh.v(zn) ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin)
             vc = eosvmin ;
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax)
             vc = eosvmax ;
       }
       if (vc <= 0.) {
          exit(VolumeError) ;
       }
    }
    */
    
    EvalEOSForElems_gpu(vnewc, length);

    CUDA( cudaFree(vnewc) );
  }
}

static inline
void ApplyMaterialPropertiesForElems_cpu()
{
  Index_t length = mesh.numElem() ;

  if (length != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = mesh.eosvmin() ;
    Real_t eosvmax = mesh.eosvmax() ;
    Real_t *vnewc = Allocate<Real_t>(length) ;

    for (Index_t i=0 ; i<length ; ++i) {
       Index_t zn = mesh.matElemlist(i) ;
       vnewc[i] = mesh.vnew(zn) ;
    }

    if (eosvmin != Real_t(0.)) {
       for(Index_t i=0 ; i<length ; ++i) {
          if (vnewc[i] < eosvmin)
             vnewc[i] = eosvmin ;
       }
    }

    if (eosvmax != Real_t(0.)) {
       for(Index_t i=0 ; i<length ; ++i) {
          if (vnewc[i] > eosvmax)
             vnewc[i] = eosvmax ;
       }
    }

    for (Index_t i=0; i<length; ++i) {
       Index_t zn = mesh.matElemlist(i) ;
       Real_t vc = mesh.v(zn) ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin)
             vc = eosvmin ;
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax)
             vc = eosvmax ;
       }
       if (vc <= 0.) {
          exit(VolumeError) ;
       }
    }

    EvalEOSForElems_cpu(vnewc, length);

    Release(&vnewc) ;

  }
}

static inline
void ApplyMaterialPropertiesForElems(int useCPU)
{
    if (useCPU) {
        FC(matElemlist); FC(vnew); FC(v); FC(e); FC(delv); FC(p); FC(q); FC(qq); FC(ql);
        ApplyMaterialPropertiesForElems_cpu();
        SG(p); SG(e); SG(q); SG(ss);
    }
    else {
        FG(matElemlist); FG(vnew); FG(v); FG(e); FG(delv); FG(p); FG(q); FG(qq); FG(ql);
        ApplyMaterialPropertiesForElems_gpu();
        SC(p); SC(e); SC(q); SC(ss);
    }
}

__global__
void UpdateVolumesForElems_kernel(Index_t numElem,Real_t v_cut,
                                  Real_t *vnew,
                                  Real_t *v)
{
    int i=blockDim.x*blockIdx.x + threadIdx.x;
    if (i<numElem) {
         Real_t tmpV ;
         tmpV = vnew[i] ;

         if ( FABS(tmpV - Real_t(1.0)) < v_cut )
            tmpV = Real_t(1.0) ;
         v[i] = tmpV ;
    }
}


static inline
void UpdateVolumesForElems_gpu()
{
   Index_t numElem = mesh.numElem();
   if (numElem != 0) {
      Real_t v_cut = mesh.v_cut();
      dim3 dimBlock=dim3(BLOCKSIZE,1,1);
      dim3 dimGrid=dim3(PAD_DIV(numElem,dimBlock.x),1,1);
      UpdateVolumesForElems_kernel<<<dimGrid,dimBlock>>>
          (numElem,v_cut,meshGPU.m_vnew,meshGPU.m_v);
   }
}


static inline
void UpdateVolumesForElems_cpu()
{
   Index_t numElem = mesh.numElem();
   if (numElem != 0) {
      Real_t v_cut = mesh.v_cut();

      for(Index_t i=0 ; i<numElem ; ++i) {
         Real_t tmpV ;
         tmpV = mesh.vnew(i) ;

         if ( FABS(tmpV - Real_t(1.0)) < v_cut )
            tmpV = Real_t(1.0) ;
         mesh.v(i) = tmpV ;
      }
   }

   return ;
}

static inline
void UpdateVolumesForElems(int useCPU)
{
    if (useCPU) {
        FC(vnew);
        UpdateVolumesForElems_cpu();
        SG(v);
    }
    else {
        FG(vnew);
        UpdateVolumesForElems_gpu();
        SC(v);
    }
}


static inline
void LagrangeElements(int useCPU)
{
  const Real_t deltatime = mesh.deltatime() ;

  CalcLagrangeElements(deltatime, useCPU) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(useCPU) ;

  ApplyMaterialPropertiesForElems(useCPU) ;

  UpdateVolumesForElems(useCPU) ;
}


__global__
void CalcCourantConstraintForElems_kernel(
    Index_t length,Real_t qqc2,
    Index_t *matElemlist,Real_t *ss,Real_t *vdov,Real_t *arealg,
    Real_t *mindtcourant)
{
    __shared__ Real_t minArray[BLOCKSIZE];

    int i=blockDim.x*blockIdx.x + threadIdx.x;
    
    Real_t dtcourant = Real_t(1.0e+20) ;
    if (i<length) {
        Index_t indx = matElemlist[i] ;
        Real_t dtf = ss[indx] * ss[indx] ;
        if ( vdov[indx] < Real_t(0.) ) {
            dtf = dtf
                + qqc2 * arealg[indx] * arealg[indx]
                * vdov[indx] * vdov[indx] ;
        }
        dtf = SQRT(dtf) ;
        dtf = arealg[indx] / dtf ;

        /* determine minimum timestep with its corresponding elem */
        if (vdov[indx] != Real_t(0.)) {
            if ( dtf < dtcourant ) {
                dtcourant = dtf ;
            }
        }
    }
    minArray[threadIdx.x]=dtcourant;
    reduceMin<Real_t,BLOCKSIZE>(minArray,threadIdx.x);
    if (threadIdx.x==0)
        mindtcourant[blockIdx.x]=minArray[0];
}


static inline
void CalcCourantConstraintForElems_gpu()
{
    Real_t qqc = mesh.qqc();
    Real_t qqc2 = Real_t(64.0) * qqc * qqc ;
    Index_t length = mesh.numElem() ;

    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(length,dimBlock.x),1,1);

    Real_t *dev_mindtcourant;
    CUDA( cudaMalloc(&dev_mindtcourant,sizeof(Real_t)*dimGrid.x) );

    CalcCourantConstraintForElems_kernel<<<dimGrid,dimBlock>>>
        (length,qqc2,
         meshGPU.m_matElemlist,meshGPU.m_ss,meshGPU.m_vdov,meshGPU.m_arealg,
         dev_mindtcourant);
    CUDA_DEBUGSYNC;

    Real_t *mindtcourant = (Real_t*)malloc(sizeof(Real_t)*dimGrid.x);
    CUDA( cudaMemcpy(mindtcourant,dev_mindtcourant,sizeof(Real_t)*dimGrid.x,cudaMemcpyDeviceToHost) );
    CUDA( cudaFree(dev_mindtcourant) );

    // finish the MIN computation over the thread blocks
    Real_t dtcourant;
    dtcourant=mindtcourant[0];
    for (int i=1; i<dimGrid.x; i++) {
        MINEQ(dtcourant,mindtcourant[i]);
    }
    free(mindtcourant);

    if (dtcourant < Real_t(1.0e+20))
        mesh.dtcourant() = dtcourant ;
}

static inline
void CalcCourantConstraintForElems_cpu()
{
   Real_t dtcourant = Real_t(1.0e+20) ;
   Index_t   courant_elem = -1 ;
   Real_t      qqc = mesh.qqc() ;
   Index_t length = mesh.numElem() ;

   Real_t  qqc2 = Real_t(64.0) * qqc * qqc ;

   for (Index_t i = 0 ; i < length ; ++i) {
      Index_t indx = mesh.matElemlist(i) ;

      Real_t dtf = mesh.ss(indx) * mesh.ss(indx) ;

      if ( mesh.vdov(indx) < Real_t(0.) ) {

         dtf = dtf
            + qqc2 * mesh.arealg(indx) * mesh.arealg(indx)
            * mesh.vdov(indx) * mesh.vdov(indx) ;
      }

      dtf = SQRT(dtf) ;

      dtf = mesh.arealg(indx) / dtf ;

   /* determine minimum timestep with its corresponding elem */
      if (mesh.vdov(indx) != Real_t(0.)) {
         if ( dtf < dtcourant ) {
            dtcourant = dtf ;
            courant_elem = indx ;
         }
      }
   }

   /* Don't try to register a time constraint if none of the elements
    * were active */
   if (courant_elem != -1) {
      mesh.dtcourant() = dtcourant ;
   }

   return ;
}


static inline
void CalcCourantConstraintForElems(int useCPU)
{
    if (useCPU) {
        FC(matElemlist); FC(ss); FC(vdov); FC(arealg);
        CalcCourantConstraintForElems_cpu();
    }
    else {
        FG(matElemlist); FG(ss); FG(vdov); FG(arealg);
        CalcCourantConstraintForElems_gpu();
    }
}


__global__
void CalcHydroConstraintForElems_kernel(
    Index_t length,Real_t dvovmax,
    Index_t *matElemlist,Real_t *vdov,
    Real_t *mindthydro)
{
    __shared__ Real_t minArray[BLOCKSIZE];

    int i=blockDim.x*blockIdx.x + threadIdx.x;

    Real_t dthydro = Real_t(1.0e+20) ;
    if (i<length) {
      Index_t indx = matElemlist[i] ;
      if (vdov[indx] != Real_t(0.)) {
         Real_t dtdvov = dvovmax / (FABS(vdov[indx])+Real_t(1.e-20)) ;
         if ( dthydro > dtdvov ) {
            dthydro = dtdvov ;
         }
      }
    }
    minArray[threadIdx.x]=dthydro;
    reduceMin<Real_t,BLOCKSIZE>(minArray,threadIdx.x);
    if (threadIdx.x==0)
        mindthydro[blockIdx.x]=minArray[0];
}


static inline
void CalcHydroConstraintForElems_gpu()
{
    Real_t dvovmax = mesh.dvovmax() ;
    Index_t length = mesh.numElem() ;

    dim3 dimBlock=dim3(BLOCKSIZE,1,1);
    dim3 dimGrid=dim3(PAD_DIV(length,dimBlock.x),1,1);

    Real_t *dev_mindthydro;
    CUDA( cudaMalloc(&dev_mindthydro,sizeof(Real_t)*dimGrid.x) );

    CalcHydroConstraintForElems_kernel<<<dimGrid,dimBlock>>>
        (length,dvovmax,
         meshGPU.m_matElemlist,meshGPU.m_vdov,
         dev_mindthydro);
    CUDA_DEBUGSYNC;

    Real_t *mindthydro = (Real_t*)malloc(sizeof(Real_t)*dimGrid.x);
    CUDA( cudaMemcpy(mindthydro,dev_mindthydro,sizeof(Real_t)*dimGrid.x,cudaMemcpyDeviceToHost) );
    CUDA( cudaFree(dev_mindthydro) );

    // finish the MIN computation over the thread blocks
    Real_t dthydro=mindthydro[0];
    for (int i=1; i<dimGrid.x; i++) {
        MINEQ(dthydro,mindthydro[i]);
    }
    free(mindthydro);
    
    if (dthydro < Real_t(1.0e+20))
        mesh.dthydro() = dthydro ;
}

static inline
void CalcHydroConstraintForElems_cpu()
{
   Real_t dthydro = Real_t(1.0e+20) ;
   Index_t hydro_elem = -1 ;
   Real_t dvovmax = mesh.dvovmax() ;
   Index_t length = mesh.numElem() ;

   for (Index_t i = 0 ; i < length ; ++i) {
      Index_t indx = mesh.matElemlist(i) ;

      if (mesh.vdov(indx) != Real_t(0.)) {
         Real_t dtdvov = dvovmax / (FABS(mesh.vdov(indx))+Real_t(1.e-20)) ;
         if ( dthydro > dtdvov ) {
            dthydro = dtdvov ;
            hydro_elem = indx ;
         }
      }
   }

   if (hydro_elem != -1) {
      mesh.dthydro() = dthydro ;
   }

   return ;
}


static inline
void CalcHydroConstraintForElems(int useCPU)
{
    if (useCPU) {
        FC(matElemlist); FC(vdov);
        CalcHydroConstraintForElems_cpu();
    }
    else {
        FG(matElemlist); FG(vdov);
        CalcHydroConstraintForElems_gpu();
    }
}



static inline
void CalcTimeConstraintsForElems(int useCPU) {
   /* evaluate time constraint */
   CalcCourantConstraintForElems(useCPU) ;

   /* check hydro constraint */
   CalcHydroConstraintForElems(useCPU) ;
}

static inline
void LagrangeLeapFrog(int useCPU)
{
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */

   LagrangeNodal(useCPU);

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(useCPU);

   CalcTimeConstraintsForElems(useCPU);

   // LagrangeRelease() ;  Creation/destruction of temps may be important to capture 
}

int main(int argc, char *argv[])
{
   Index_t edgeElems = 45 ;
   Index_t edgeNodes = edgeElems+1 ;
   // Real_t ds = Real_t(1.125)/Real_t(edgeElems) ; /* may accumulate roundoff */
   Real_t tx, ty, tz ;
   Index_t nidx, zidx ;
   Index_t meshElems ;

   /* get run options to measure various metrics */

   /* ... */
   
   cuda_init();
   
   /****************************/
   /*   Initialize Sedov Mesh  */
   /****************************/

   /* construct a uniform box for this processor */

   mesh.sizeX()   = edgeElems ;
   mesh.sizeY()   = edgeElems ;
   mesh.sizeZ()   = edgeElems ;
   mesh.numElem() = edgeElems*edgeElems*edgeElems ;
   mesh.numNode() = edgeNodes*edgeNodes*edgeNodes ;

   meshElems = mesh.numElem() ;


   /* allocate field memory */

   mesh.AllocateElemPersistent(mesh.numElem()) ;
   mesh.AllocateElemTemporary (mesh.numElem()) ;

   mesh.AllocateNodalPersistent(mesh.numNode()) ;
   mesh.AllocateNodesets(edgeNodes*edgeNodes) ;


   /* initialize nodal coordinates */

   nidx = 0 ;
   tz  = Real_t(0.) ;
   for (Index_t plane=0; plane<edgeNodes; ++plane) {
      ty = Real_t(0.) ;
      for (Index_t row=0; row<edgeNodes; ++row) {
         tx = Real_t(0.) ;
         for (Index_t col=0; col<edgeNodes; ++col) {
            mesh.x(nidx) = tx ;
            mesh.y(nidx) = ty ;
            mesh.z(nidx) = tz ;
            ++nidx ;
            // tx += ds ; /* may accumulate roundoff... */
            tx = Real_t(1.125)*Real_t(col+1)/Real_t(edgeElems) ;
         }
         // ty += ds ;  /* may accumulate roundoff... */
         ty = Real_t(1.125)*Real_t(row+1)/Real_t(edgeElems) ;
      }
      // tz += ds ;  /* may accumulate roundoff... */
      tz = Real_t(1.125)*Real_t(plane+1)/Real_t(edgeElems) ;
   }


   /* embed hexehedral elements in nodal point lattice */

   nidx = 0 ;
   zidx = 0 ;
   for (Index_t plane=0; plane<edgeElems; ++plane) {
      for (Index_t row=0; row<edgeElems; ++row) {
         for (Index_t col=0; col<edgeElems; ++col) {
            mesh.nodelist(zidx,0) = nidx                                       ;
            mesh.nodelist(zidx,1) = nidx                                   + 1 ;
            mesh.nodelist(zidx,2) = nidx                       + edgeNodes + 1 ;
            mesh.nodelist(zidx,3) = nidx                       + edgeNodes     ;
            mesh.nodelist(zidx,4) = nidx + edgeNodes*edgeNodes                 ;
            mesh.nodelist(zidx,5) = nidx + edgeNodes*edgeNodes             + 1 ;
            mesh.nodelist(zidx,6) = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
            mesh.nodelist(zidx,7) = nidx + edgeNodes*edgeNodes + edgeNodes     ;
            ++zidx ;
            ++nidx ;
         }
         ++nidx ;
      }
      nidx += edgeNodes ;
   }

   /* Create a material IndexSet (entire mesh same material for now) */
   for (Index_t i=0; i<meshElems; ++i) {
      mesh.matElemlist(i) = i ;
   }
   
   /* initialize material parameters */
   mesh.dtfixed() = Real_t(-1.0e-7) ;
   mesh.deltatime() = Real_t(1.0e-7) ;
   mesh.deltatimemultlb() = Real_t(1.1) ;
   mesh.deltatimemultub() = Real_t(1.2) ;
   if (argc == 2) {
       mesh.stoptime()  = Real_t(atof(argv[1])) ;
   } else {
       mesh.stoptime()  = Real_t(1.0e-2) ;
   }
   mesh.dtcourant() = Real_t(1.0e+20) ;
   mesh.dthydro()   = Real_t(1.0e+20) ;
   mesh.dtmax()     = Real_t(1.0e-2) ;
   mesh.time()    = Real_t(0.) ;
   mesh.cycle()   = 0 ;

   mesh.e_cut() = Real_t(1.0e-7) ;
   mesh.p_cut() = Real_t(1.0e-7) ;
   mesh.q_cut() = Real_t(1.0e-7) ;
   mesh.u_cut() = Real_t(1.0e-7) ;
   mesh.v_cut() = Real_t(1.0e-10) ;

   mesh.hgcoef()      = Real_t(3.0) ;
   mesh.ss4o3()       = Real_t(4.0)/Real_t(3.0) ;

   mesh.qstop()              =  Real_t(1.0e+12) ;
   mesh.monoq_max_slope()    =  Real_t(1.0) ;
   mesh.monoq_limiter_mult() =  Real_t(2.0) ;
   mesh.qlc_monoq()          = Real_t(0.5) ;
   mesh.qqc_monoq()          = Real_t(2.0)/Real_t(3.0) ;
   mesh.qqc()                = Real_t(2.0) ;

   mesh.pmin() =  Real_t(0.) ;
   mesh.emin() = Real_t(-1.0e+15) ;

   mesh.dvovmax() =  Real_t(0.1) ;

   mesh.eosvmax() =  Real_t(1.0e+9) ;
   mesh.eosvmin() =  Real_t(1.0e-9) ;

   mesh.refdens() =  Real_t(1.0) ;

   /* initialize field data */
   for (Index_t i=0; i<meshElems; ++i) {
      Real_t x_local[8], y_local[8], z_local[8] ;
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
        Index_t gnode = mesh.nodelist(i,lnode);
        x_local[lnode] = mesh.x(gnode);
        y_local[lnode] = mesh.y(gnode);
        z_local[lnode] = mesh.z(gnode);
      }

      // volume calculations
      Real_t volume = CalcElemVolume(x_local, y_local, z_local );
      mesh.volo(i) = volume ;
      mesh.elemMass(i) = volume ;
      for (Index_t j=0; j<8; ++j) {
	 Index_t idx = mesh.nodelist(i,j);
         mesh.nodalMass(idx) += volume / Real_t(8.0) ;
      }
   }

   /* deposit energy */
   mesh.e(0) = Real_t(3.948746e+7) ;

   /* set up symmetry nodesets */
   nidx = 0 ;
   for (Index_t i=0; i<edgeNodes; ++i) {
      Index_t planeInc = i*edgeNodes*edgeNodes ;
      Index_t rowInc   = i*edgeNodes ;
      for (Index_t j=0; j<edgeNodes; ++j) {
         mesh.symmX(nidx) = planeInc + j*edgeNodes ;
         mesh.symmY(nidx) = planeInc + j ;
         mesh.symmZ(nidx) = rowInc   + j ;
         ++nidx ;
      }
   }

   /* set up elemement connectivity information */
   mesh.lxim(0) = 0 ;
   for (Index_t i=1; i<meshElems; ++i) {
      mesh.lxim(i)   = i-1 ;
      mesh.lxip(i-1) = i ;
   }
   mesh.lxip(meshElems-1) = meshElems-1 ;

   for (Index_t i=0; i<edgeElems; ++i) {
      mesh.letam(i) = i ; 
      mesh.letap(meshElems-edgeElems+i) = meshElems-edgeElems+i ;
   }
   for (Index_t i=edgeElems; i<meshElems; ++i) {
      mesh.letam(i) = i-edgeElems ;
      mesh.letap(i-edgeElems) = i ;
   }

   for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
      mesh.lzetam(i) = i ;
      mesh.lzetap(meshElems-edgeElems*edgeElems+i) = meshElems-edgeElems*edgeElems+i ;
   }
   for (Index_t i=edgeElems*edgeElems; i<meshElems; ++i) {
      mesh.lzetam(i) = i - edgeElems*edgeElems ;
      mesh.lzetap(i-edgeElems*edgeElems) = i ;
   }

   /* set up boundary condition information */
   for (Index_t i=0; i<meshElems; ++i) {
      mesh.elemBC(i) = 0 ;  /* clear BCs by default */
   }

   /* faces on "external" boundaries will be */
   /* symmetry plane or free surface BCs */
   for (Index_t i=0; i<edgeElems; ++i) {
      Index_t planeInc = i*edgeElems*edgeElems ;
      Index_t rowInc   = i*edgeElems ;
      for (Index_t j=0; j<edgeElems; ++j) {
         mesh.elemBC(planeInc+j*edgeElems) |= XI_M_SYMM ;
         mesh.elemBC(planeInc+j*edgeElems+edgeElems-1) |= XI_P_FREE ;
         mesh.elemBC(planeInc+j) |= ETA_M_SYMM ;
         mesh.elemBC(planeInc+j+edgeElems*edgeElems-edgeElems) |= ETA_P_FREE ;
         mesh.elemBC(rowInc+j) |= ZETA_M_SYMM ;
         mesh.elemBC(rowInc+j+meshElems-edgeElems*edgeElems) |= ZETA_P_FREE ;
      }
   }

   mesh.AllocateNodeElemIndexes();
   

   
   /* initialize meshGPU */
   meshGPU.init(&mesh);
   meshGPU.freshenGPU();
   
   /* timestep to solution */
   int its=0;
#if 0
   while (its<50) {
#else
   while(mesh.time() < mesh.stoptime() ) {
#endif
      TimeIncrement() ;
      LagrangeLeapFrog(0) ;
      its++;
      /* problem->commNodes->Transfer(CommNodes::syncposvel) ; */
#if LULESH_SHOW_PROGRESS
      printf("time = %e, dt=%e\n",
             double(mesh.time()), double(mesh.deltatime()) ) ;
#endif
   }
   printf("iterations: %d\n",its);

//   FC(x);
//   FILE *fp = fopen("x.asc","wb");
//   for (Index_t i=0; i<mesh.numElem(); i++)
//       fprintf(fp,"%.6f\n",mesh.x(i));
//   fclose(fp);

   return 0 ;
}

