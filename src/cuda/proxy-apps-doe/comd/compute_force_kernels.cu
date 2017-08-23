/* Chebyshev coefficients opt */
//#define USE_CHEBY

#include "compute_force_common.h"
#include "potentials.h"

#include "lj_thread_atom.h"

#include "eam_thread_atom.h"
#include "eam_cta_box.h"
#include "eam_cta_box_agg.h"

// ecx-specific implementations
#include "eam_thread_atom_warp_sync.h"

// experimental implementations
#include <eam_warp_atom_smem.h>
#include <eam_warp_atom_regs.h>
#include <eam_block_box_fixed.h>
#include <eam_block_box_reduce.h>

#include <string>
#include <stdio.h>

/* EAM potential implemented in 3 steps */ 

template<int step>
void eam_force(const char *method, sim_t sim_D)
{
  std::string m = method;

  if (m == "thread_atom")
  {
#ifndef ECX_TARGET
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif
    int grid = (sim_D.total_atoms + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
    int block = THREAD_ATOM_CTA;
    EAM_Force_thread_atom<step><<<grid,block>>>(sim_D);
  }
  else if (m == "thread_atom_warp_sync")
  {
#ifndef ECX_TARGET
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif
    int grid = (sim_D.total_atoms + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
    int block = THREAD_ATOM_CTA;
    EAM_Force_thread_atom_warp_sync<step><<<grid,block>>>(sim_D);
  }
  else if (m == "cta_box")
  {
#ifndef ECX_TARGET
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );
#endif
    int grid = sim_D.n_cells;
    int block = CTA_BOX_CTA;
        
    int smem = 0;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 350
    smem += block * sizeof(real_t);
#endif

    switch (step) {
      case 1: smem += (3 * block + 3 * sim_D.max_atoms + 5 * sim_D.max_atoms) * sizeof(real_t); break;
      case 3: smem += (4 * block + 3 * sim_D.max_atoms + 3 * sim_D.max_atoms) * sizeof(real_t); break;
    }
#ifndef ECX_TARGET
    EAM_Force_cta_box<step><<<grid,block,smem>>>(sim_D);
#else
    EAM_Force_cta_box<step><<<grid,block>>>(sim_D);		// currently cannot handle explicit smem param (always allocates 64KB per CTA?)
#endif
  }
  else if (m == "cta_box_agg")
  {
#ifndef ECX_TARGET
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );
#endif
    int grid = sim_D.n_cells;
    int block = CTA_BOX_CTA;

    int smem = 64 * CTA_BOX_WARPS;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 350
    smem += block * sizeof(real_t);
#endif

    switch (step) {
      case 1: smem += (3 * block + 3 * sim_D.max_atoms + 5 * sim_D.max_atoms) * sizeof(real_t); break;
      case 3: smem += (4 * block + 3 * sim_D.max_atoms + 3 * sim_D.max_atoms) * sizeof(real_t); break;
    }
#ifndef ECX_TARGET
    EAM_Force_cta_box_agg<step><<<grid,block,smem>>>(sim_D);
#else
    EAM_Force_cta_box_agg<step><<<grid,block>>>(sim_D);         // currently cannot handle explicit smem param (always allocates 64KB per CTA?)
#endif
  }
  else {
    printf("ERROR: method %s does not exist!\n", m.c_str());
    printf("Available methods are:\n  thread_atom\n  thread_atom_warp_sync\n  cta_box\n  cta_box_agg\n");
    exit(1);
  }
   
  cudaDeviceSynchronize();
}

extern "C"
void copy_halos(sim_t sim_D)
{
	int grid = sim_D.n_cells;
	int block = sim_D.max_atoms;
	copy_halos_kernel<<<grid, block>>>(sim_D);

	cudaDeviceSynchronize();
}

extern "C"
void eam_force_1(const char *method, sim_t sim_D)
{
	eam_force<1>(method, sim_D);
}

extern "C"
void eam_force_2(sim_t sim_D)
{
	int grid = sim_D.n_cells;
	int block = sim_D.max_atoms;
	EAM_Force_2<<<grid,block>>>(sim_D); 

	cudaDeviceSynchronize();
}

extern "C"
void eam_force_3(const char *method, sim_t sim_D)
{
	eam_force<3>(method, sim_D);
}

/* LJ forces computations */

extern "C"
void lj_force(sim_t sim_D)
{
#ifndef ECX_TARGET
        cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif
        int grid = (sim_D.total_atoms + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
        int block = THREAD_ATOM_CTA;

        LJ_Force_thread_atom<<<grid,block>>>(sim_D);
        cudaDeviceSynchronize();
}



