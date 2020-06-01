/* 
   algorithm description:
     using 1 block per box, threads are processing multiple values
     inner atoms are processed one by one in an outer loop
     inside the loop first we mark neighbors that pass cutoff check
     do cta reduction and store result in gmem
*/

#define BLOCK_BOX_REDUCE_WARPS		4	
#define BLOCK_BOX_REDUCE_CTA		(32 * BLOCK_BOX_REDUCE_WARPS)

#ifdef DOUBLE
// 62% occupancy for DP
#define BLOCK_BOX_REDUCE_ACTIVE_CTAS	10
#else
// 100% occupancy for SP
#define BLOCK_BOX_REDUCE_ACTIVE_CTAS	16
#endif

__device__
void cta_reduce(real_t &reg, volatile real_t *smem)
{      
	smem[threadIdx.x] = reg;
      __syncthreads();

     // CTA-wide reduction
      for( int off = BLOCK_BOX_REDUCE_CTA / 2; off >= 32; off = off / 2 ) {
        if( threadIdx.x < off )
          smem[threadIdx.x] += smem[threadIdx.x + off];
          __syncthreads();
      }

  // warp reduce
  if( threadIdx.x < 32 ) {
    smem[threadIdx.x] += smem[threadIdx.x+16];
    smem[threadIdx.x] += smem[threadIdx.x+8];
    smem[threadIdx.x] += smem[threadIdx.x+4];
    smem[threadIdx.x] += smem[threadIdx.x+2];
    smem[threadIdx.x] += smem[threadIdx.x+1];
  }
}

__device__
void warp_reduce_shfl(real_t &reg)
{
  // warp reduction using shuffle op
  for (int i = 16; i > 0; i /= 2) 
    reg += __shfl_xor<BLOCK_BOX_REDUCE_CTA>(reg, i);
}

template<int step>
__global__
__launch_bounds__(BLOCK_BOX_REDUCE_CTA, BLOCK_BOX_REDUCE_ACTIVE_CTAS)
void EAM_Force_bb_reduce(sim_t sim)
{
  // 1 block per box
  int ibox = blockIdx.x;
  int natoms = sim.grid.n_atoms[ibox];
  int nneigh = sim.grid.n_num_neigh[ibox];

  // divide shared memory
  extern __shared__ real_t smem[];

  // neighbor positions
  volatile real_t *sdx = smem + 0;
  volatile real_t *sdy = smem + BLOCK_BOX_REDUCE_CTA;
  volatile real_t *sdz = smem + BLOCK_BOX_REDUCE_CTA * 2;

  // per-warp reduction results
  volatile real_t *sred_fx = smem + BLOCK_BOX_REDUCE_CTA * 3;
  volatile real_t *sred_fy = smem + BLOCK_BOX_REDUCE_CTA * 3 + BLOCK_BOX_REDUCE_WARPS;
  volatile real_t *sred_fz = smem + BLOCK_BOX_REDUCE_CTA * 3 + 2 * BLOCK_BOX_REDUCE_WARPS;
  volatile real_t *sred_e = smem + BLOCK_BOX_REDUCE_CTA * 3 + 3 * BLOCK_BOX_REDUCE_WARPS;
  volatile real_t *sred_rho = smem + BLOCK_BOX_REDUCE_CTA * 3 + 4 * BLOCK_BOX_REDUCE_WARPS;

  // neighbor force for step 3
  volatile real_t *sfi;
  if (step == 3) {
    sfi = smem + BLOCK_BOX_REDUCE_CTA * 3 + 5 * BLOCK_BOX_REDUCE_WARPS;
  }

  // box atoms positions
  volatile real_t *isdx = smem + BLOCK_BOX_REDUCE_CTA * (3 + (step == 3)) + 5 * BLOCK_BOX_REDUCE_WARPS;
  volatile real_t *isdy = isdx + sim.max_atoms;
  volatile real_t *isdz = isdy + sim.max_atoms;

  // box atoms forces
  volatile real_t *sfx = isdz + sim.max_atoms;
  volatile real_t *sfy = sfx + sim.max_atoms;
  volatile real_t *sfz = sfy + sim.max_atoms; 
  volatile real_t *se = sfz + sim.max_atoms; 
  volatile real_t *srho = se + sim.max_atoms;

  if (threadIdx.x < natoms) {
    if (step == 1) { 
      sfx[threadIdx.x] = 0;
      sfy[threadIdx.x] = 0;
      sfz[threadIdx.x] = 0;
      se[threadIdx.x] = 0;
      srho[threadIdx.x] = 0;
    }
    else {
      sfx[threadIdx.x] = sim.f.x[ibox * N_MAX_ATOMS + threadIdx.x];
      sfy[threadIdx.x] = sim.f.y[ibox * N_MAX_ATOMS + threadIdx.x];
      sfz[threadIdx.x] = sim.f.z[ibox * N_MAX_ATOMS + threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x < natoms) {
    int n_id = ibox * N_MAX_ATOMS * N_MAX_NEIGHBORS + sim.grid.itself_start_idx[ibox] + threadIdx.x;
    int jbox = sim.grid.n_neigh_boxes[n_id];
    int j_particle = jbox * N_MAX_ATOMS + sim.grid.n_neigh_atoms[n_id]; // global offset of particle

    isdx[threadIdx.x] = sim.r.x[j_particle];
    isdy[threadIdx.x] = sim.r.y[j_particle];
    isdz[threadIdx.x] = sim.r.z[j_particle];
  }

  int global_base = 0;
  while (global_base < nneigh)
  {
    int global_neighbor = global_base + threadIdx.x;

    // check if last chunk is incomplete
    int tail = 0;
    if (global_base + BLOCK_BOX_REDUCE_CTA > nneigh) {
       tail = nneigh - global_base;
    }

    // load up neighbor particles in smem: 1 thread per neighbor atom
    if (global_neighbor < nneigh) 
    {
      int n_id = ibox * N_MAX_ATOMS * N_MAX_NEIGHBORS + global_neighbor;
      int jbox = sim.grid.n_neigh_boxes[n_id];
      int j_particle = jbox * N_MAX_ATOMS + sim.grid.n_neigh_atoms[n_id]; // global offset of particle

      // compute box center offsets
      real_t dxbox = sim.grid.r_box.x[ibox] - sim.grid.r_box.x[jbox];
      real_t dybox = sim.grid.r_box.y[ibox] - sim.grid.r_box.y[jbox];
      real_t dzbox = sim.grid.r_box.z[ibox] - sim.grid.r_box.z[jbox];

      // correct for periodic 
      if (PERIODIC)
      {
        if (dxbox < -0.5 * sim.grid.bounds[0]) dxbox += sim.grid.bounds[0];
        else if (dxbox > 0.5 * sim.grid.bounds[0] ) dxbox -= sim.grid.bounds[0];
        if (dybox < -0.5 * sim.grid.bounds[1]) dybox += sim.grid.bounds[1];
        else if (dybox > 0.5 * sim.grid.bounds[1] ) dybox -= sim.grid.bounds[1];
        if (dzbox < -0.5 * sim.grid.bounds[2]) dzbox += sim.grid.bounds[2];
        else if (dzbox > 0.5 * sim.grid.bounds[2] ) dzbox -= sim.grid.bounds[2];
      }

      sdx[threadIdx.x] = - sim.r.x[j_particle] + dxbox;
      sdy[threadIdx.x] = - sim.r.y[j_particle] + dybox;
      sdz[threadIdx.x] = - sim.r.z[j_particle] + dzbox;

      if (step == 3) 
        sfi[threadIdx.x] = sim.fi[j_particle];
    }

    __syncthreads();
	
    // do cta reduction for each atom
    for (int iatom = 0; iatom < natoms; iatom++) 
    {
      // atom global index
      int i_offset = ibox * N_MAX_ATOMS;
      int i_particle = i_offset + iatom;

      // square cutoff
      real_t r2cut = sim.eam_pot.cutoff * sim.eam_pot.cutoff;
    
      real_t rhoTmp;
      real_t phiTmp;
      real_t dTmp, dTmp2;

      // accumulate local force in regs
      real_t fx = 0;
      real_t fy = 0;
      real_t fz = 0;
      real_t e = 0;
      real_t rho = 0;

      // last part could be incomplete 
      int neighbor = threadIdx.x;
      if (tail == 0 || neighbor < tail) 
      {
        // load neighbor positions from smem
        real_t dx = isdx[iatom] + sdx[neighbor];
        real_t dy = isdy[iatom] + sdy[neighbor];
        real_t dz = isdz[iatom] + sdz[neighbor];
 
        real_t r2 = dx*dx + dy*dy + dz*dz;
 
        // no divide by zero
        if (r2 <= r2cut && r2 > 0.0) 
        {
          real_t r = sqrt(r2);

          switch (step) {
          case 1:
            eamInterpolateDeriv_opt(r, sim.eam_pot.phi, sim.eam_pot.phi_x0, sim.eam_pot.phi_xn, sim.eam_pot.phi_invDx, phiTmp, dTmp);
            eamInterpolateDeriv_opt(r, sim.eam_pot.rho, sim.eam_pot.rho_x0, sim.eam_pot.rho_xn, sim.eam_pot.rho_invDx, rhoTmp, dTmp2);
          break;
          case 3:
            eamInterpolateDeriv_opt(r, sim.eam_pot.rho, sim.eam_pot.rho_x0, sim.eam_pot.rho_xn, sim.eam_pot.rho_invDx, rhoTmp, dTmp);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
            dTmp *= (__ldg(sim.fi + i_particle) + sfi[neighbor]);
#else 
            dTmp *= (sim.fi[i_particle] + sfi[neighbor]);
#endif
          break;
          }
	 
          dTmp /= r;
    	 
   	  fx = dTmp * dx;
 	  fy = dTmp * dy;
	  fz = dTmp * dz;

	  if (step == 1) {
	    e = phiTmp;
	    rho = rhoTmp;
          }
        }
      }

      warp_reduce_shfl(fx); 
      warp_reduce_shfl(fy); 
      warp_reduce_shfl(fz); 
      if (step == 1) {
        warp_reduce_shfl(e); 
        warp_reduce_shfl(rho); 
      }

      int warp_id = threadIdx.x / 32;
      int lane_id = threadIdx.x % 32;

      if (lane_id == 0) {
        sred_fx[warp_id] = fx;
        sred_fy[warp_id] = fy;
        sred_fz[warp_id] = fz;
        if (step == 1) {
          sred_e[warp_id] = e;
          sred_rho[warp_id] = rho;
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        for (int i = 1; i < BLOCK_BOX_REDUCE_WARPS; i++) {
          sred_fx[0] += sred_fx[i];
          sred_fy[0] += sred_fy[i];
          sred_fz[0] += sred_fz[i];
	  if (step == 1) {
            sred_e[0] += sred_e[i];
            sred_rho[0] += sred_rho[i];
          }
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
	sfx[iatom] += sred_fx[0];
        sfy[iatom] += sred_fy[0];
        sfz[iatom] += sred_fz[0];
	if (step == 1) {
          se[iatom] += sred_e[0];
          srho[iatom] += sred_rho[0];
	}
      }
    }

    __syncthreads();

    global_base += BLOCK_BOX_REDUCE_CTA;
  }

  __syncthreads();
		
  // only 1 thread writes final result
  if (threadIdx.x < natoms) 
  {
    int iatom = threadIdx.x;
    int i_particle = ibox * N_MAX_ATOMS + iatom;

    sim.f.x[i_particle] = sfx[iatom];
    sim.f.y[i_particle] = sfy[iatom];
    sim.f.z[i_particle] = sfz[iatom];

    // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
    if (step == 1) {
      sim.e[i_particle] = se[iatom] * 0.5;
      sim.rho[i_particle] = srho[iatom];	
    }
  }
}
