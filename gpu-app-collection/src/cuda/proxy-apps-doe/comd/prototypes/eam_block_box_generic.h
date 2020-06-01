/* 
   algorithm description:
     using 1 block per box, generic version, i.e. threads are processing multiple values
     1 cta loads up chunk of neighbors and add their contribution to forces of current atoms
     1 atom is assinged to 1 warp accumulating data in registers, 1 warp can process multiple atoms
     reduction using shuffles into register, then add to smem storage
     in the final stage we store result from smem to global mem

   pros:
     don't need to tweak block size, it should work for all data sets
*/

#define BLOCK_BOX_GENERIC_WARPS		4	
#define BLOCK_BOX_GENERIC_CTA		(32 * BLOCK_BOX_GENERIC_WARPS)

#ifdef DOUBLE
// 62% occupancy for DP: optimal sweet spot before spilling too much into local memory
#define BLOCK_BOX_GENERIC_ACTIVE_CTAS	10
#else
// 100% occupancy for SP
#define BLOCK_BOX_GENERIC_ACTIVE_CTAS	16
#endif

template<int step>
__global__
__launch_bounds__(BLOCK_BOX_GENERIC_CTA, BLOCK_BOX_GENERIC_ACTIVE_CTAS)
void EAM_Force_bb_generic(sim_t sim)
{
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  // 1 block per box
  int ibox = blockIdx.x;
  int natoms = sim.grid.n_atoms[ibox];
  int nneigh = sim.grid.n_num_neigh[ibox];

  // divide shared memory
  extern __shared__ real_t smem[];

  // neighbor positions
  volatile real_t *sdx = smem + 0;
  volatile real_t *sdy = smem + BLOCK_BOX_GENERIC_CTA;
  volatile real_t *sdz = smem + BLOCK_BOX_GENERIC_CTA * 2;

  // neighbor force for step 3
  volatile real_t *sfi;
  if (step == 3) {
    sfi = smem + BLOCK_BOX_GENERIC_CTA * 3;
  }

  // box atoms positions
  volatile real_t *isdx = smem + BLOCK_BOX_GENERIC_CTA * (3 + (step == 3));
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
    if (global_base + BLOCK_BOX_GENERIC_CTA > nneigh) {
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
	
    // 1 warp process 1 atom now
    int iatom_base = 0;

    // only process atoms inside current box
    while (iatom_base < natoms) 
    {
      int iatom = iatom_base + warp_id;
      if (iatom < natoms) 
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

        int neighbor = lane_id;
	for (int iters = 0; iters < BLOCK_BOX_GENERIC_WARPS; iters++)
        {
	  if (tail > 0 && neighbor >= tail) break;

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
   	 
	    fx += dTmp * dx;
	    fy += dTmp * dy;
	    fz += dTmp * dz;

	    if (step == 1) {
	      e += phiTmp;
	      rho += rhoTmp;
            }
	  }

	  neighbor += 32;
	}

	// warp reduction using shuffle op
	for (int i = 16; i > 0; i /= 2) 
	{
	  fx += __shfl_xor<BLOCK_BOX_GENERIC_CTA>(fx, i);
	  fy += __shfl_xor<BLOCK_BOX_GENERIC_CTA>(fy, i);
	  fz += __shfl_xor<BLOCK_BOX_GENERIC_CTA>(fz, i);
	  if (step == 1) {
	    e += __shfl_xor<BLOCK_BOX_GENERIC_CTA>(e, i);
	    rho += __shfl_xor<BLOCK_BOX_GENERIC_CTA>(rho, i);
          }
	} 

	// accumulate in smem
	if (lane_id == 0) {
	  sfx[iatom] += fx;
	  sfy[iatom] += fy;
	  sfz[iatom] += fz;
          if (step == 1) {
	    se[iatom] += e;
	    srho[iatom] += rho;
          }
        }
      }

      iatom_base += BLOCK_BOX_GENERIC_WARPS;    
    }

    __syncthreads();

    global_base += BLOCK_BOX_GENERIC_CTA;
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
