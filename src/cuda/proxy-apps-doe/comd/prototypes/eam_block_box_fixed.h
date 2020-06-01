/* 
   algorithm description:
     using 1 block per box, number of threads must be >= max neighbor cells * max atoms
     1 thread loads up exactly 1 neighbor value to smem buffer (some of threads will be idle)
     then 1 warp is assinged to 1 atom accumulating data in registers (some of warps will be idle)

   issues:
     algorithm is not generic, i.e. you have to specify proper number of warps per block
     also it won't work for large number of atoms per box and inefficient for small number of atoms
*/

// this must be greater or equal to max atoms per box
#define BLOCK_BOX_FIXED_WARPS		14	
#define BLOCK_BOX_FIXED_CTA		(32 * BLOCK_BOX_FIXED_WARPS)

template<int step>
__global__
__launch_bounds__(BLOCK_BOX_FIXED_CTA, 3)
void EAM_Force_bb_fixed(sim_t sim)
{
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  // 1 block per box
  int ibox = blockIdx.x;
  int is_idx = sim.grid.itself_start_idx[ibox];

  // load up neighbor particles in smem: 1 thread per neighbor atom
  __shared__ volatile real_t sdx[BLOCK_BOX_FIXED_CTA];
  __shared__ volatile real_t sdy[BLOCK_BOX_FIXED_CTA];
  __shared__ volatile real_t sdz[BLOCK_BOX_FIXED_CTA];
  __shared__ volatile real_t sfi[BLOCK_BOX_FIXED_CTA];

  int neighbor = threadIdx.x;
  if (neighbor < sim.grid.n_num_neigh[ibox])
  {
    int n_id = ibox * N_MAX_ATOMS * N_MAX_NEIGHBORS + neighbor;
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

    sdx[neighbor] = - sim.r.x[j_particle] + dxbox;
    sdy[neighbor] = - sim.r.y[j_particle] + dybox;
    sdz[neighbor] = - sim.r.z[j_particle] + dzbox;

    if (step == 3) 
      sfi[neighbor] = sim.fi[j_particle];
  }

  __syncthreads();
	
  // 1 warp process 1 atom now
  int iatom = warp_id;

  // only process atoms inside current box
  if (iatom >= sim.grid.n_atoms[ibox]) return;

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
	
  neighbor = lane_id;
  while (neighbor < sim.grid.n_num_neigh[ibox]) 
  {
    // load neighbor positions from smem
    real_t dx = - sdx[is_idx + iatom] + sdx[neighbor];
    real_t dy = - sdy[is_idx + iatom] + sdy[neighbor];
    real_t dz = - sdz[is_idx + iatom] + sdz[neighbor];

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
        dTmp *= (sfi[is_idx + iatom] + sfi[neighbor]);
        break;
      }

      fx += dTmp * dx / r;
      fy += dTmp * dy / r;
      fz += dTmp * dz / r;

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
    fx += __shfl_xor<BLOCK_BOX_FIXED_CTA>(fx, i);
    fy += __shfl_xor<BLOCK_BOX_FIXED_CTA>(fy, i);
    fz += __shfl_xor<BLOCK_BOX_FIXED_CTA>(fz, i);
    if (step == 1) {
      e += __shfl_xor<BLOCK_BOX_FIXED_CTA>(e, i);
      rho += __shfl_xor<BLOCK_BOX_FIXED_CTA>(rho, i);
    }
  }
		
  // only 1 thread in a warp writes the result
  if (lane_id == 0) 
  {
    // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
    if (step == 1) {
      sim.f.x[i_particle] = fx;
      sim.f.y[i_particle] = fy;
      sim.f.z[i_particle] = fz;

      sim.e[i_particle] = e * 0.5;
      sim.rho[i_particle] = rho;	
    }
    else {
      sim.f.x[i_particle] += fx;
      sim.f.y[i_particle] += fy;
      sim.f.z[i_particle] += fz;
    }
  }
}
