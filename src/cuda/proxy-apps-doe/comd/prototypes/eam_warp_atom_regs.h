#define WARP_ATOM_REGS_WARPS		2	
#define WARP_ATOM_REGS_CTA		(32 * WARP_ATOM_REGS_WARPS)

template<int step>
__global__
__launch_bounds__(WARP_ATOM_REGS_CTA, 16)
void EAM_Force_wa_regs(sim_t sim)
{
  // 1 warp per 1 atom
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int id = blockIdx.x * WARP_ATOM_REGS_WARPS + warp_id;
  if (id >= sim.total_atoms) return;

  int ibox = sim.grid.n_list_boxes[id];
  int iatom = sim.grid.n_list_atoms[id];

  // only process atoms inside current box
  if (iatom >= sim.grid.n_atoms[ibox]) return;
	
  // atom global index
  int i_offset = ibox * N_MAX_ATOMS;
  int i_particle = i_offset + iatom;

  // square cutoff
  real_t r2cut = sim.eam_pot.cutoff * sim.eam_pot.cutoff;
    
  real_t rhoTmp, phiTmp;
  real_t dTmp, dTmp2;

  // accumulate local force in regs
  real_t fx = 0;
  real_t fy = 0;
  real_t fz = 0;
  real_t e = 0;
  real_t rho = 0;

  if (step == 3) {
    // zero out forces on particles
    fx = sim.f.x[i_particle];
    fy = sim.f.y[i_particle];
    fz = sim.f.z[i_particle];
  }
	
  int neighbor = lane_id;
  while (neighbor < sim.grid.n_num_neigh[ibox]) 
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

    real_t dx = sim.r.x[i_particle] - sim.r.x[j_particle] + dxbox;
    real_t dy = sim.r.y[i_particle] - sim.r.y[j_particle] + dybox;
    real_t dz = sim.r.z[i_particle] - sim.r.z[j_particle] + dzbox;

    real_t r2 = dx*dx + dy*dy + dz*dz;

    // no divide by zero
    if (r2 <= r2cut && r2 > 0.0) 
    {
      real_t r = sqrt(r2);

      switch (step) {
      case 1:
#ifdef USE_CHEBY
        phiTmp = chebev(sim.ch_pot.phi, r);
        dTmp = chebev(sim.ch_pot.dphi, r);
        rhoTmp = chebev(sim.ch_pot.rho, r);
        dTmp2 = chebev(sim.ch_pot.drho, r);
#else
        eamInterpolateDeriv_opt(r, sim.eam_pot.phi, sim.eam_pot.phi_x0, sim.eam_pot.phi_xn, sim.eam_pot.phi_invDx, phiTmp, dTmp);
        eamInterpolateDeriv_opt(r, sim.eam_pot.rho, sim.eam_pot.rho_x0, sim.eam_pot.rho_xn, sim.eam_pot.rho_invDx, rhoTmp, dTmp2);
#endif
        break;
      case 3:
        eamInterpolateDeriv_opt(r, sim.eam_pot.rho, sim.eam_pot.rho_x0, sim.eam_pot.rho_xn, sim.eam_pot.rho_invDx, rhoTmp, dTmp);
        dTmp *= sim.fi[i_particle] + sim.fi[j_particle];
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
    fx += __shfl_xor<WARP_ATOM_REGS_CTA>(fx, i);
    fy += __shfl_xor<WARP_ATOM_REGS_CTA>(fy, i);
    fz += __shfl_xor<WARP_ATOM_REGS_CTA>(fz, i);
    if (step == 1) {
      e += __shfl_xor<WARP_ATOM_REGS_CTA>(e, i);
      rho += __shfl_xor<WARP_ATOM_REGS_CTA>(rho, i);
    }
  }
		
  // only 1 thread in a warp writes the result
  if (lane_id == 0) 
  {
    sim.f.x[i_particle] = fx;
    sim.f.y[i_particle] = fy;
    sim.f.z[i_particle] = fz;

    // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
    if (step == 1) {
      sim.e[i_particle] = e * 0.5;
      sim.rho[i_particle] = rho;	
    }
  }
}
