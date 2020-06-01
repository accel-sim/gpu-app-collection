#define WARP_ATOM_SMEM_WARPS		2
#define WARP_ATOM_SMEM_CTA		(32 * WARP_ATOM_SMEM_WARPS)

template<int step>
__global__
__launch_bounds__(WARP_ATOM_SMEM_CTA, 16)
void EAM_Force_wa_smem(sim_t sim)
{
	// 1 warp per 1 atom
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;

	int id = blockIdx.x * WARP_ATOM_SMEM_WARPS + warp_id;
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

	// accumulate local force in smem
    __shared__ volatile real_t acc_fx[WARP_ATOM_SMEM_CTA];
    __shared__ volatile real_t acc_fy[WARP_ATOM_SMEM_CTA];
    __shared__ volatile real_t acc_fz[WARP_ATOM_SMEM_CTA];
    __shared__ volatile real_t acc_e[WARP_ATOM_SMEM_CTA];
    __shared__ volatile real_t acc_rho[WARP_ATOM_SMEM_CTA];

	acc_fx[threadIdx.x] = 0;
	acc_fy[threadIdx.x] = 0;
	acc_fz[threadIdx.x] = 0;
	acc_rho[threadIdx.x] = 0;
	acc_e[threadIdx.x] = 0;

        if (step == 3) {
          // zero out forces on particles
          acc_fx[threadIdx.x] = sim.f.x[i_particle];
          acc_fy[threadIdx.x] = sim.f.y[i_particle];
          acc_fz[threadIdx.x] = sim.f.z[i_particle];
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

		    acc_fx[threadIdx.x] += dTmp * dx / r;
		    acc_fy[threadIdx.x] += dTmp * dy / r;
		    acc_fz[threadIdx.x] += dTmp * dz / r;

		    if (step == 1) {
		      acc_e[threadIdx.x] += phiTmp;
		      acc_rho[threadIdx.x] += rhoTmp;
		    }
		}

		neighbor += 32;
	}

	// warp reduction
    if (lane_id < 16) {
        acc_fx[threadIdx.x] += acc_fx[threadIdx.x + 16];
        acc_fy[threadIdx.x] += acc_fy[threadIdx.x + 16];
        acc_fz[threadIdx.x] += acc_fz[threadIdx.x + 16];
	if (step == 1) {
          acc_e[threadIdx.x] += acc_e[threadIdx.x + 16];
          acc_rho[threadIdx.x] += acc_rho[threadIdx.x + 16];
 	}
	}
	if (lane_id < 8) {
        acc_fx[threadIdx.x] += acc_fx[threadIdx.x + 8];
        acc_fy[threadIdx.x] += acc_fy[threadIdx.x + 8];
        acc_fz[threadIdx.x] += acc_fz[threadIdx.x + 8];
	if (step == 1) {
        acc_e[threadIdx.x] += acc_e[threadIdx.x + 8];
        acc_rho[threadIdx.x] += acc_rho[threadIdx.x + 8];
 	}
	}
	if (lane_id < 4) {
        acc_fx[threadIdx.x] += acc_fx[threadIdx.x + 4];
        acc_fy[threadIdx.x] += acc_fy[threadIdx.x + 4];
        acc_fz[threadIdx.x] += acc_fz[threadIdx.x + 4];
	if (step == 1) {
        acc_e[threadIdx.x] += acc_e[threadIdx.x + 4];
        acc_rho[threadIdx.x] += acc_rho[threadIdx.x + 4];
 	}
	}
	if (lane_id < 2) {
        acc_fx[threadIdx.x] += acc_fx[threadIdx.x + 2];
        acc_fy[threadIdx.x] += acc_fy[threadIdx.x + 2];
        acc_fz[threadIdx.x] += acc_fz[threadIdx.x + 2];
	if (step == 1) {
        acc_e[threadIdx.x] += acc_e[threadIdx.x + 2];
        acc_rho[threadIdx.x] += acc_rho[threadIdx.x + 2];
 	}
	}
	if (lane_id < 1) {
        acc_fx[threadIdx.x] += acc_fx[threadIdx.x + 1];
        acc_fy[threadIdx.x] += acc_fy[threadIdx.x + 1];
        acc_fz[threadIdx.x] += acc_fz[threadIdx.x + 1];
	if (step == 1) {
        acc_e[threadIdx.x] += acc_e[threadIdx.x + 1];
        acc_rho[threadIdx.x] += acc_rho[threadIdx.x + 1];
 	}
	}
		
    // only 1 thread in a warp writes the result
    if (lane_id == 0) 
    {
		sim.f.x[i_particle] = acc_fx[warp_id * 32];
		sim.f.y[i_particle] = acc_fy[warp_id * 32];
		sim.f.z[i_particle] = acc_fz[warp_id * 32];

 		// since we loop over all particles, each particle contributes 1/2 the pair energy to the total
		if (step == 1) {
 		  sim.e[i_particle] = acc_e[warp_id * 32] * 0.5;
		  sim.rho[i_particle] = acc_rho[warp_id * 32];	
		}
	}
}
