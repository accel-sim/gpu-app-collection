#define THREAD_ATOM_CTA		128

#ifdef DOUBLE
// 50% occupancy for DP: optimal sweet spot before spilling too much into local memory
#define THREAD_ATOM_ACTIVE_CTAS	8
#else
// 100% occupancy for SP
#define THREAD_ATOM_ACTIVE_CTAS	16
#endif

template<int step>
__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void EAM_Force_thread_atom(sim_t sim)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (id >= sim.total_atoms) return;

    int iatom = sim.grid.n_list_atoms[id];
    int ibox = sim.grid.n_list_boxes[id]; 

    real_t dx, dy, dz;
    real_t r, r2;

    // accumulate energy & rho
    real_t e_i = 0;
    real_t rho_i = 0;

    real_t dxbox, dybox, dzbox;

    // accumulate local force value
    real_t fx_i = 0;
    real_t fy_i = 0;
    real_t fz_i = 0;

    real_t rcut = sim.eam_pot.cutoff;
    real_t r2cut = rcut * rcut;
    real_t rhoTmp, phiTmp;
    real_t dTmp, dTmp2;

    int j;
    int jatom;

    int i_offset, j_offset;
    int i_particle, j_particle;

    // zero out forces on particles
    i_offset = ibox * N_MAX_ATOMS;
    i_particle = i_offset + iatom;

    real_t cx = sim.r.x[i_particle];
    real_t cy = sim.r.y[i_particle];
    real_t cz = sim.r.z[i_particle];

    // each thread executes on a single atom in the box
    if (iatom < sim.grid.n_atoms[ibox]) 
    {
	if (step == 3) {
	  // zero out forces on particles
          fx_i = sim.f.x[i_particle];
          fy_i = sim.f.y[i_particle];
          fz_i = sim.f.z[i_particle];
	}

	// loop over neighbor cells
	for (j = 0; j < sim.grid.n_neighbors[ibox]; j++) 
	{ 
	    int jbox = sim.grid.neighbor_list[ibox * N_MAX_NEIGHBORS + j];
	    j_offset = jbox * N_MAX_ATOMS;

	    // compute box center offsets
	    dxbox = sim.grid.r_box.x[ibox] - sim.grid.r_box.x[jbox];
	    dybox = sim.grid.r_box.y[ibox] - sim.grid.r_box.y[jbox];
	    dzbox = sim.grid.r_box.z[ibox] - sim.grid.r_box.z[jbox];

	    // loop over all groups in neighbor cell 
	    for (jatom = 0; jatom < sim.grid.n_atoms[jbox]; jatom++) 
	    {  
		j_particle = j_offset + jatom; // global offset of particle

		dx = cx - sim.r.x[j_particle] + dxbox;
		dy = cy - sim.r.y[j_particle] + dybox;
		dz = cz - sim.r.z[j_particle] + dzbox;

		r2 = dx*dx + dy*dy + dz*dz;

		// no divide by zero
		if (r2 <= r2cut && r2 > 0) 
		{
		    r = sqrt_opt(r2);

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
              	      dTmp *= (sim.fi[i_particle] + sim.fi[j_particle]);
        	      break;
	            }

		    dTmp /= r;

		    fx_i += dTmp * dx;
		    fy_i += dTmp * dy;
		    fz_i += dTmp * dz;
		
		    if (step == 1) {
		      e_i += phiTmp;
		      rho_i += rhoTmp;
		    }
		} 
	    } // loop over all atoms   
	} // loop over neighbor cells

	sim.f.x[i_particle] = fx_i;
	sim.f.y[i_particle] = fy_i;
	sim.f.z[i_particle] = fz_i;

	// since we loop over all particles, each particle contributes 1/2 the pair energy to the total
        if (step == 1) {
	  sim.e[i_particle] = (real_t)0.5 * e_i;
	  sim.rho[i_particle] = rho_i;
	}
    }
}

__global__ 
void EAM_Force_2(sim_t sim)
{
    int iatom = threadIdx.x;
    int ibox = blockIdx.x;

    real_t fi, fiprime;

    int i_offset;
    int i_particle;

    i_offset = ibox * N_MAX_ATOMS;
    i_particle = i_offset + iatom;

    // each thread executes on a single atom in the box
    if (iatom < sim.grid.n_atoms[ibox] && !is_halo(sim, ibox)) 
    {
	eamInterpolateDeriv(sim.rho[i_particle], sim.eam_pot.F, sim.eam_pot.n_values[2], &fi, &fiprime);

        // update rhoprime
	sim.fi[i_particle] = fiprime; 

	// update energy terms 
	sim.e[i_particle] += fi;
    }
}

__global__ 
void copy_halos_kernel(sim_t sim)
{
    int iatom = threadIdx.x;
    int ibox = blockIdx.x;

    int i_offset;
    int i_particle;

    i_offset = ibox * N_MAX_ATOMS;
    i_particle = i_offset + iatom;

    if (iatom < sim.grid.n_atoms[ibox] && is_halo(sim, ibox)) 
    {
		// find mirror box
		int jbox = get_mirror_cell_id(sim, ibox);

		// mirror particle index
		int j_offset = jbox * N_MAX_ATOMS;
		int j_particle = j_offset + iatom;

        // update rhoprime
		sim.fi[i_particle] = sim.fi[j_particle]; 

		// update positions
		sim.r.x[i_particle] = sim.r.x[j_particle];
		sim.r.y[i_particle] = sim.r.y[j_particle];
		sim.r.z[i_particle] = sim.r.z[j_particle];
    }
}



