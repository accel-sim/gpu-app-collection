#define THREAD_ATOM_CTA		128

#ifdef DOUBLE
// 50% occupancy for DP: optimal sweet spot before spilling too much into local memory
#define THREAD_ATOM_ACTIVE_CTAS	8
#else
// 100% occupancy for SP
#define THREAD_ATOM_ACTIVE_CTAS	16
#endif

__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void LJ_Force_thread_atom(sim_t sim)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (id >= sim.total_atoms) return;

    int iatom = sim.grid.n_list_atoms[id];
    int ibox = sim.grid.n_list_boxes[id]; 

    real_t dx, dy, dz;
    real_t r2, r6;
    real_t e_i;

    real_t dxbox, dybox, dzbox;

    // accumulate local force value
    real_t fx_i, fy_i, fz_i;

    real_t rcut = sim.lj_pot.cutoff;
    real_t r2cut = rcut * rcut;
    real_t s6 = sim.lj_pot.sigma * sim.lj_pot.sigma;
    s6 = s6 * s6 * s6;

    real_t dTmp;

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

    fx_i = 0;
    fy_i = 0;
    fz_i = 0;
    e_i = 0;

    // each thread executes on a single atom in the box
    if (iatom < sim.grid.n_atoms[ibox]) 
    {
	// loop over neighbor cells
	for (j = 0; j < sim.grid.n_neighbors[ibox]; j++) 
	{ 
	    int jbox = sim.grid.neighbor_list[ibox * N_MAX_NEIGHBORS + j];
	    j_offset = jbox * N_MAX_ATOMS;

	    // compute box center offsets
	    dxbox = sim.grid.r_box.x[ibox] - sim.grid.r_box.x[jbox];
	    dybox = sim.grid.r_box.y[ibox] - sim.grid.r_box.y[jbox];
	    dzbox = sim.grid.r_box.z[ibox] - sim.grid.r_box.z[jbox];

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

	    // loop over all groups in neighbor cell 
	    for (jatom = 0; jatom < sim.grid.n_atoms[jbox]; jatom++) 
	    {  
		j_particle = j_offset + jatom; // global offset of particle

		dx = cx - sim.r.x[j_particle] + dxbox;
		dy = cy - sim.r.y[j_particle] + dybox;
		dz = cz - sim.r.z[j_particle] + dzbox;

		r2 = dx*dx + dy*dy + dz*dz;

		// no divide by zero
		if (r2 <= r2cut && r2 > 0.0) 
		{
		    r2 = (real_t)1.0/r2;
                    r6 = r2 * r2 * r2;
                    
                    dTmp = 4.0 * sim.lj_pot.epsilon * s6 * r2 * r6 * ((real_t)12.0 * r6 * s6 - (real_t)6.0);

		    fx_i += dTmp * dx;
		    fy_i += dTmp * dy;
		    fz_i += dTmp * dz;
		
		    e_i += r6 * (s6 * r6 - (real_t)1.0);
		} 
	    } // loop over all atoms
	} // loop over neighbor cells

	sim.f.x[i_particle] = fx_i;
	sim.f.y[i_particle] = fy_i;
	sim.f.z[i_particle] = fz_i;

	sim.e[i_particle] = e_i * 2 * sim.lj_pot.epsilon * s6;
    }
}

