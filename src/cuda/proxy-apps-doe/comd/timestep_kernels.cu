// timestep subroutines

#include "interface.h"

__device__
static bool is_halo(sim_t &sim, int ibox)
{
        if ((ibox % sim.nx) >= sim.nx-2) return true;
        if ((ibox / sim.nx) % sim.ny >= sim.ny-2) return true;
        if ((ibox / sim.nx) / sim.ny >= sim.nz-2) return true;
        return false;
}


__global__ void AdvanceVelocity (sim_t sim)
{
    int i_atom = threadIdx.x;
    for( int ibox = blockIdx.x; ibox < sim.n_cells; ibox += gridDim.x * blockDim.x ) {
	if (is_halo(sim, ibox)) continue;
    int offset = N_MAX_ATOMS;

    real_t dt_local = sim.dt;

    if (i_atom < sim.grid.n_atoms[ibox]) {
        sim.p.x[i_atom + offset*ibox] -= dt_local*sim.f.x[i_atom + offset*ibox];
        sim.p.y[i_atom + offset*ibox] -= dt_local*sim.f.y[i_atom + offset*ibox];
        sim.p.z[i_atom + offset*ibox] -= dt_local*sim.f.z[i_atom + offset*ibox];
    }
    }
}


extern "C"
void advance_velocity(sim_t sim_D)
{
	int grid = sim_D.n_cells;
        int block = sim_D.max_atoms;

	AdvanceVelocity<<<grid,block>>>(sim_D); 
	cudaDeviceSynchronize();
}

__global__ void AdvancePositions (sim_t sim)
{
    int i_atom = threadIdx.x;
    for( int ibox = blockIdx.x; ibox < sim.n_cells; ibox += gridDim.x * blockDim.x ) {
	if (is_halo(sim, ibox)) continue;
    int offset = N_MAX_ATOMS; 

    real_t dt_local = (sim.dt/2) / sim.rmass; 

    if (i_atom < sim.grid.n_atoms[ibox]) {
        sim.r.x[i_atom + offset*ibox] += dt_local*sim.p.x[i_atom + offset*ibox];
        sim.r.y[i_atom + offset*ibox] += dt_local*sim.p.y[i_atom + offset*ibox];
        sim.r.z[i_atom + offset*ibox] += dt_local*sim.p.z[i_atom + offset*ibox];
    }
    }
}

extern "C"
void advance_position(sim_t sim_D)
{
	int grid = sim_D.n_cells;
        int block = sim_D.max_atoms;
	
	AdvancePositions<<<grid,block>>>(sim_D); 
	cudaDeviceSynchronize();
}

