__global__ void AdvanceVelocity(SimGpu sim, real_t dt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sim.n_local_atoms) return;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * N_MAX_ATOMS + iAtom;

  sim.p.x[iOff] += dt * sim.f.x[iOff]; 
  sim.p.y[iOff] += dt * sim.f.y[iOff]; 
  sim.p.z[iOff] += dt * sim.f.z[iOff]; 
}

__global__ void AdvancePosition(SimGpu sim, real_t dt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sim.n_local_atoms) return;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * N_MAX_ATOMS + iAtom;
  
  int iSpecies = sim.species_ids[iOff];
  real_t invMass = 1.0/sim.species_mass[iSpecies];

  sim.r.x[iOff] += dt * sim.p.x[iOff] * invMass;
  sim.r.y[iOff] += dt * sim.p.y[iOff] * invMass;
  sim.r.z[iOff] += dt * sim.p.z[iOff] * invMass;
}
