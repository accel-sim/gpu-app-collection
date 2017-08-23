__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void LJ_Force_thread_atom(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid]; 

  // common constants for LJ potential
  // TODO: this can be precomputed
  real_t sigma = sim.lj_pot.sigma;
  real_t epsilon = sim.lj_pot.epsilon;
  real_t rCut = sim.lj_pot.cutoff;
  real_t rCut2 = rCut*rCut;

  real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

  real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
  real_t eShift = rCut6 * (rCut6 - 1.0);

  // zero out forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;

  // fetch position
  int iOff = iBox * N_MAX_ATOMS + iAtom;
  real_t irx = sim.r.x[iOff];
  real_t iry = sim.r.y[iOff];
  real_t irz = sim.r.z[iOff];
 
  // loop over my neighbor cells
  for (int j = 0; j < N_MAX_NEIGHBORS; j++) 
  { 
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];

    // loop over all atoms in the neighbor cell 
    for (int jAtom = 0; jAtom < sim.num_atoms[jBox]; jAtom++) 
    {  
      int jOff = jBox * N_MAX_ATOMS + jAtom; 

      real_t dx = irx - sim.r.x[jOff];
      real_t dy = iry - sim.r.y[jOff];
      real_t dz = irz - sim.r.z[jOff];

      // distance^2
      real_t r2 = dx*dx + dy*dy + dz*dz;

      // no divide by zero
      if (r2 <= rCut2 && r2 > 0.0) 
      {
        r2 = 1.0/r2;
        real_t r6 = s6 * (r2*r2*r2);
        real_t eLocal = r6 * (r6 - 1.0) - eShift;

	// update energy
        ie += 0.5 * eLocal;

        // different formulation to avoid sqrt computation
        real_t fr = 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
  
        // update forces
        ifx += fr * dx;
        ify += fr * dy;
        ifz += fr * dz;
      } 
    } // loop over all atoms
  } // loop over neighbor cells

  sim.f.x[iOff] = ifx;
  sim.f.y[iOff] = ify;
  sim.f.z[iOff] = ifz;

  sim.e[iOff] = ie * 4 * epsilon;
}

