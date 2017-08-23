// templated for the 1st and 3rd EAM passes
template<int step>
__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void EAM_Force_thread_atom_warp_sync(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid]; 
  int iOff = iBox * N_MAX_ATOMS + iAtom;

  // check if we can reconverge for this warp
#ifdef ECX_TARGET
  int warp_id = threadIdx.x / WARP_SIZE;
  int id_lane0 = blockIdx.x * blockDim.x + warp_id * WARP_SIZE;
  int ibox_lane0 = list.cells[id_lane0];
  bool can_reconverge = __all(iBox == ibox_lane0);
#endif

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = rCut*rCut;

  // init forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;
  real_t irho = 0;

  if (step == 3) {
    ifx = sim.f.x[iOff];
    ify = sim.f.y[iOff];
    ifz = sim.f.z[iOff];
  }

  // fetch position
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
        real_t r = sqrt(r2);

        real_t phiTmp, dPhi, rhoTmp, dRho;
	if (step == 1) {
          interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
          interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
        }
        else {
	  // step = 3
	  interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
	  dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
	}

        dPhi /= r;

        // update forces
        ifx -= dPhi * dx;
        ify -= dPhi * dy;
	ifz -= dPhi * dz;

        // update energy & accumulate rhobar
        if (step == 1) {
          ie += phiTmp;
          irho += rhoTmp;
        }
      } 
#ifdef ECX_TARGET
      if (can_reconverge)
        _Z_intrinsic_pseudo_syncwarp();
#endif
    } // loop over all atoms   
#ifdef ECX_TARGET
    _Z_intrinsic_pseudo_syncwarp();
#endif
  } // loop over neighbor cells

  sim.f.x[iOff] = ifx;
  sim.f.y[iOff] = ify;
  sim.f.z[iOff] = ifz;

  if (step == 1) {
    sim.e[iOff] = 0.5 * ie;
    sim.eam_pot.rhobar[iOff] = irho;
  }
}

