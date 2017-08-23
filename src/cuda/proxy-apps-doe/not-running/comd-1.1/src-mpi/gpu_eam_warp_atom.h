// templated for the 1st and 3rd EAM passes
template<int step>
__global__
__launch_bounds__(WARP_ATOM_CTA, WARP_ATOM_ACTIVE_CTAS)
void EAM_Force_warp_atom(SimGpu sim, AtomListGpu list)
{
  // warp & lane ids
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int tid = blockIdx.x * (WARP_ATOM_CTA / WARP_SIZE) + warp_id;
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid];

  // per-warp neighbor offsets
  __shared__ int smem_nl_off[(WARP_ATOM_CTA / WARP_SIZE) * 64];
  int *nl_off = smem_nl_off + warp_id * 64;

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = rCut*rCut;
  
  int iOff = iBox * N_MAX_ATOMS + iAtom;

  // init forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;
  real_t irho = 0;

  // fetch position
  real_t irx = sim.r.x[iOff];
  real_t iry = sim.r.y[iOff];
  real_t irz = sim.r.z[iOff];

  // create neighbor list
  int j = lane_id;
  int numNeigh = sim.num_neigh_atoms[iBox];
  int numSteps = (numNeigh + (WARP_SIZE-1)) / WARP_SIZE;
  int warpTotal = 0;
  for (int it = 0; it < numSteps; it++) 
  {
    int jOff;
    real_t dx, dy, dz, r2;

    // check for out of bounds
    if (j < numNeigh) {
      // index
      jOff = sim.neighbor_atoms[iBox * N_MAX_NEIGHBORS * N_MAX_ATOMS + j];

      dx = irx - sim.r.x[jOff];
      dy = iry - sim.r.y[jOff];
      dz = irz - sim.r.z[jOff];

      // distance^2
      r2 = dx*dx + dy*dy + dz*dz;
    }

    // aggregate neighbors that passes cut-off check
    // warp-scan using ballot/popc 
    uint flag = (r2 <= rCut2 && r2 > 0 && j < numNeigh);  // flag(lane id) 
    uint bits = __ballot(flag);                           // 0 1 0 1  1 1 0 0 = flag(0) flag(1) .. flag(31)
    uint mask = bfi(0, 0xffffffff, 0, lane_id);           // bits < lane id = 1, bits > lane id = 0
    uint exc = __popc(mask & bits);                       // exclusive scan 

    if (flag) 
      nl_off[warpTotal + exc] = jOff;     		  // fill nl array - compacted

    warpTotal += __popc(bits);                            // total 1s per warp

    // move on to the next neighbor atom
    j += WARP_SIZE;
  }

  int neighbor_id = lane_id;
  for (int iters = 0; iters < 64 / WARP_SIZE; iters++) 
  {
    if (neighbor_id >= warpTotal) break;
    int jOff = nl_off[neighbor_id];

    real_t dx = irx - sim.r.x[jOff];
    real_t dy = iry - sim.r.y[jOff];
    real_t dz = irz - sim.r.z[jOff];

    real_t r2 = dx*dx + dy*dy + dz*dz;
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

    neighbor_id += WARP_SIZE;
  }

  warp_reduce<step>(ifx, ify, ifz, ie, irho);

  // single thread writes the final result
  if (lane_id == 0) {
    if (step == 1)
    {
      sim.f.x[iOff] = ifx;
      sim.f.y[iOff] = ify;
      sim.f.z[iOff] = ifz;
      sim.e[iOff] = 0.5 * ie;
      sim.eam_pot.rhobar[iOff] = irho;
    }
    else {
      // step 3
      sim.f.x[iOff] += ifx;
      sim.f.y[iOff] += ify;
      sim.f.z[iOff] += ifz;
    }
  }
}

