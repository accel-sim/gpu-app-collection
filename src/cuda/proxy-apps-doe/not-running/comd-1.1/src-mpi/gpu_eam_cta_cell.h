// templated for the 1st and 3rd EAM passes
// 1 cta is assigned to 1 cell
// all threads shared the same set of neighbors
// so we can store neighbor positions in smem
// 1 warp processes 1 atom
template<int step>
__global__
__launch_bounds__(CTA_CELL_CTA, CTA_CELL_ACTIVE_CTAS)
void EAM_Force_cta_cell(SimGpu sim)
{
  // warp & lane ids
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  // compute box ID
  int iBox = blockIdx.x;

  // distribute smem
#if 0
  extern __shared__ real_t smem[];

  // neighbor positions
  volatile real_t *rx = smem;
  volatile real_t *ry = rx + CTA_CELL_CTA;
  volatile real_t *rz = rx + 2 * CTA_CELL_CTA;

  // neighbor embed force
  volatile real_t *fe = 0;
  if (step == 3)
    fe = rx + 3 * CTA_CELL_CTA;

  // local positions
  volatile real_t *irx = smem + ((step == 1) ? 3 * CTA_CELL_CTA : 4 * CTA_CELL_CTA);
  volatile real_t *iry = irx + sim.max_atoms_cell;
  volatile real_t *irz = iry + sim.max_atoms_cell;

  // local forces
  volatile real_t *ifx = irz + sim.max_atoms_cell;
  volatile real_t *ify = ifx + sim.max_atoms_cell;
  volatile real_t *ifz = ify + sim.max_atoms_cell;
  volatile real_t *ie = 0;
  volatile real_t *irho = 0;
  if (step == 1) {
    ie = ifz + sim.max_atoms_cell;
    irho = ie + sim.max_atoms_cell;
  }

  // per-warp neighbor offsets
  volatile void *smem_nl_off = ((step == 1) ? irho : ifz) + sim.max_atoms_cell;
  volatile char *nl_off = (char*)smem_nl_off + warp_id * 64;
#else
  __shared__ real_t rx[CTA_CELL_CTA];
  __shared__ real_t ry[CTA_CELL_CTA];
  __shared__ real_t rz[CTA_CELL_CTA];

  real_t *fe;

  __shared__ real_t irx[32];
  __shared__ real_t iry[32];
  __shared__ real_t irz[32];

  __shared__ real_t ifx[32];
  __shared__ real_t ify[32];
  __shared__ real_t ifz[32];
  __shared__ real_t ie[32];
  __shared__ real_t irho[32];

  __shared__ int smem_nl_off[(CTA_CELL_CTA / WARP_SIZE) * 64];
  int *nl_off = smem_nl_off + warp_id * 64;
#endif

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = rCut*rCut;
  
  // num of cell atoms & neighbor atoms
  int natoms = sim.num_atoms[iBox];
  int nneigh = sim.num_neigh_atoms[iBox];

  // process neighbors in chunks of CTA size to save on smem
  int global_base = 0;
  while (global_base < nneigh)
  {
    int global_neighbor = global_base + threadIdx.x;

    // last chunk might be incomplete
    int tail = 0;
    if (global_base + CTA_CELL_CTA > nneigh)
      tail = nneigh - global_base;

    // load up neighbor particles in smem: 1 thread per neighbor atom
    if (global_neighbor < nneigh)
    {
      int jOff = sim.neighbor_atoms[iBox * N_MAX_NEIGHBORS * N_MAX_ATOMS + global_neighbor];

      rx[threadIdx.x] = sim.r.x[jOff];
      ry[threadIdx.x] = sim.r.y[jOff];
      rz[threadIdx.x] = sim.r.z[jOff];

      if (step == 3)
	fe[threadIdx.x] = sim.eam_pot.dfEmbed[jOff];
 
      // save local atoms positions
      if (global_neighbor < natoms) {
	irx[threadIdx.x] = rx[threadIdx.x];
	iry[threadIdx.x] = ry[threadIdx.x];
	irz[threadIdx.x] = rz[threadIdx.x];
 
	ifx[threadIdx.x] = 0;
	ify[threadIdx.x] = 0;
	ifz[threadIdx.x] = 0;
	ie[threadIdx.x] = 0;
	irho[threadIdx.x] = 0;
      }
    }

    // ensure data is loaded
    __syncthreads();

    // 1 warp is assigned to 1 atom
    int iatom_base = 0;

    // only process atoms inside current box
    while (iatom_base < natoms)
    {
      int iAtom = iatom_base + warp_id;
      if (iAtom < natoms)
      {
        // init forces and energy
        real_t reg_ifx = 0;
        real_t reg_ify = 0;
        real_t reg_ifz = 0;
        real_t reg_ie = 0;
        real_t reg_irho = 0;
	  
        real_t dx, dy, dz, r2;
		
        // create neighbor list
        int j = lane_id;
        int warpTotal = 0;
	for (int it = 0; it < CTA_CELL_CTA / WARP_SIZE; it++) 
  	{
	  if (tail == 0 || j < tail) 
          {
	    dx = irx[iAtom] - rx[j];
	    dy = iry[iAtom] - ry[j];
	    dz = irz[iAtom] - rz[j];

            // distance^2
            r2 = dx*dx + dy*dy + dz*dz;
	  }

	  // aggregate neighbors that passes cut-off check
	  // warp-scan using ballot/popc 	
	  uint flag = (r2 <= rCut2 && r2 > 0 && (tail == 0 || j < tail));  // flag(lane id) 
	  uint bits = __ballot(flag);                           // 0 1 0 1  1 1 0 0 = flag(0) flag(1) .. flag(31)
	  uint mask = bfi(0, 0xffffffff, 0, lane_id);           // bits < lane id = 1, bits > lane id = 0
	  uint exc = __popc(mask & bits);                       // exclusive scan 

	  if (flag) 
	    nl_off[warpTotal + exc] = j; 	    		  // fill nl array - compacted

	  warpTotal += __popc(bits);                            // total 1s per warp

	  // move on to the next neighbor atom
	  j += WARP_SIZE;
	} // compute neighbor lists

        int neighbor_id = lane_id;
        for (int iters = 0; iters < 64 / WARP_SIZE; iters++) 
    	{
	  if (neighbor_id >= warpTotal) break;
	  j = nl_off[neighbor_id];

	  dx = irx[iAtom] - rx[j];
	  dy = iry[iAtom] - ry[j];
	  dz = irz[iAtom] - rz[j];

	  r2 = dx*dx + dy*dy + dz*dz;
	  real_t r = sqrt(r2);	

	  real_t phiTmp, dPhi, rhoTmp, dRho;
	  if (step == 1) {
	    interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
	    interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
	  }
	  else {
	    // step = 3
	    // TODO: this is not optimal
	    interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
	    int iOff = iBox * N_MAX_ATOMS + iAtom;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
            dPhi = (__ldg(sim.eam_pot.dfEmbed + iOff) + fe[j]) * dRho;
#else
	    dPhi = (sim.eam_pot.dfEmbed[iOff] + fe[j]) * dRho;
#endif
	  }
	
	  dPhi /= r;

	  // update forces
	  reg_ifx -= dPhi * dx;
	  reg_ify -= dPhi * dy;
	  reg_ifz -= dPhi * dz;
	
	  // update energy & accumulate rhobar
	  if (step == 1) {
	    reg_ie += phiTmp;
	    reg_irho += rhoTmp;
	  }	

	  neighbor_id += WARP_SIZE;
	} // accumulate forces in regs

	warp_reduce<step>(reg_ifx, reg_ify, reg_ifz, reg_ie, reg_irho);

  	// single thread writes the final result
	if (lane_id == 0) 
        {
	  ifx[iAtom] += reg_ifx;
	  ify[iAtom] += reg_ify;
	  ifz[iAtom] += reg_ifz;

    	  if (step == 1) {
	    ie[iAtom] += reg_ie;
	    irho[iAtom] += reg_irho;
	  }
        }
      }  // check if iAtom < num atoms

      iatom_base += CTA_CELL_CTA / WARP_SIZE;
    }  // iterate on all atoms in cell

    __syncthreads();
 
    global_base += CTA_CELL_CTA;
  }  // iterate on all neighbors

  // single thread writes the final result for each atom
  if (threadIdx.x < natoms) {
    int iAtom = threadIdx.x;
    int iOff = iBox * N_MAX_ATOMS + threadIdx.x;
    if (step == 1)
    {
      sim.f.x[iOff] = ifx[iAtom];
      sim.f.y[iOff] = ify[iAtom];
      sim.f.z[iOff] = ifz[iAtom];
      sim.e[iOff] = 0.5 * ie[iAtom];
      sim.eam_pot.rhobar[iOff] = irho[iAtom];
    }
    else {
      // step 3
      sim.f.x[iOff] += ifx[iAtom];
      sim.f.y[iOff] += ify[iAtom];
      sim.f.z[iOff] += ifz[iAtom];
    }
  }
}

