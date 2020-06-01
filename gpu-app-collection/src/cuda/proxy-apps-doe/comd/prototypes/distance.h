/*
  precompute distance between neighbor particles 
*/

__global__
__launch_bounds__(CTA_BOX_CTA, CTA_BOX_ACTIVE_CTAS)
void distance(sim_t sim)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  // 1 block per box
  int ibox = blockIdx.x;
  int natoms = sim.grid.n_atoms[ibox];
  int nneigh = sim.grid.n_num_neigh[ibox];

  // divide shared memory
  extern __shared__ real_t smem[];

  // neighbor positions
  volatile real_t *sdx = smem + 0;
  volatile real_t *sdy = smem + CTA_BOX_CTA;
  volatile real_t *sdz = smem + CTA_BOX_CTA * 2;

  // box atoms positions
  volatile real_t *isdx = smem + CTA_BOX_CTA * 3;
  volatile real_t *isdy = isdx + sim.max_atoms;
  volatile real_t *isdz = isdy + sim.max_atoms;

  if (threadIdx.x < natoms) {
    int n_id = ibox * N_MAX_ATOMS * N_MAX_NEIGHBORS + sim.grid.itself_start_idx[ibox] + threadIdx.x;
    int jbox = sim.grid.n_neigh_boxes[n_id];
    int j_particle = jbox * N_MAX_ATOMS + sim.grid.n_neigh_atoms[n_id]; // global offset of particle

    isdx[threadIdx.x] = sim.r.x[j_particle];
    isdy[threadIdx.x] = sim.r.y[j_particle];
    isdz[threadIdx.x] = sim.r.z[j_particle];
  }

  int global_base = 0;
  while (global_base < nneigh)
  {
    int global_neighbor = global_base + threadIdx.x;

    // check if last chunk is incomplete
    int tail = 0;
    if (global_base + CTA_BOX_CTA > nneigh) {
       tail = nneigh - global_base;
    }

    // load up neighbor particles in smem: 1 thread per neighbor atom
    if (global_neighbor < nneigh) 
    {
      int n_id = ibox * N_MAX_ATOMS * N_MAX_NEIGHBORS + global_neighbor;
      int jbox = sim.grid.n_neigh_boxes[n_id];
      int j_particle = jbox * N_MAX_ATOMS + sim.grid.n_neigh_atoms[n_id]; // global offset of particle

      // compute box center offsets
      real_t dxbox = sim.grid.r_box.x[ibox] - sim.grid.r_box.x[jbox];
      real_t dybox = sim.grid.r_box.y[ibox] - sim.grid.r_box.y[jbox];
      real_t dzbox = sim.grid.r_box.z[ibox] - sim.grid.r_box.z[jbox];

      // correct for periodic 
      if (PERIODIC)
      {
        if (dxbox < -(real_t)0.5 * sim.grid.bounds[0]) dxbox += sim.grid.bounds[0];
        else if (dxbox > (real_t)0.5 * sim.grid.bounds[0] ) dxbox -= sim.grid.bounds[0];
        if (dybox < -(real_t)0.5 * sim.grid.bounds[1]) dybox += sim.grid.bounds[1];
        else if (dybox > (real_t)0.5 * sim.grid.bounds[1] ) dybox -= sim.grid.bounds[1];
        if (dzbox < -(real_t)0.5 * sim.grid.bounds[2]) dzbox += sim.grid.bounds[2];
        else if (dzbox > (real_t)0.5 * sim.grid.bounds[2] ) dzbox -= sim.grid.bounds[2];
      }

      sdx[threadIdx.x] = - sim.r.x[j_particle] + dxbox;
      sdy[threadIdx.x] = - sim.r.y[j_particle] + dybox;
      sdz[threadIdx.x] = - sim.r.z[j_particle] + dzbox;
    }

    __syncthreads();
	
    // 1 warp process 1 atom now
    int iatom_base = 0;

    // only process atoms inside current box
    while (iatom_base < natoms) 
    {
      int iatom = iatom_base + warp_id;
      if (iatom < natoms) 
      {
        // atom global index
        int i_offset = ibox * N_MAX_ATOMS;
        int i_particle = i_offset + iatom;

        // square cutoff
        real_t r2cut = sim.eam_pot.cutoff * sim.eam_pot.cutoff;

	// num of neighbors is < 64 
	real_t *dist = &sim.dist[i_particle * N_MAX_ATOMS];

        int neighbor = lane_id;
        uint warpTotal = 0;
	for (int iters = 0; iters < CTA_BOX_WARPS; iters++)
        {
          // load neighbor positions from smem
	  real_t dx = isdx[iatom] + sdx[neighbor];
	  real_t dy = isdy[iatom] + sdy[neighbor];
	  real_t dz = isdz[iatom] + sdz[neighbor];
	  real_t r2 = dx*dx + dy*dy + dz*dz;

	  // aggregate neighbors that passes cut-off check
	  // warp-scan using ballot/popc 
	  uint flag = (r2 <= r2cut && r2 > 0 && 
		       (tail == 0 || neighbor < tail));		// flag(lane id) 
	  uint bits = __ballot(flag);				// 0 1 0 1  1 1 0 0 = flag(0) flag(1) .. flag(31)
          uint mask = bfi(0, 0xffffffff, 0, lane_id);		// bits < lane id = 1, bits > lane id = 0
	  uint exc = __popc(mask & bits);			// exclusive scan 

	  if (flag) dist[warpTotal + exc] = r2;			// fill nl array - compacted
	
	  warpTotal += __popc(bits);				// total 1s per warp
	  neighbor += WARP_SIZE;
        }

        _Z_intrinsic_pseudo_syncwarp();
      }

      iatom_base += CTA_BOX_WARPS;    
    }

    global_base += CTA_BOX_CTA;
  }
}
