__device__
int getBoxFromTuple(LinkCellGpu cells, int ix, int iy, int iz)
{
   int iBox = 0;

   // Halo in Z+
   if (iz == cells.gridSize.z)
   {
      iBox = cells.n_local + 2 * cells.gridSize.z * cells.gridSize.y + 2 * cells.gridSize.z * (cells.gridSize.x + 2) +
         (cells.gridSize.x + 2) * (cells.gridSize.y + 2) + (cells.gridSize.x + 2) * (iy + 1) + (ix + 1);
   }
   // Halo in Z-
   else if (iz == -1)
   {
      iBox = cells.n_local + 2 * cells.gridSize.z * cells.gridSize.y + 2 * cells.gridSize.z * (cells.gridSize.x + 2) +
         (cells.gridSize.x + 2) * (iy + 1) + (ix + 1);
   }
   // Halo in Y+
   else if (iy == cells.gridSize.y)
   {
      iBox = cells.n_local + 2 * cells.gridSize.z * cells.gridSize.y + cells.gridSize.z * (cells.gridSize.x + 2) +
         (cells.gridSize.x + 2) * iz + (ix + 1);
   }
   // Halo in Y-
   else if (iy == -1)
   {
      iBox = cells.n_local + 2 * cells.gridSize.z * cells.gridSize.y + iz * (cells.gridSize.x + 2) + (ix + 1);
   }
   // Halo in X+
   else if (ix == cells.gridSize.x)
   {
      iBox = cells.n_local + cells.gridSize.y * cells.gridSize.z + iz * cells.gridSize.y + iy;
   }
   // Halo in X-
   else if (ix == -1)
   {
      iBox = cells.n_local + iz * cells.gridSize.y + iy;
   }
   // local link celll.
   else
   {
      iBox = ix + cells.gridSize.x * iy + cells.gridSize.x * cells.gridSize.y * iz;
   }

   return iBox;
}

/// Get the index of the link cell that contains the specified
/// coordinate.  This can be either a halo or a local link cell.
///
/// Because the rank ownership of an atom is strictly determined by the
/// atom's position, we need to take care that all ranks will agree which
/// rank owns an atom.  The conditionals at the end of this function are
/// special care to ensure that all ranks make compatible link cell
/// assignments for atoms that are near a link cell boundaries.  If no
/// ranks claim an atom in a local cell it will be lost.  If multiple
/// ranks claim an atom it will be duplicated.
__device__
int getBoxFromCoord(LinkCellGpu cells, real_t rx, real_t ry, real_t rz)
{
   int ix = (int)(floor((rx - cells.localMin.x) * cells.invBoxSize.x));
   int iy = (int)(floor((ry - cells.localMin.y) * cells.invBoxSize.y));
   int iz = (int)(floor((rz - cells.localMin.z) * cells.invBoxSize.z));

   // For each axis, if we are inside the local domain, make sure we get
   // a local link cell.  Otherwise, make sure we get a halo link cell.
   if (rx < cells.localMax.x)
   {
      if (ix == cells.gridSize.x) ix = cells.gridSize.x - 1;
   }
   else
      ix = cells.gridSize.x; // assign to halo cell
   if (ry < cells.localMax.y)
   {
      if (iy == cells.gridSize.y) iy = cells.gridSize.y - 1;
   }
   else
      iy = cells.gridSize.y;
   if (rz < cells.localMax.z)
   {
      if (iz == cells.gridSize.z) iz = cells.gridSize.z - 1;
   }
   else
      iz = cells.gridSize.z;

   return getBoxFromTuple(cells, ix, iy, iz);
}

__global__ void UpdateLinkCells(SimGpu sim, LinkCellGpu cells, int *flags)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sim.n_local_atoms) return;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * N_MAX_ATOMS + iAtom;

  int jBox = getBoxFromCoord(cells, sim.r.x[iOff], sim.r.y[iOff], sim.r.z[iOff]);
  
  if (jBox != iBox) {
    // find new position in jBox list
    int jAtom = atomicAdd(&sim.num_atoms[jBox], 1);
    int jOff = jBox * N_MAX_ATOMS + jAtom;
    
    // flag set/unset   
    flags[jOff] = tid+1;
    flags[iOff] = 0;

    // copy over the atoms data
    sim.r.x[jOff] = sim.r.x[iOff];
    sim.r.y[jOff] = sim.r.y[iOff];
    sim.r.z[jOff] = sim.r.z[iOff];
    sim.p.x[jOff] = sim.p.x[iOff];
    sim.p.y[jOff] = sim.p.y[iOff];
    sim.p.z[jOff] = sim.p.z[iOff];
    sim.gid[jOff] = sim.gid[iOff];
    sim.species_ids[jOff] = sim.species_ids[iOff];
    sim.a_list.atoms[tid] = jAtom;
    sim.a_list.cells[tid] = jBox;
  }
  else
    flags[iOff] = tid+1;
}

// TODO: improve parallelism, currently it's one thread per cell!
__global__ void CompactAtoms(SimGpu sim, int num_cells, int *flags)
{
  // only process local cells
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_cells) return;

  int iBox = tid;
  int jBox = tid;
  int jAtom = 0;

  for (int iAtom = 0, iOff = iBox * N_MAX_ATOMS; iAtom < N_MAX_ATOMS; iAtom++, iOff++)
    if (flags[iBox * N_MAX_ATOMS + iAtom] > 0) {
      int jOff = jBox * N_MAX_ATOMS + jAtom;
      if (iOff != jOff) {
        sim.r.x[jOff] = sim.r.x[iOff];
        sim.r.y[jOff] = sim.r.y[iOff];
        sim.r.z[jOff] = sim.r.z[iOff];
        sim.p.x[jOff] = sim.p.x[iOff];
        sim.p.y[jOff] = sim.p.y[iOff];
        sim.p.z[jOff] = sim.p.z[iOff];
        sim.gid[jOff] = sim.gid[iOff];
        sim.species_ids[jOff] = sim.species_ids[iOff];
      }
      jAtom++;
    }

  // update # of atoms in the box
  sim.num_atoms[jBox] = jAtom;
}

// 1 warp per neighbor cell, 1 CTA per cell
__global__ void UpdateNeighborAtomIndices(SimGpu sim, int num_cells, int *cell_list, int *scan)
{
  int iBox = blockIdx.x;
  if (cell_list != NULL)
    iBox = cell_list[blockIdx.x];

  // load num atoms into smem
  __shared__ real_t ncell[N_MAX_NEIGHBORS];
  __shared__ real_t natoms[N_MAX_NEIGHBORS];
  __shared__ real_t npos[N_MAX_NEIGHBORS];
  if (threadIdx.x < N_MAX_NEIGHBORS) { 
    int j = threadIdx.x;
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
    ncell[j] = jBox;
    natoms[j] = sim.num_atoms[jBox];
    npos[j] = scan[iBox * N_MAX_NEIGHBORS + j]; 
  }

  __syncthreads();

  // each thread finds its box index
  int local_index = threadIdx.x;
  int j = 0;
  while (j < N_MAX_NEIGHBORS) {
    while (j < N_MAX_NEIGHBORS && natoms[j] <= local_index) { local_index -= natoms[j]; j++; }
    if (j < N_MAX_NEIGHBORS) {
      int pos = iBox * N_MAX_NEIGHBORS * N_MAX_ATOMS + npos[j] + local_index; 
      sim.neighbor_atoms[pos] = ncell[j] * N_MAX_ATOMS + local_index;
      local_index += blockDim.x;
    }
  }
/*
  int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
  int pos = scan[iBox * N_MAX_NEIGHBORS + j];
  int num = sim.num_atoms[jBox];

  if (lane_id < num) 
    sim.neighbor_atoms[iBox * N_MAX_NEIGHBORS * N_MAX_ATOMS + pos + lane_id] = jBox * N_MAX_ATOMS + lane_id;
*/
}

__global__ void UpdateNeighborNumAtoms(SimGpu sim, int num_cells, int *cell_list, int *scan)
{
  // only process local cells
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_cells) return;

  int iBox = tid;
  if (cell_list != NULL) 
    iBox = cell_list[tid];

  int num_neigh = 0;
  for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
    scan[iBox * N_MAX_NEIGHBORS + j] = num_neigh;
    num_neigh += sim.num_atoms[jBox];
  }

  sim.num_neigh_atoms[iBox] = num_neigh;
}

// 1 warp per cell
__global__ void UpdateAtomList(SimGpu sim, AtomListGpu list, int nCells, int *cell_offsets)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;

  int iBox = tid / WARP_SIZE;
  if (iBox >= nCells) return;

  int nAtoms = sim.num_atoms[iBox];
  for (int i = lane_id; i < nAtoms; i += WARP_SIZE) {
    int off = cell_offsets[iBox] + i;
    list.atoms[off] = i;
    list.cells[off] = iBox;
  }
}

// 1 warp per cell
__global__ void UpdateBoundaryList(SimGpu sim, AtomListGpu list, int nCells, int *cell_offsets, int *cellList)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;

  int iListBox = tid / WARP_SIZE;
  if (iListBox >= nCells) return;

  int iBox = cellList[iListBox];
  int nAtoms = sim.num_atoms[iBox];
  for (int i = lane_id; i < nAtoms; i += WARP_SIZE) {
    int off = cell_offsets[iListBox] + i;
    list.atoms[off] = i;
    list.cells[off] = iBox;
  }
}

__global__ void LoadAtomsBufferPacked(char *buf, int *gpu_cells, int nCells, SimGpu sim, int *cell_indices, real_t shift_x, real_t shift_y, real_t shift_z)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iCell = tid / N_MAX_ATOMS;
  int iAtom = tid % N_MAX_ATOMS;

  int iBox = gpu_cells[iCell];
  int ii = iBox * N_MAX_ATOMS + iAtom;
  int size = cell_indices[nCells];

  if (iAtom < sim.num_atoms[iBox]) 
  {
    int nBuf = cell_indices[iCell] + iAtom;

    int *buf_gid = (int*)buf;
    int *buf_type = buf_gid + size;
    real_t *buf_rx = (real_t*)(buf_type + size);
    real_t *buf_ry = buf_rx + size;
    real_t *buf_rz = buf_ry + size;
    real_t *buf_px = buf_rz + size;
    real_t *buf_py = buf_px + size;
    real_t *buf_pz = buf_py + size;

    // coalescing writes: structure of arrays
    buf_gid[nBuf]  = sim.gid[ii];
    buf_type[nBuf] = sim.species_ids[ii];
    buf_rx[nBuf] = sim.r.x[ii] + shift_x;
    buf_ry[nBuf] = sim.r.y[ii] + shift_y;
    buf_rz[nBuf] = sim.r.z[ii] + shift_z;
    buf_px[nBuf] = sim.p.x[ii];
    buf_py[nBuf] = sim.p.y[ii];
    buf_pz[nBuf] = sim.p.z[ii];
  }
}

__global__ void UnloadAtomsBufferPacked(char *buf, int nBuf, SimGpu sim, LinkCellGpu cells)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nBuf) return;

  int size = nBuf;
  int *buf_gid = (int*)buf;
  int *buf_type = buf_gid + size;
  real_t *buf_rx = (real_t*)(buf_type + size);
  real_t *buf_ry = buf_rx + size;
  real_t *buf_rz = buf_ry + size;
  real_t *buf_px = buf_rz + size;
  real_t *buf_py = buf_px + size;
  real_t *buf_pz = buf_py + size;

  real_t rx = buf_rx[tid];
  real_t ry = buf_ry[tid];
  real_t rz = buf_rz[tid];

  int iBox = getBoxFromCoord(cells, rx, ry, rz);
  int iAtom = atomicAdd(&sim.num_atoms[iBox], 1);
  int iOff = iBox * N_MAX_ATOMS + iAtom;

  // copy data
  sim.r.x[iOff] = rx;
  sim.r.y[iOff] = ry;
  sim.r.z[iOff] = rz;
  sim.p.x[iOff] = buf_px[tid];
  sim.p.y[iOff] = buf_py[tid];
  sim.p.z[iOff] = buf_pz[tid];
  sim.species_ids[iOff] = buf_type[tid];
  sim.gid[iOff] = buf_gid[tid];
}

__global__ void UpdateCompactIndices(int *cell_indices, int nLocalBoxes, SimGpu sim)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iBox = tid / N_MAX_ATOMS;
  int iAtom = tid % N_MAX_ATOMS;

  if (iBox < nLocalBoxes && iAtom < sim.num_atoms[iBox]) 
  {
    int iAtom = tid % N_MAX_ATOMS;
    int id = cell_indices[iBox] + iAtom;
    sim.a_list.atoms[id] = iAtom;
    sim.a_list.cells[id] = iBox;
  }
}

__global__ void LoadForceBuffer(ForceMsg *buf, int nCells, int *gpu_cells, SimGpu sim, int *cell_indices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iCell = tid / N_MAX_ATOMS;
  int iAtom = tid % N_MAX_ATOMS;

  if (iCell < nCells) {
    int iBox = gpu_cells[iCell];
    int ii = iBox * N_MAX_ATOMS + iAtom;

    if (iAtom < sim.num_atoms[iBox])
    {
      int nBuf = cell_indices[iCell] + iAtom;
      buf[nBuf].dfEmbed = sim.eam_pot.dfEmbed[ii];
    }
  }
}

__global__ void UnloadForceBuffer(ForceMsg *buf, int nCells, int *gpu_cells, SimGpu sim, int *cell_indices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iCell = tid / N_MAX_ATOMS;
  int iAtom = tid % N_MAX_ATOMS;

  if (iCell < nCells) {
    int iBox = gpu_cells[iCell];
    int ii = iBox * N_MAX_ATOMS + iAtom;

    if (iAtom < sim.num_atoms[iBox])
    {
      int nBuf = cell_indices[iCell] + iAtom;
      sim.eam_pot.dfEmbed[ii] = buf[nBuf].dfEmbed;
    }
  }
}

template<typename T>
__device__ void swap(T &a, T &b)
{
  T c = a;
  a = b;
  b = c;
}

__global__ void SetLinearIndices(SimGpu sim, int num_cells, int *cell_list, int *new_indices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  if (warp_id >= num_cells) return;

  int iBox = cell_list[warp_id];
  int num_atoms = sim.num_atoms[iBox];
  for (int iAtom = lane_id; iAtom < num_atoms; iAtom += WARP_SIZE) {
    int iOff = iBox * N_MAX_ATOMS + iAtom;
    new_indices[iOff] = iOff;
  }
}

__global__ void SetLinearIndices(SimGpu sim, int nLocalBoxes, int nTotalBoxes, int *new_indices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int iBox = nLocalBoxes + tid / N_MAX_ATOMS;
  int iAtom = tid % N_MAX_ATOMS;
  
  if (iBox < nTotalBoxes) {
    if (iAtom < sim.num_atoms[iBox]) {
      int iOff = iBox * N_MAX_ATOMS + iAtom;
      new_indices[iOff] = iOff;
    }
  }
}

#if 0
// bubble sort
__global__ void SortAtomsByGlobalId(SimGpu sim, int nTotalBoxes, int *new_indices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iBox = tid;
  
  if (iBox < nTotalBoxes) {
    int natoms = sim.num_atoms[iBox];
    for (int i = 0; i < natoms; i++) {
      int iOff = iBox * N_MAX_ATOMS + i;
      for (int j = i+1; j < natoms; j++) {
	int jOff = iBox * N_MAX_ATOMS + j;
        if (sim.gid[new_indices[iOff]] > sim.gid[new_indices[jOff]]) {
	  swap(new_indices[iOff], new_indices[jOff]);
        }
      }
    }
  }
}
#else
__device__ void bottomUpMerge(int *gid, int *A, int iLeft, int iRight, int iEnd, int *B)
{
  int i0 = iLeft;
  int i1 = iRight;

  for (int j = iLeft; j < iEnd; j++) {
    if (i0 < iRight && (i1 >= iEnd || gid[A[i0]] <= gid[A[i1]])) {   
      B[j] = A[i0];
      i0++;
    }
    else {
      B[j] = A[i1];
      i1++;
    }
  }
}

// merge sort
__global__ void SortAtomsByGlobalId(SimGpu sim, int nLocalBoxes, int nTotalBoxes, int *boundary_cells, int nBoundaryCells, int *new_indices, int *tmp_sort)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int iBox;
  if (tid >= nBoundaryCells) iBox = nLocalBoxes + tid - nBoundaryCells;
    else iBox = boundary_cells[tid];
  
  if (iBox < nTotalBoxes && new_indices[iBox * N_MAX_ATOMS] >= 0) 
  {
    int n = sim.num_atoms[iBox];
    int *A = new_indices + iBox * N_MAX_ATOMS;
    int *B = tmp_sort + iBox * N_MAX_ATOMS;
    // each 1-element run in A is already "sorted"
    // make succcessively longer sorted runs of length 2, 4, 8, etc.
    for (int width = 1; width < n; width *= 2) {
      // full or runs of length width
      for (int i = 0; i < n; i = i + 2 * width) {
        // merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
        bottomUpMerge(sim.gid, A, i, min(i+width, n), min(i+2*width, n), B);
      }
      // swap A and B for the next iteration
      swap(A, B);
      // now A is full of runs of length 2*width
    }
    
    // copy to B just in case it is new_indices array
    // TODO: avoid this copy?
    for (int i = 0; i < n; i++)
      B[i] = A[i];
  }
}
#endif

__global__ void ShuffleAtomsData(SimGpu sim, int nLocalBoxes, int nTotalBoxes, int *boundary_cells, int nBoundaryCells, int *new_indices)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
  __shared__ volatile real_t shfl_mem[THREAD_ATOM_CTA];		// assuming block size = THREAD_ATOM_CTA
#endif
 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int iBox;
  if (warp_id >= nBoundaryCells) iBox = nLocalBoxes + warp_id - nBoundaryCells;
    else iBox = boundary_cells[warp_id];
  int iAtom = lane_id;

  if (iBox >= nTotalBoxes || new_indices[iBox * N_MAX_ATOMS] < 0) return;

  int iOff = iBox * N_MAX_ATOMS + iAtom;

  int id;
  real_t rx, ry, rz, px, py, pz;

  // load into regs
  if (iAtom < sim.num_atoms[iBox]) {
    id = sim.species_ids[iOff];
    rx = sim.r.x[iOff];
    ry = sim.r.y[iOff];
    rz = sim.r.z[iOff];
    px = sim.p.x[iOff];
    py = sim.p.y[iOff];
    pz = sim.p.z[iOff];
  }

  int idx;
  if (iAtom < sim.num_atoms[iBox]) 
    idx = new_indices[iOff] - iBox * N_MAX_ATOMS;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
  if (iAtom < sim.num_atoms[iBox])
    sim.species_ids[iOff] = __shfl(id, idx, (int*)shfl_mem);
  __syncthreads();
  if (iAtom < sim.num_atoms[iBox]) {
    sim.r.x[iOff] = __shfl(rx, idx, shfl_mem);
    sim.r.y[iOff] = __shfl(ry, idx, shfl_mem);
    sim.r.z[iOff] = __shfl(rz, idx, shfl_mem);
    sim.p.x[iOff] = __shfl(px, idx, shfl_mem);
    sim.p.y[iOff] = __shfl(py, idx, shfl_mem);
    sim.p.z[iOff] = __shfl(pz, idx, shfl_mem);
  }
#else
  if (iAtom < sim.num_atoms[iBox]) {
    sim.species_ids[iOff] = __shfl(id, idx);
    sim.r.x[iOff] = __shfl(rx, idx);
    sim.r.y[iOff] = __shfl(ry, idx);
    sim.r.z[iOff] = __shfl(rz, idx);
    sim.p.x[iOff] = __shfl(px, idx);
    sim.p.y[iOff] = __shfl(py, idx);
    sim.p.z[iOff] = __shfl(pz, idx);
  }
#endif
}
