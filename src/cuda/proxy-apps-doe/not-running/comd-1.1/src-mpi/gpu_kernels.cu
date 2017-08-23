#include <stdio.h>

#include "CoMDTypes.h"
#include "haloExchange.h"

#include "gpu_types.h"

#include "gpu_common.h"
#include "gpu_redistribute.h"

#include "gpu_scan.h"
#include "gpu_reduce.h"

#include "gpu_lj_thread_atom.h"
#include "gpu_eam_thread_atom.h"
#include "gpu_eam_thread_atom_warp_sync.h"
#include "gpu_eam_warp_atom.h"

#ifndef ECX_TARGET
#include "gpu_eam_cta_cell.h"
#endif

#include "gpu_timestep.h"

extern "C"
void ljForceGpu(SimGpu sim)
{
#ifndef ECX_TARGET
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
  int grid = (sim.n_local_atoms + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  LJ_Force_thread_atom<<<grid, block>>>(sim, sim.a_list);
}

template<int step>
int compute_eam_smem_size(SimGpu sim)
{
  int smem = 0;

  // neighbors data
  // positions
  smem += 3 * sizeof(real_t) * CTA_CELL_CTA;

  // embed force
  if (step == 3)
    smem += sizeof(real_t) * CTA_CELL_CTA;

  // local data
  // forces
  smem += 3 * sim.max_atoms_cell * sizeof(real_t);

  // positions
  smem += 3 * sim.max_atoms_cell * sizeof(real_t);

  // ie, irho
  if (step == 1) 
    smem += 2 * sim.max_atoms_cell * sizeof(real_t);

  // local neighbor list
  smem += (CTA_CELL_CTA / WARP_SIZE) * 64 * sizeof(char);

//  return smem;
  return 0;
}

template<int step>
void eamForce(SimGpu sim, AtomListGpu list, cudaStream_t stream = NULL)
{
  if (sim.method == 0) { 
    int grid = (list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
    int block = THREAD_ATOM_CTA;
    EAM_Force_thread_atom<step><<<grid, block, 0, stream>>>(sim, list);
  } 
  else if (sim.method == 1) { 
    int grid = (list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
    int block = THREAD_ATOM_CTA;
    EAM_Force_thread_atom_warp_sync<step><<<grid, block, 0, stream>>>(sim, list);
  }
  else if (sim.method == 2) {
    int block = WARP_ATOM_CTA;
    int grid = (list.n + (WARP_ATOM_CTA/WARP_SIZE)-1)/ (WARP_ATOM_CTA/WARP_SIZE);
    EAM_Force_warp_atom<step><<<grid, block, 0, stream>>>(sim, list);
  } 
  else if (sim.method == 3) {
    // doesn't work in ECX
#ifndef ECX_TARGET
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); // necessary for good occupancy
    int block = CTA_CELL_CTA;
    int grid = sim.n_local_cells;
    EAM_Force_cta_cell<step><<<grid, block, compute_eam_smem_size<step>(sim), stream>>>(sim);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
  }
}

extern "C"
void eamForce1Gpu(SimGpu sim)
{
#ifndef ECX_TARGET
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
  eamForce<1>(sim, sim.a_list);
}

// async launch, latency hiding opt
extern "C" 
void eamForce1GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream)
{
#ifndef ECX_TARGET
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
  eamForce<1>(sim, list, stream);
}

extern "C"
void eamForce2Gpu(SimGpu sim)
{
  int grid = (sim.n_local_atoms + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  EAM_Force_thread_atom<2><<<grid, block>>>(sim, sim.a_list);
}

extern "C"
void eamForce2GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream)
{
  int grid = (sim.n_local_atoms + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  EAM_Force_thread_atom<2><<<grid, block, 0, stream>>>(sim, list);
}

extern "C"
void eamForce3Gpu(SimGpu sim)
{
  eamForce<3>(sim, sim.a_list);
}

extern "C" 
void eamForce3GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream)
{
  eamForce<3>(sim, list, stream);
}

extern "C"
void advanceVelocity(SimGpu sim, real_t dt)
{
  int grid = (sim.n_local_atoms + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  AdvanceVelocity<<<grid, block>>>(sim, dt);
}

extern "C"
void advancePosition(SimGpu sim, real_t dt)
{
  int grid = (sim.n_local_atoms + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  AdvancePosition<<<grid, block>>>(sim, dt);
}

extern "C"
void updateNeighborsGpuAsync(SimGpu sim, int *temp, int nCells, int *cellList, cudaStream_t stream)
{
  // update # of neighbor atoms per cell - 1 thread per cell
  int block = THREAD_ATOM_CTA;
  int grid = (nCells + (block-1))/ block;
  UpdateNeighborNumAtoms<<<grid, block, 0, stream>>>(sim, nCells, cellList, temp);

  // update atom indices - 1 CTA per cell
  grid = nCells;
  UpdateNeighborAtomIndices<<<grid, block, 0, stream>>>(sim, nCells, cellList, temp);
}

extern "C"
void updateNeighborsGpu(SimGpu sim, int *temp)
{
  // update # of neighbor atoms per cell - 1 thread per cell
  int block = THREAD_ATOM_CTA;
  int grid = (sim.n_local_cells + (block-1))/ block;
  UpdateNeighborNumAtoms<<<grid, block>>>(sim, sim.n_local_cells, NULL, temp);

  // update atom indices - 1 CTA per cell
  grid = sim.n_local_cells;
  UpdateNeighborAtomIndices<<<grid, block>>>(sim, sim.n_local_cells, NULL, temp);
}

void setup_gpu_cells(SimFlat *flat, LinkCellGpu *gpu_cells)
{
  gpu_cells->n_total = flat->boxes->nTotalBoxes;
  gpu_cells->n_local = flat->boxes->nLocalBoxes;

  gpu_cells->gridSize.x = flat->boxes->gridSize[0];
  gpu_cells->gridSize.y = flat->boxes->gridSize[1];
  gpu_cells->gridSize.z = flat->boxes->gridSize[2];

  gpu_cells->localMin.x = flat->boxes->localMin[0];
  gpu_cells->localMin.y = flat->boxes->localMin[1];
  gpu_cells->localMin.z = flat->boxes->localMin[2];

  gpu_cells->localMax.x = flat->boxes->localMax[0];
  gpu_cells->localMax.y = flat->boxes->localMax[1];
  gpu_cells->localMax.z = flat->boxes->localMax[2];

  gpu_cells->invBoxSize.x = flat->boxes->invBoxSize[0];
  gpu_cells->invBoxSize.y = flat->boxes->invBoxSize[1];
  gpu_cells->invBoxSize.z = flat->boxes->invBoxSize[2];
}

__global__ void fill(int *natoms_buf, int nCells, int *cellList, int *num_atoms)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nCells)
    natoms_buf[tid] = num_atoms[cellList[tid]];
  else if (tid == nCells)
    natoms_buf[tid] = 0;
}

__global__ void fill(int *natoms_buf, int nCells, int *num_atoms)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nCells)
    natoms_buf[tid] = num_atoms[tid];
  else if (tid == nCells)
    natoms_buf[tid] = 0;
}

void scanCells(int *natoms_buf, int nCells, int *cellList, int *num_atoms, int *partial_sums, cudaStream_t stream = NULL)
{
  // natoms[i] = num_atoms[cellList[i]]
  int block = THREAD_ATOM_CTA;
  int grid = (nCells + 1 + block-1) / block;
  fill<<<grid, block, 0, stream>>>(natoms_buf, nCells, cellList, num_atoms);

  // scan to compute linear index
  scan(natoms_buf, nCells + 1, partial_sums, stream);
}

void scanCells(int *natoms_buf, int nCells, int *num_atoms, int *partial_sums, cudaStream_t stream = NULL)
{
  // natoms[i] = num_atoms[i]
  int block = THREAD_ATOM_CTA;
  int grid = (nCells + 1 + block-1) / block;
  fill<<<grid, block, 0, stream>>>(natoms_buf, nCells, num_atoms);

  // scan to compute linear index
  scan(natoms_buf, nCells + 1, partial_sums, stream);
}

void BuildAtomLists(SimFlat *s)
{
  int nCells = s->boxes->nLocalBoxes;
  int n_interior_cells = s->boxes->nLocalBoxes - s->n_boundary_cells;

  int size = nCells+1;
  if (size % 256 != 0) size = ((size + 255)/256)*256;

  int *cell_offsets1;
  int *cell_offsets2;
  cudaMalloc(&cell_offsets1, size * sizeof(int));
  cudaMalloc(&cell_offsets2, size * sizeof(int));
  int *partial_sums;
  cudaMalloc(&partial_sums, size * sizeof(int));

  scanCells(cell_offsets1, nCells, s->gpu.num_atoms, partial_sums);

  int block = THREAD_ATOM_CTA;
  int grid = (nCells + (block/WARP_SIZE)-1)/(block/WARP_SIZE);
  UpdateAtomList<<<grid, block>>>(s->gpu, s->gpu.a_list, nCells, cell_offsets1);   

  // build interior & boundary lists
  scanCells(cell_offsets1, s->n_boundary_cells, s->boundary_cells, s->gpu.num_atoms, partial_sums);
  scanCells(cell_offsets2, n_interior_cells, s->interior_cells, s->gpu.num_atoms, partial_sums);

  grid = (s->n_boundary_cells + (block/WARP_SIZE)-1)/(block/WARP_SIZE);
  UpdateBoundaryList<<<grid, block>>>(s->gpu, s->gpu.b_list, s->n_boundary_cells, cell_offsets1, s->boundary_cells);   

  grid = (n_interior_cells + (block/WARP_SIZE)-1)/(block/WARP_SIZE);
  UpdateBoundaryList<<<grid, block>>>(s->gpu, s->gpu.i_list, n_interior_cells, cell_offsets2, s->interior_cells);   

  cudaMemcpy(&s->gpu.b_list.n, cell_offsets1 + s->n_boundary_cells, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s->gpu.i_list.n, cell_offsets2 + n_interior_cells, sizeof(int), cudaMemcpyDeviceToHost);
}

/// \details
/// This is the first step in returning data structures to a consistent
/// state after the atoms move each time step.  First we discard all
/// atoms in the halo link cells.  These are all atoms that are
/// currently stored on other ranks and so any information we have about
/// them is stale.  Next, we move any atoms that have crossed link cell
/// boundaries into their new link cells.  It is likely that some atoms
/// will be moved into halo link cells.  Since we have deleted halo
/// atoms from other tasks, it is clear that any atoms that are in halo
/// cells at the end of this routine have just transitioned from local
/// to halo atoms.  Such atom must be sent to other tasks by a halo
/// exchange to avoid being lost.
/// \see redistributeAtoms
extern "C"
void updateLinkCells(SimFlat *flat)
{
  int *flags = flat->flags;

  // setup link cells
  LinkCellGpu gpu_cells;
  setup_gpu_cells(flat, &gpu_cells);

  // empty halo cells
  cudaMemset(flat->gpu.num_atoms + gpu_cells.n_local, 0, (gpu_cells.n_total - gpu_cells.n_local) * sizeof(int));

  // set all flags to 0
  cudaMemset(flags, 0, gpu_cells.n_total * N_MAX_ATOMS * sizeof(int));
 
  // 1 thread updates 1 atom
  int grid = (flat->gpu.n_local_atoms + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  UpdateLinkCells<<<grid, block>>>(flat->gpu, gpu_cells, flags);

  // 1 thread updates 1 cell
  grid = (gpu_cells.n_local + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  block = THREAD_ATOM_CTA;
  CompactAtoms<<<grid, block>>>(flat->gpu, gpu_cells.n_local, flags);

  // build new lists
  BuildAtomLists(flat);
}

/// The loadBuffer function for a halo exchange of atom data.  Iterates
/// link cells in the cellList and load any atoms into the send buffer.
/// This function also shifts coordinates of the atoms by an appropriate
/// factor if they are being sent across a periodic boundary.
extern "C" 
void loadAtomsBufferFromGpu(char *buf, int *nbuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, real_t *shift, SimFlat *s, char *gpu_buf, cudaStream_t stream)
{
  scanCells(natoms_buf, nCells, cellList, s->gpu.num_atoms, partial_sums, stream);

  // copy data to compacted array
  int block = N_MAX_ATOMS;
  int grid = nCells;
  LoadAtomsBufferPacked<<<grid, block, 0, stream>>>(gpu_buf, cellList, nCells, s->gpu, natoms_buf, shift[0], shift[1], shift[2]);

  int nBuf;
  cudaMemcpyAsync(&nBuf, natoms_buf + nCells, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(buf, gpu_buf, nBuf * sizeof(AtomMsg), cudaMemcpyDeviceToHost, stream);
  
  // make sure we copied the buffer to the host
  cudaStreamSynchronize(stream);
  *nbuf = nBuf;
}

/// The unloadBuffer function for a halo exchange of atom data.
/// Iterates the receive buffer and places each atom that was received
/// into the link cell that corresponds to the atom coordinate.  Note
/// that this naturally accomplishes transfer of ownership of atoms that
/// have moved from one spatial domain to another.  Atoms with
/// coordinates in local link cells automatically become local
/// particles.  Atoms that are owned by other ranks are automatically
/// placed in halo kink cells.
extern "C"
void unloadAtomsBufferToGpu(char *buf, int nBuf, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream)
{
  // setup link cells
  LinkCellGpu gpu_cells;
  setup_gpu_cells(s, &gpu_cells);

  cudaMemcpyAsync(gpu_buf, buf, nBuf * sizeof(AtomMsg), cudaMemcpyHostToDevice, stream);

  int grid = (nBuf + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  UnloadAtomsBufferPacked<<<grid, block, 0, stream>>>(gpu_buf, nBuf, s->gpu, gpu_cells);

  int nCells = s->boxes->nLocalBoxes;
  scanCells(natoms_buf, nCells, s->gpu.num_atoms, partial_sums, stream);

  // rebuild compact list of atoms & cells
  grid = (nCells * N_MAX_ATOMS + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  UpdateCompactIndices<<<grid, block, 0, stream>>>(natoms_buf, nCells, s->gpu);

  // new number of local atoms
  cudaMemcpyAsync(&s->gpu.n_local_atoms, natoms_buf + nCells, sizeof(int), cudaMemcpyDeviceToHost, stream);
}

/// The loadBuffer function for a force exchange.
/// Iterate the send list and load the derivative of the embedding
/// energy with respect to the local density into the send buffer.
extern "C"
void loadForceBufferFromGpu(char *buf, int *nbuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream)
{
  scanCells(natoms_buf, nCells, cellList, s->gpu.num_atoms, partial_sums, stream);

  // copy data to compacted array
  int grid = (nCells * N_MAX_ATOMS + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  LoadForceBuffer<<<grid, block, 0, stream>>>((ForceMsg*)gpu_buf, nCells, cellList, s->gpu, natoms_buf);

  int nBuf;
  cudaMemcpyAsync(&nBuf, natoms_buf + nCells, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(buf, gpu_buf, nBuf * sizeof(ForceMsg), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
  *nbuf = nBuf;
}

/// The unloadBuffer function for a force exchange.
/// Data is received in an order that naturally aligns with the atom
/// storage so it is simple to put the data where it belongs.
extern "C"
void unloadForceBufferToGpu(char *buf, int nBuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream)
{
  // copy raw data to gpu
  cudaMemcpyAsync(gpu_buf, buf, nBuf * sizeof(ForceMsg), cudaMemcpyHostToDevice, stream);

  scanCells(natoms_buf, nCells, cellList, s->gpu.num_atoms, partial_sums, stream);

  // copy data for the list of cells
  int grid = (nCells * N_MAX_ATOMS + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  UnloadForceBuffer<<<grid, block, 0, stream>>>((ForceMsg*)gpu_buf, nCells, cellList, s->gpu, natoms_buf);
}

extern "C"
void sortAtomsGpu(SimFlat *s, cudaStream_t stream)
{
  int *new_indices = s->flags;
  // set all indices to -1
  cudaMemsetAsync(new_indices, 255, s->boxes->nTotalBoxes * N_MAX_ATOMS * sizeof(int), stream);
  
  // one thread per atom, only update boundary cells
  int block = N_MAX_ATOMS;
  int grid = (s->n_boundary1_cells * WARP_SIZE + block-1)/block;
  SetLinearIndices<<<grid, block, 0, stream>>>(s->gpu, s->n_boundary1_cells, s->boundary1_cells, new_indices);

  // update halo cells
  grid = ((s->boxes->nTotalBoxes - s->boxes->nLocalBoxes) * N_MAX_ATOMS + block-1)/block;
  SetLinearIndices<<<grid, block, 0, stream>>>(s->gpu, s->boxes->nLocalBoxes, s->boxes->nTotalBoxes, new_indices);

  // one thread per cell: process halo & boundary cells only
  int block2 = N_MAX_ATOMS;
  int grid2 = (s->n_boundary1_cells + (s->boxes->nTotalBoxes - s->boxes->nLocalBoxes) + block2-1) / block2;
  SortAtomsByGlobalId<<<grid2, block2, 0, stream>>>(s->gpu, s->boxes->nLocalBoxes, s->boxes->nTotalBoxes, s->boundary1_cells, s->n_boundary1_cells, new_indices, s->tmp_sort);

  // one warp per cell
  int block3 = THREAD_ATOM_CTA;
  int grid3 = ((s->n_boundary1_cells + (s->boxes->nTotalBoxes - s->boxes->nLocalBoxes)) * WARP_SIZE + block3-1) / block3;
  ShuffleAtomsData<<<grid3, block3, 0, stream>>>(s->gpu, s->boxes->nLocalBoxes, s->boxes->nTotalBoxes, s->boundary1_cells, s->n_boundary1_cells, new_indices);
}

extern "C"
void computeEnergy(SimFlat *flat, real_t *eLocal)
{
  real_t *e_gpu;
  cudaMalloc(&e_gpu, 2 * sizeof(real_t));
  cudaMemset(e_gpu, 0, 2 * sizeof(real_t));

  int grid = (flat->gpu.n_local_atoms + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  ReduceEnergy<<<grid, block>>>(flat->gpu, &e_gpu[0], &e_gpu[1]);
  
  cudaMemcpy(eLocal, e_gpu, 2 * sizeof(real_t), cudaMemcpyDeviceToHost);
}
