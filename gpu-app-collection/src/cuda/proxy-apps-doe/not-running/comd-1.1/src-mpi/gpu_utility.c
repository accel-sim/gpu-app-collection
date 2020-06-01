#include "gpu_utility.h"
#include <cuda.h>

// fallback for 5.0
#if (CUDA_VERSION < 5050)
  cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority) {
    printf("WARNING: priority streams are not supported in CUDA 5.0, falling back to regular streams");
    return cudaStreamCreate(stream);
  }
#endif

void SetupGpu(int deviceId)
{
  cudaSetDevice(deviceId);
  
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  printf("Host %s using GPU %i: %s\n\n", hostname, deviceId, props.name);
}

// input is haloExchange structure for forces
// this function sets the following static GPU arrays:
//   gpu.cell_type - 0 if interior, 1 if boundary (assuming 2-rings: corresponding to boundary/interior)
//   n_boundary_cells - number of 2-ring boundary cells
//   n_boundary1_cells - number of immediate boundary cells (1 ring)
//   boundary_cells - list of boundary cells ids (2 rings)
//   interior_cells - list of interior cells ids (w/o 2 rings)
//   boundary1_cells - list of immediate boundary cells ids (1 ring)
// also it creates necessary streams
void SetBoundaryCells(SimFlat *flat, HaloExchange *hh)
{
  int n_all_cells = flat->boxes->nLocalBoxes;
  int *h_boundary_cells = (int*)malloc(n_all_cells * sizeof(int)); 
  int *h_boundary1_cells = (int*)malloc(n_all_cells * sizeof(int)); 
  int *h_cell_type = (int*)malloc(n_all_cells * sizeof(int));
  memset(h_cell_type, 0, n_all_cells * sizeof(int));

  // gather data to a single list, set cell type
  int n = 0;
  ForceExchangeParms *parms = (ForceExchangeParms*)hh->parms;
  for (int ii=0; ii<6; ++ii) {
    int *cellList = parms->sendCells[ii];               
    for (int j = 0; j < parms->nCells[ii]; j++) 
      if (cellList[j] < n_all_cells && h_cell_type[cellList[j]] == 0) {
	h_boundary1_cells[n] = cellList[j];
	h_boundary_cells[n] = cellList[j];
	h_cell_type[cellList[j]] = 1;
        n++;
      }
  }

  flat->n_boundary1_cells = n;
  int n_boundary1_cells = n;

  // find 2nd ring
  int neighbor_cells[N_MAX_NEIGHBORS];
  for (int i = 0; i < n_all_cells; i++)
    if (h_cell_type[i] == 0) {
      getNeighborBoxes(flat->boxes, i, neighbor_cells);
      for (int j = 0; j < N_MAX_NEIGHBORS; j++)
        if (h_cell_type[neighbor_cells[j]] == 1) {  
          // found connection to the boundary node - add to the list
          h_boundary_cells[n] = i;
          h_cell_type[i] = 2;
          n++;
          break;
        }
    }

  flat->n_boundary_cells = n;
  int n_boundary_cells = n;

  int n_interior_cells = flat->boxes->nLocalBoxes - n;

  // find interior cells
  int *h_interior_cells = (int*)malloc(n_interior_cells * sizeof(int));
  n = 0;
  for (int i = 0; i < n_all_cells; i++) {
    if (h_cell_type[i] == 0) {
      h_interior_cells[n] = i;
      n++;
    }
    else if (h_cell_type[i] == 2) {
      h_cell_type[i] = 1;
    }
  }

  // allocate on GPU
  cudaMalloc((void**)&flat->boundary1_cells, n_boundary1_cells * sizeof(int));
  cudaMalloc((void**)&flat->boundary_cells, n_boundary_cells * sizeof(int));
  cudaMalloc((void**)&flat->interior_cells, n_interior_cells * sizeof(int));

  // copy to GPU  
  cudaMemcpy(flat->boundary1_cells, h_boundary1_cells, n_boundary1_cells * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(flat->boundary_cells, h_boundary_cells, n_boundary_cells * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(flat->interior_cells, h_interior_cells, n_interior_cells * sizeof(int), cudaMemcpyHostToDevice);

  // set cell types
  cudaMalloc((void**)&flat->gpu.cell_type, n_all_cells * sizeof(int));
  cudaMemcpy(flat->gpu.cell_type, h_cell_type, n_all_cells * sizeof(int), cudaMemcpyHostToDevice);

  if (flat->gpuAsync) {
    // create priority & normal streams
    cudaStreamCreateWithPriority(&flat->boundary_stream, 0, -1);	// set higher priority
    cudaStreamCreate(&flat->interior_stream);
  }
  else {
    // set streams to NULL
    flat->interior_stream = NULL;
    flat->boundary_stream = NULL;
  }
}

void AllocateGpu(SimFlat *flat, int do_eam, char *method)
{
  int deviceId;
  struct cudaDeviceProp props;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);

  SimGpu *gpu = &flat->gpu;
  if (!strcmp(method, "thread_atom")) gpu->method = 0;
  if (!strcmp(method, "thread_atom_warp_sync")) gpu->method = 1;
  if (!strcmp(method, "warp_atom")) gpu->method = 2;
  if (!strcmp(method, "cta_cell")) gpu->method = 3;

  int total_boxes = flat->boxes->nTotalBoxes;
  int local_boxes = flat->boxes->nLocalBoxes;
  int num_species = 1;

  // allocate positions, momentum, forces & energies
  int r_size = total_boxes * N_MAX_ATOMS * sizeof(real_t);
  int f_size = local_boxes * N_MAX_ATOMS * sizeof(real_t);

  cudaMalloc((void**)&gpu->r.x, r_size);
  cudaMalloc((void**)&gpu->r.y, r_size);
  cudaMalloc((void**)&gpu->r.z, r_size);  

  cudaMalloc((void**)&gpu->p.x, r_size);
  cudaMalloc((void**)&gpu->p.y, r_size);
  cudaMalloc((void**)&gpu->p.z, r_size);

  cudaMalloc((void**)&gpu->f.x, f_size);
  cudaMalloc((void**)&gpu->f.y, f_size);
  cudaMalloc((void**)&gpu->f.z, f_size);

  cudaMalloc((void**)&gpu->e, f_size);

  cudaMalloc((void**)&gpu->gid, total_boxes * N_MAX_ATOMS * sizeof(int));

  // species data
  cudaMalloc((void**)&gpu->species_ids, total_boxes * N_MAX_ATOMS * sizeof(int));
  cudaMalloc((void**)&gpu->species_mass, num_species * sizeof(real_t));

  // allocate indices, neighbors, etc.
  cudaMalloc((void**)&gpu->neighbor_cells, local_boxes * N_MAX_NEIGHBORS * sizeof(int));
  cudaMalloc((void**)&gpu->neighbor_atoms, local_boxes * N_MAX_NEIGHBORS * N_MAX_ATOMS * sizeof(int));
  cudaMalloc((void**)&gpu->num_neigh_atoms, local_boxes * sizeof(int));
  cudaMalloc((void**)&gpu->num_atoms, total_boxes * sizeof(int));

  // total # of atoms in local boxes
  int n = 0;
  for (int iBox=0; iBox < flat->boxes->nLocalBoxes; iBox++)
    n += flat->boxes->nAtoms[iBox];
  gpu->a_list.n = n;
  cudaMalloc((void**)&gpu->a_list.atoms, n * sizeof(int));
  cudaMalloc((void**)&gpu->a_list.cells, n * sizeof(int));

  // allocate other lists as well
  cudaMalloc((void**)&gpu->i_list.atoms, n * sizeof(int));
  cudaMalloc((void**)&gpu->i_list.cells, n * sizeof(int));
  cudaMalloc((void**)&gpu->b_list.atoms, n * sizeof(int));
  cudaMalloc((void**)&gpu->b_list.cells, n * sizeof(int));

  // init EAM arrays
  if (do_eam)  
  {
    EamPotential* pot = (EamPotential*) flat->pot;

    cudaMalloc((void**)&gpu->eam_pot.f.values, (pot->f->n+3) * sizeof(real_t));
    cudaMalloc((void**)&gpu->eam_pot.rho.values, (pot->rho->n+3) * sizeof(real_t));
    cudaMalloc((void**)&gpu->eam_pot.phi.values, (pot->phi->n+3) * sizeof(real_t));

    cudaMalloc((void**)&gpu->eam_pot.dfEmbed, r_size);
    cudaMalloc((void**)&gpu->eam_pot.rhobar, r_size);
  }

  // initialize host data as well
  SimGpu *host = &flat->host;

  host->r.x = (real_t*)malloc(r_size);
  host->r.y = (real_t*)malloc(r_size);
  host->r.z = (real_t*)malloc(r_size);

  host->p.x = (real_t*)malloc(r_size);
  host->p.y = (real_t*)malloc(r_size);
  host->p.z = (real_t*)malloc(r_size);

  host->f.x = (real_t*)malloc(f_size);
  host->f.y = (real_t*)malloc(f_size);
  host->f.z = (real_t*)malloc(f_size);

  host->e = (real_t*)malloc(f_size);
  
  host->neighbor_cells = (int*)malloc(local_boxes * N_MAX_NEIGHBORS * sizeof(int));
  host->neighbor_atoms = (int*)malloc(local_boxes * N_MAX_NEIGHBORS * N_MAX_ATOMS * sizeof(int));
  host->num_neigh_atoms = (int*)malloc(local_boxes * sizeof(int));
  host->num_atoms = (int*)malloc(total_boxes * sizeof(int));

  // on host allocate list of all local atoms only
  host->a_list.atoms = (int*)malloc(n * sizeof(int));
  host->a_list.cells = (int*)malloc(n * sizeof(int));

  // temp arrays
  cudaMalloc((void**)&flat->flags, flat->boxes->nTotalBoxes * N_MAX_ATOMS * sizeof(int));
  cudaMalloc((void**)&flat->tmp_sort, flat->boxes->nTotalBoxes * N_MAX_ATOMS * sizeof(int));
  cudaMalloc((void**)&flat->gpu_atoms_buf, flat->boxes->nTotalBoxes * N_MAX_ATOMS * sizeof(AtomMsg));
  cudaMalloc((void**)&flat->gpu_force_buf, flat->boxes->nTotalBoxes * N_MAX_ATOMS * sizeof(ForceMsg));
}

void DestroyGpu(SimFlat *flat)
{
  SimGpu *gpu = &flat->gpu;
  SimGpu *host = &flat->host;

  cudaFree(gpu->r.x);
  cudaFree(gpu->r.y);
  cudaFree(gpu->r.z);

  cudaFree(gpu->p.x);
  cudaFree(gpu->p.y);
  cudaFree(gpu->p.z);

  cudaFree(gpu->f.x);
  cudaFree(gpu->f.y);
  cudaFree(gpu->f.z);

  cudaFree(gpu->e);

  cudaFree(gpu->gid);

  cudaFree(gpu->species_ids);
  cudaFree(gpu->species_mass);

  cudaFree(gpu->neighbor_cells);
  cudaFree(gpu->neighbor_atoms);
  cudaFree(gpu->num_neigh_atoms);
  cudaFree(gpu->num_atoms);

  cudaFree(gpu->a_list.atoms);
  cudaFree(gpu->a_list.cells);

  cudaFree(gpu->i_list.atoms);
  cudaFree(gpu->i_list.cells);

  cudaFree(gpu->b_list.atoms);
  cudaFree(gpu->b_list.cells);

  cudaFree(flat->flags);
  cudaFree(flat->tmp_sort);
  cudaFree(flat->gpu_atoms_buf);
  cudaFree(flat->gpu_force_buf);

  if (gpu->eam_pot.f.values) cudaFree(gpu->eam_pot.f.values);
  if (gpu->eam_pot.rho.values) cudaFree(gpu->eam_pot.rho.values);
  if (gpu->eam_pot.phi.values) cudaFree(gpu->eam_pot.phi.values);

  if (gpu->eam_pot.dfEmbed) cudaFree(gpu->eam_pot.dfEmbed);
  if (gpu->eam_pot.rhobar) cudaFree(gpu->eam_pot.rhobar);

  free(host->r.x);
  free(host->r.y);
  free(host->r.z);

  free(host->p.x);
  free(host->p.y);
  free(host->p.z);

  free(host->f.x);
  free(host->f.y);
  free(host->f.z);

  free(host->e);

  free(host->species_mass);

  free(host->neighbor_cells);
  free(host->neighbor_atoms);
  free(host->num_neigh_atoms);
  free(host->num_atoms);

  free(host->a_list.atoms);
  free(host->a_list.cells);
}

void CopyDataToGpu(SimFlat *flat, int do_eam)
{
  SimGpu *gpu = &flat->gpu;
  SimGpu *host = &flat->host;

  // set potential
  if (do_eam) 
  {
    EamPotential* pot = (EamPotential*) flat->pot;
    gpu->eam_pot.cutoff = pot->cutoff;

    gpu->eam_pot.f.n = pot->f->n;
    gpu->eam_pot.rho.n = pot->rho->n;
    gpu->eam_pot.phi.n = pot->phi->n;

    gpu->eam_pot.f.x0 = pot->f->x0;
    gpu->eam_pot.rho.x0 = pot->rho->x0;
    gpu->eam_pot.phi.x0 = pot->phi->x0;

    gpu->eam_pot.f.xn = pot->f->x0 + pot->f->n / pot->f->invDx;
    gpu->eam_pot.rho.xn = pot->rho->x0 + pot->rho->n / pot->rho->invDx;
    gpu->eam_pot.phi.xn = pot->phi->x0 + pot->phi->n / pot->phi->invDx;

    gpu->eam_pot.f.invDx = pot->f->invDx;
    gpu->eam_pot.rho.invDx = pot->rho->invDx;
    gpu->eam_pot.phi.invDx = pot->phi->invDx;

    cudaMemcpy(gpu->eam_pot.f.values, pot->f->values-1, (pot->f->n+3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->eam_pot.rho.values, pot->rho->values-1, (pot->rho->n+3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->eam_pot.phi.values, pot->phi->values-1, (pot->phi->n+3) * sizeof(real_t), cudaMemcpyHostToDevice);
  }
  else
  {
    LjPotential* pot = (LjPotential*)flat->pot;
    gpu->lj_pot.sigma = pot->sigma;
    gpu->lj_pot.cutoff = pot->cutoff;
    gpu->lj_pot.epsilon = pot->epsilon;
  }

  int total_boxes = flat->boxes->nTotalBoxes;
  int local_boxes = flat->boxes->nLocalBoxes;
  int r_size = total_boxes * N_MAX_ATOMS * sizeof(real_t);
  int f_size = local_boxes * N_MAX_ATOMS * sizeof(real_t);
  int num_species = 1;

  // copy positions for all boxes (local & halo)
  for (int iBox=0; iBox < total_boxes; iBox++) {
    int nIBox = flat->boxes->nAtoms[iBox];
    if (nIBox == 0) continue;
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {
      host->r.x[iOff] = flat->atoms->r[iOff][0];
      host->r.y[iOff] = flat->atoms->r[iOff][1];
      host->r.z[iOff] = flat->atoms->r[iOff][2];
      
//      if (iBox < flat->boxes->nLocalBoxes) {
        host->p.x[iOff] = flat->atoms->p[iOff][0];
        host->p.y[iOff] = flat->atoms->p[iOff][1];
        host->p.z[iOff] = flat->atoms->p[iOff][2];
//      }
    }
  }

  // copy neighbors for local boxes and num atoms for all boxes
  for (int iBox=0; iBox < total_boxes; iBox++)
    host->num_atoms[iBox] = flat->boxes->nAtoms[iBox];
  for (int iBox=0; iBox < local_boxes; iBox++) {
    getNeighborBoxes(flat->boxes, iBox, host->neighbor_cells + iBox * N_MAX_NEIGHBORS);

    // find itself and put first
    for (int j = 0; j < N_MAX_NEIGHBORS; j++)
      if (host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j] == iBox) {
        int q = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
	host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j] = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + 0];
        host->neighbor_cells[iBox * N_MAX_NEIGHBORS + 0] = q;
        break;
      }
  }

  // prepare neighbor list
  for (int iBox=0; iBox < local_boxes; iBox++) {
    int num_neigh_atoms = 0;
    for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
      int jBox = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
      for (int k = 0; k < flat->boxes->nAtoms[jBox]; k++) {
        host->neighbor_atoms[iBox * N_MAX_NEIGHBORS * N_MAX_ATOMS + num_neigh_atoms] = jBox * N_MAX_ATOMS + k;
        num_neigh_atoms++;
      }
    }
    host->num_neigh_atoms[iBox] = num_neigh_atoms;
  }

  // compute total # of atoms in local boxes
  int n_total = 0;
  for (int iBox=0; iBox < flat->boxes->nLocalBoxes; iBox++) 
    n_total += flat->boxes->nAtoms[iBox];
  gpu->a_list.n = n_total;
  gpu->n_local_atoms = n_total;
  gpu->n_local_cells = flat->boxes->nLocalBoxes;
  gpu->max_atoms_cell = 32;

  // compute and copy compact list of all atoms/cells
  int cur = 0;
  for (int iBox=0; iBox < flat->boxes->nLocalBoxes; iBox++) {
    int nIBox = flat->boxes->nAtoms[iBox];
    if (nIBox == 0) continue;
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {
      host->a_list.atoms[cur] = ii;
      host->a_list.cells[cur] = iBox;
      cur++;
    }
  }

  // initialize species
  host->species_mass = (real_t*)malloc(num_species * sizeof(real_t));
  for (int i = 0; i < num_species; i++)
    host->species_mass[i] = flat->species[i].mass;

  // copy all data to gpus
  cudaMemcpy(gpu->r.x, host->r.x, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->r.y, host->r.y, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->r.z, host->r.z, r_size, cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->p.x, host->p.x, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->p.y, host->p.y, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->p.z, host->p.z, r_size, cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->species_ids, flat->atoms->iSpecies, local_boxes * N_MAX_ATOMS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->species_mass, host->species_mass, num_species * sizeof(real_t), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->neighbor_cells, host->neighbor_cells, local_boxes * N_MAX_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->neighbor_atoms, host->neighbor_atoms, local_boxes * N_MAX_NEIGHBORS * N_MAX_ATOMS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->num_neigh_atoms, host->num_neigh_atoms, local_boxes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->num_atoms, host->num_atoms, total_boxes * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->a_list.atoms, host->a_list.atoms, n_total * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->a_list.cells, host->a_list.cells, n_total * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->gid, flat->atoms->gid, total_boxes * N_MAX_ATOMS * sizeof(int), cudaMemcpyHostToDevice);
}

void GetDataFromGpu(SimFlat *flat)
{
  SimGpu *gpu = &flat->gpu;
  SimGpu *host = &flat->host;

  // copy back forces & energies
  int f_size = flat->boxes->nLocalBoxes * N_MAX_ATOMS * sizeof(real_t);

  // update num atoms
  cudaMemcpy(flat->boxes->nAtoms, gpu->num_atoms, flat->boxes->nTotalBoxes * sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(host->p.x, gpu->p.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(host->p.y, gpu->p.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(host->p.z, gpu->p.z, f_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(host->f.x, gpu->f.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(host->f.y, gpu->f.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(host->f.z, gpu->f.z, f_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(host->e, gpu->e, f_size, cudaMemcpyDeviceToHost);
 
  // assign energy and forces
  // compute total energy
  flat->ePotential = 0.0;
  for (int iBox=0; iBox < flat->boxes->nLocalBoxes; iBox++) {
    int nIBox = flat->boxes->nAtoms[iBox];
    if (nIBox == 0) continue;
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {
      flat->atoms->p[iOff][0] = host->p.x[iOff];
      flat->atoms->p[iOff][1] = host->p.y[iOff];
      flat->atoms->p[iOff][2] = host->p.z[iOff];

      flat->atoms->f[iOff][0] = host->f.x[iOff];
      flat->atoms->f[iOff][1] = host->f.y[iOff];
      flat->atoms->f[iOff][2] = host->f.z[iOff];

      flat->atoms->U[iOff] = host->e[iOff]; 

      flat->ePotential += host->e[iOff];
    }
  }
}

