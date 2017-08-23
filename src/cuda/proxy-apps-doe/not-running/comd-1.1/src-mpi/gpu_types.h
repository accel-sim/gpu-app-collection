#pragma once

#include "mytype.h"

// TODO: we can change that to 16 since # of max atoms is always less than 15
#define N_MAX_ATOMS 	 64
#define N_MAX_NEIGHBORS  27

typedef struct vec_t {
  real_t* x;
  real_t* y;
  real_t* z;
} vec_t;

typedef struct real3_t {
  real_t x;
  real_t y;
  real_t z;
} real3_t;

typedef struct int3_t {
  int x;
  int y;
  int z;
} int3_t;

typedef struct LjPotentialGpuSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t sigma;
   real_t epsilon;

} LjPotentialGpu;

typedef struct InterpolationObjectGpuSt
{
   int n;          //!< the number of values in the table
   real_t x0;      //!< the starting ordinate range
   real_t xn;      //!< the ending ordinate range
   real_t invDx;   //!< the inverse of the table spacing
   real_t* values; //!< the abscissa values

} InterpolationObjectGpu;

typedef struct EamPotentialGpuSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms

   InterpolationObjectGpu phi;  //!< Pair energy
   InterpolationObjectGpu rho;  //!< Electron Density
   InterpolationObjectGpu f;    //!< Embedding Energy

   real_t* rhobar;        //!< per atom storage for rhobar
   real_t* dfEmbed;       //!< per atom storage for derivative of Embedding

} EamPotentialGpu;

typedef struct LinkCellGpuSt
{
  // # of local/total boxes
  int n_local;
  int n_total;

  int3_t gridSize;

  real3_t localMin;
  real3_t localMax;
  real3_t invBoxSize;

} LinkCellGpu;

// compacted list of atoms & corresponding cells
typedef struct AtomList
{
  int n;
  int *atoms;
  int *cells;
} AtomListGpu;

typedef struct SimGpuSt {
  int n_local_atoms;		// number of local atoms
  int n_local_cells;		// number of local cells
  int max_atoms_cell;		// max atoms per cell (usually < 32)

  // atoms data
  vec_t r;			// atoms positions
  vec_t p;			// atoms momentum
  vec_t f;			// atoms forces
  real_t *e;			// atoms energies
  int *species_ids;		// atoms species id
  int *gid;			// atoms global id

  // species data
  real_t *species_mass;		// masses of species

  int *neighbor_cells;		// neighbor cells indices
  int *neighbor_atoms;		// neighbor atom offsets 
  int *num_neigh_atoms;		// number of neighbor atoms per cell
  int *num_atoms;		// number of atoms per cell

  int *cell_type;		// type of cell: 0 - interior, 1 - boundary

  AtomListGpu a_list;		// all local cells
  AtomListGpu i_list;		// interior cells
  AtomListGpu b_list;		// boundary cells

  // potentials
  LjPotentialGpu lj_pot;
  EamPotentialGpu eam_pot;

  // method for EAM: 
  // 0 - thread per atom
  // 1 - warp per atom (Kepler only)
  int method;
} SimGpu;

