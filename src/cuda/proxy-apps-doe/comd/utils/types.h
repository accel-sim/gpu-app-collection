#pragma once

#include "interface.h"

#define FMT2 "%lg" /**< \def format argument for real2_t **/
typedef double real2_t; /**< used where converting from double to single -- see domains.c **/
typedef real_t real3[3]; /**< a convenience vector with three real_t **/
typedef real_t real4[4]; /**< a convenience vector with four real_t **/


/**
 * \brief the LJ potential
 *
 * - All potentials are expected to conform to the
 * following units:
 *   -# atomic distances are in Angstroms
 *   -# atomic masses are in AMUs (atomic mass units)
 *   -# forces are returned in hartrees/angstrom
 *   -# energies are computed in hartrees
 *
 * - Lennard-Jones potential adds to these
 *   -# sigma, the location of the energy well
 *   -# epsilon, the depth of the energy well
 *
 * This is a simple 6-12 LJ potential.
 *
 **/
typedef struct ljpotential_t {
  /**<
   **/
  real_t cutoff;         /* cutoff for potential */
  real_t mass;           /* mass of atoms in amu*/
  int (*force)(void *s);
  void (*destroy)(void **pot);
  real_t sigma;
  real_t epsilon;
} ljpotential_t;


/**
 * - All potentials are expected to conform to the
 * following units:
 *   -# atomic distances are in Angstroms
 *   -# atomic masses are in AMUs (atomic mass units)
 *   -# forces are returned in hartrees/angstrom
 *   -# energies are computed in hartrees
 *
 * - EAM potential adds to these:
 *   -# phi, the phi array
 *   -# rho, the rho array
 *   -# f,   the F array.
 *
 * Note that phi, rho, and f are of type potentialarray_t.
 *
 **/
typedef struct eampotential_t {
  real_t cutoff;       /**< the potential cutoff distance **/
  real_t mass;           /**< mass of atoms **/
  int (*force)(void *s); /**< the force function **/
  void (*destroy)(void **pot); /**< the deallocate function **/
  struct potentialarray_t *phi; /**< the phi array **/
  struct potentialarray_t *rho; /**< the rho array **/
  struct potentialarray_t *f;   /**< the F array   **/
  real_t lat;           /**< lattice constant **/
} eampotential_t;

// the Chebychev 'potential' is based on the EAM potential type
typedef struct eam_cheby_t {
  real_t cutoff;       /**< the potential cutoff distance **/
  real_t mass;           /**< mass of atoms **/
  struct potentialarray_t *phi; /**< the phi array **/
  struct potentialarray_t *rho; /**< the rho array **/
  struct potentialarray_t *f;   /**< the F array   **/
  struct potentialarray_t *dphi; /**< the phi array **/
  struct potentialarray_t *drho; /**< the rho array **/
  struct potentialarray_t *df;   /**< the F array   **/
  real_t lat;           /**< lattice constant **/
} eam_cheby_t;


/**
 * \brief the base form off of which all potentials will be set
 *
 * - All potentials are expected to conform to the
 * following units:
 *   -# atomic distances are in Angstroms
 *   -# atomic masses are in AMUs (atomic mass units)
 *   -# forces are returned in hartrees/angstrom
 *   -# energies are computed in hartrees
 * Note that phi, rho, and f are of type potentialarray_t.
 *
 **/
typedef struct pmd_base_potential_t {
  real_t cutoff;         /**< potential cutoff distance in Angstroms **/
  real_t mass;           /**< mass of atoms in atomic mass units **/
  int (*force)(void *s); /**< the actual parameter is "struct simulation_t *s" **/
  void (*destroy)(void *pot); /**< destruction of the potential **/
} pmd_base_potential_t;

typedef struct fileAtom {
  float  x, y, z;       
  float  bond_order;
  float  centrosymmetry;
} FileAtom;

/**
 * The basic flat simulation data structure with MAXATOMS in every box **/
typedef struct simflat_t {
  int stateflag; /**< unused for now **/
  int nbx[3];    /**< number of boxes in each dimension **/
  int nboxes;    /**< total number of boxes **/
  int ntot; /**< total number of atoms**/
  real3 bounds; /**< periodic bounds**/
  real3 boxsize; /**< size of domains**/
  real3 *dcenter; /**< an array that contains the center of each box **/
  int *natoms;    /**< the total number of atoms in the simulation **/
  int *id;     /**< The original ID of the atom  **/
  int *iType;  /**< the type of atoms**/
  real_t *mass; /**< mass of the atoms**/
  real3 *r; /**< positions**/
  real3 *p; /**< momenta of atoms**/
  real4 *f; /**< fx, fy, fz, energy**/
  real_t *rho;   /**< rhosum for EAM potential**/
  real_t *fi;    /**< rhobar for EAM potential**/

#ifdef USE_IN_SITU_VIZ
  real_t *centro;
#endif

  /** the total potential energy of the simulation **/
  real_t e;    /**< the total energy of the system **/
  int nAtoms;  /**< The total number of atoms in the simulation  **/


  pmd_base_potential_t *pot; /**< the potential**/

  char *comment; /**< free form string that describes the simulation **/

// new variables to store meso-micro data. Will be tensors eventually
  real_t stress; /**< virial stress**/
  real_t defgrad; /**< deformation gradient**/
  real_t bf; /**< box factor **/
  eam_cheby_t *ch_pot; /**< Chebychev coefficients**/

} simflat_t; 

typedef struct command_t {
  char filename[1024];/**< the input file name **/
  char potdir[1024];  /**< the directory where eam potentials reside **/
  char potname[1024]; /**< the name of the potential **/
  int debug;          /**< a flag to determine whether to debugt or not **/
  int doeam;          /**< a flag to determine whether we're running EAM potentials **/
  int periodic;       /**< a flag that controls whether periodic boundary conditions are used **/
  int usegpu;         /**< a flag that controls whether OpenCL tries to target a gpu **/
  int nx;             /**< number of unit cells in x **/
  int ny;             /**< number of unit cells in y **/
  int nz;             /**< number of unit cells in z **/
  double bf;          /**< ratio of box size to cutoff radius **/
  double lat;         /**< lattice constant **/
  double defgrad;     /**< deformation gradient **/
  double temp;        /**< simulation temperature **/
  int nsteps;         /**< number of fraction steps per iteration **/
  int niters;         /**< number of iterations **/
  char method[1024];  /**< implementation name **/
  int doref;          /**< a flag to run reference implementation **/
} command_t;
