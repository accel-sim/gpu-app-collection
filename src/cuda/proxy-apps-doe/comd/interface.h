#pragma once

#include <string>

#define FMT2 "%lg" /**< \def format argument for real2_t **/

#ifdef DOUBLE
  typedef double real_t;
  #define FMT1 "%lg" /**< \def format argument for doubles **/
  #define EMT1 "%le" /**< \def format argument for eng doubles **/
#else
  typedef float real_t;
  #define FMT1 "%g" /**< /def format argument for floats **/
  #define EMT1 "%e" /**< /def format argument for eng floats **/
#endif

// close upper limit 
#define N_MAX_ATOMS 	 16
#define N_MAX_NEIGHBORS  27
#define PERIODIC 	 1

/**
 * the velocity conversion from A/second to bohr/atu  **/
static const real_t bohr_per_atu_to_A_per_s = (real_t)((double)0.529/(double)2.418e-17);

/**
 * the mass conversion from amu to \f$m_e\f$  **/
static const real_t amu_to_m_e = (real_t)((double)1822.83);


typedef struct vec_t {
    real_t* x;
    real_t* y;
    real_t* z;
} vec_t;

typedef struct grid_t {
    vec_t r_box;
    int* neighbor_list;
    int* n_neighbors;
    int* n_atoms;
    real_t* bounds;
    int n_n_size;
    int n_nl_size;
    int n_r_size;

	// for each box list all neighbor atoms
	int *n_num_neigh;
	int *n_neigh_atoms;
	int *n_neigh_boxes;

	// list all atoms and their boxes
	int *n_list_atoms;
	int *n_list_boxes;
	int *itself_start_idx;
} grid_t;

typedef struct eam_pot_t {
    real_t* rho;
    real_t* phi;
    real_t* F;
    int* n_values;

    real_t rho_x0;
    real_t rho_xn;
    real_t rho_invDx;

    real_t phi_x0;
    real_t phi_xn;
    real_t phi_invDx;

    real_t cutoff;
    int n_p_rho_size;
    int n_p_phi_size;
    int n_p_F_size;
} eam_pot_t;

/**
 * EAM uses interpolation.  We store interpolation tables in a struct
 * potentialarray_t.  The interpolation is supported on the range
 * \f$[x_0,x_n]\f$, has n values, and each interval has the width
 * \f$1/invDx\f$.
 **/
typedef struct potentialarray_t {
  int n;          /**< the number of entries in the array **/
  real_t x0;      /**< the starting ordinate range **/
  real_t xn;      /**< the ending ordinate range **/
  real_t invDx;   /**< the inverse of the interval \f$=n/(x_n-x_0)\f$**/
  real_t *values; /**< the abscissa values **/
} potentialarray_t;


typedef struct eam_ch_t {
    potentialarray_t rho;
    potentialarray_t phi;
    potentialarray_t drho;
    potentialarray_t dphi;
} eam_ch_t;

typedef struct lj_pot_t {
    real_t cutoff;
    real_t sigma;
    real_t epsilon;
} lj_pot_t;

typedef struct sim_t {
    // grid array values
    vec_t r;
    vec_t p;
    vec_t f;
    real_t* e;
    real_t* m;
    real_t* fi;
    real_t* rho;
    // grid data 
    grid_t grid;
    // real scalars
    real_t dt;
    real_t rmass;
    real_t cfac;
    real_t energy;
    // integer values
    int array_size;
    int n_cells;
    int nx, ny, nz;
    int total_atoms;
    int max_atoms;		// max number of atoms per box
    // eam flag
    int eam_flag;
    eam_pot_t eam_pot;
    lj_pot_t lj_pot;
    eam_ch_t ch_pot;
} sim_t;

typedef struct config {
  bool eam_flag;
  int x, y, z;
  int iters;
  int steps;
  std::string method;
} config;


