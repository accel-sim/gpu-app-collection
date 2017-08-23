#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<math.h>
#include<omp.h>
#include<unistd.h>
#include<sys/time.h>

// Papi Definition (comment / uncomment to toggle PAPI)
//#define __PAPI

// Papi Header
#ifdef __PAPI
#include "/usr/local/include/papi.h"
#endif

// Variable to add extra flops at each lookup from unionized grid.
//#define ADD_EXTRAS
#define EXTRA_FLOPS 0
#define EXTRA_LOADS 0

// I/O Specifiers
#define INFO 1
#define DEBUG 1
#define SAVE 0
#define PRINT_PAPI_INFO 1

// Structures
typedef struct{
	double energy;
	double total_xs;
	double elastic_xs;
	double absorbtion_xs;
	double fission_xs;
	double nu_fission_xs;
} NuclideGridPoint;

typedef struct{
	double energy;
	int * xs_ptrs;
} GridPoint;


// CHAGED:
#ifdef __cplusplus
  #define RESTRICT __restrict
  #define C_LINKAGE extern "C" 
#else
  #define RESTRICT 
  #define C_LINKAGE 
#endif

#define N_ELEMENTS 5
#define NUM_RESULTS 10000
#define STRIP_RANDOM 1
#define CUDA_PTR_CALCULATE 0

/// End chagned zone



// Function Prototypes


#ifdef __cplusplus
  extern "C" {
#endif

void logo(void);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(int a);

NuclideGridPoint ** gpmatrix(size_t m, size_t n);

void gpmatrix_free( NuclideGridPoint ** M );

int NGP_compare( const void * a, const void * b );

void generate_grids( NuclideGridPoint ** nuclide_grids,
                     int n_isotopes, int n_gridpoints );

void sort_nuclide_grids( NuclideGridPoint ** nuclide_grids, int n_isotopes,
                         int n_gridpoints );

GridPoint * generate_energy_grid( int n_isotopes, int n_gridpoints,
                                  NuclideGridPoint ** nuclide_grids);

void set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints );

int binary_search( NuclideGridPoint * A, double quarry, int n );

void calculate_macro_xs(   double p_energy, int mat, int n_isotopes,
                           int n_gridpoints, int * RESTRICT num_nucs,
                           double ** RESTRICT concs,
						   GridPoint * RESTRICT energy_grid,
                           NuclideGridPoint ** RESTRICT nuclide_grids,
						   int ** RESTRICT mats,
                           double * RESTRICT macro_xs_vector );

void calculate_micro_xs(   double p_energy, int nuc, int n_isotopes,
                           int n_gridpoints,
                           GridPoint * RESTRICT energy_grid,
                           NuclideGridPoint ** RESTRICT nuclide_grids, int idx,
                           double * RESTRICT xs_vector );

int grid_search( int n, double quarry, GridPoint * A);

int * load_num_nucs(int n_isotopes);
int ** load_mats( int * num_nucs, int n_isotopes );
double ** load_concs( int * num_nucs );
int pick_mat(unsigned long * seed);
double rn(unsigned long * seed);
int rn_int(unsigned long * seed);
void counter_stop( int * eventset, int num_papi_events );
void counter_init( int * eventset, int * num_papi_events );
void do_flops(void);
void do_loads( int nuc, NuclideGridPoint ** RESTRICT nuclide_grids,    int n_gridpoints );
	
#ifdef  __cplusplus
  }
#endif


#endif
