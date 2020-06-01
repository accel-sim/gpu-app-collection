#include "XSbench_header.h"

// Generates randomized energy grid for each nuclide
// Note that this is done as part of initialization (serial), so
// rand() is used.
void generate_grids( NuclideGridPoint ** nuclide_grids,
                     int n_isotopes, int n_gridpoints ) {
	for( int i = 0; i < n_isotopes; i++ )
		for( int j = 0; j < n_gridpoints; j++ )
		{
			nuclide_grids[i][j].energy       =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].total_xs     =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].elastic_xs   =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].absorbtion_xs=((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].fission_xs   =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].nu_fission_xs=((double)rand()/(double)RAND_MAX);
		}
}

// Sorts the nuclide grids by energy (lowest -> highest)
void sort_nuclide_grids( NuclideGridPoint ** nuclide_grids, int n_isotopes,
                         int n_gridpoints )
{
	printf("Sorting Nuclide Energy Grids...\n");
	
	int (*cmp) (const void *, const void *);
	cmp = NGP_compare;
	
	for( int i = 0; i < n_isotopes; i++ )
		qsort( nuclide_grids[i], n_gridpoints, sizeof(NuclideGridPoint),
		       cmp );
}

// Allocates unionized energy grid, and assigns union of energy levels
// from nuclide grids to it.
GridPoint * generate_energy_grid( int n_isotopes, int n_gridpoints,
                                  NuclideGridPoint ** nuclide_grids) {
	printf("Generating Unionized Energy Grid...\n");
	
	int n_unionized_grid_points = n_isotopes*n_gridpoints;
	int (*cmp) (const void *, const void *);
	cmp = NGP_compare;
	
	GridPoint * energy_grid = (GridPoint *)malloc( n_unionized_grid_points
	                                               * sizeof( GridPoint ) );
	
	printf("Copying and Sorting all nuclide grids...\n");
	
	NuclideGridPoint ** n_grid_sorted = gpmatrix( n_isotopes, n_gridpoints );
	
	memcpy( n_grid_sorted[0], nuclide_grids[0], n_isotopes*n_gridpoints*
	                                      sizeof( NuclideGridPoint ) );
	
	qsort( &n_grid_sorted[0][0], n_unionized_grid_points,
	       sizeof(NuclideGridPoint), cmp);
	
	printf("Assigning energies to unionized grid...\n");
	
	for( int i = 0; i < n_unionized_grid_points; i++ )
		energy_grid[i].energy = n_grid_sorted[0][i].energy;
	
	gpmatrix_free(n_grid_sorted);
	
	for( int i = 0; i < n_unionized_grid_points; i++ )
		energy_grid[i].xs_ptrs = (NuclideGridPoint **)
		                         malloc( n_isotopes*sizeof(NuclideGridPoint*));
	
	return energy_grid;
}

// Searches each nuclide grid for the closest energy level and assigns
// pointer from unionized grid to the correct spot in the nuclide grid.
// This process is time consuming, as the number of binary searches
// required is:  binary searches = n_gridpoints * n_isotopes^2
void set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints ){
	
	printf("Assigning pointers to Unionized Energy Grid...\n");
	
	#pragma omp parallel for default(none) \
	shared( energy_grid, nuclide_grids, n_isotopes, n_gridpoints )
	for( int i = 0; i < n_isotopes * n_gridpoints ; i++ )
	{
		double quarry = energy_grid[i].energy;
		if( INFO && omp_get_thread_num() == 0 && i % 200 == 0 )
			printf("\rAligning Unionized Grid...(%.0lf%% complete)",
			       100.0 * (double) i / (n_isotopes*n_gridpoints /
				                         omp_get_num_threads())     );
		for( int j = 0; j < n_isotopes; j++ )
		{
			// j is the nuclide i.d.
			// log n binary search
			energy_grid[i].xs_ptrs[j] = 
				binary_search( nuclide_grids[j], quarry, n_gridpoints);
		}
	}
	printf("\n");

	//test
	/*
	for( int i=0; i < n_isotopes * n_gridpoints; i++ )
		for( int j = 0; j < n_isotopes; j++ )
			printf("E = %.4lf\tNuclide %d->%p->%.4lf\n",
			       energy_grid[i].energy,
                   j,
				   energy_grid[i].xs_ptrs[j],
				   (energy_grid[i].xs_ptrs[j])->energy
				   );
	*/
}
