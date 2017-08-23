#include "XSbench_header.h"

// Calculates the microscopic cross section for a given nuclide & energy
void calculate_micro_xs(   double p_energy, int nuc, int n_isotopes,
                           int n_gridpoints,
                           GridPoint * restrict energy_grid,
                           NuclideGridPoint ** restrict nuclide_grids,
                           int idx, double * restrict xs_vector ){
	
	// Variables
	double f;
	
	// pull ptr from energy grid
	NuclideGridPoint * low = &nuclide_grids[nuc][energy_grid[idx].xs_ptrs[nuc]];
	NuclideGridPoint * high = low + 1;
	
	// calculate the re-useable interpolation factor
	f = (high->energy - p_energy) / (high->energy - low->energy);

	// Total XS
	xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
	
	#ifdef ADD_EXTRAS
	do_flops();
	do_loads( nuc, nuclide_grids, n_gridpoints );	
	#endif
	
	// Elastic XS
	xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
	
	#ifdef ADD_EXTRAS
		do_flops();
		do_loads( (nuc+1) % n_isotopes, nuclide_grids, n_gridpoints );	
	#endif
	
	// Absorbtion XS
	xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
	
	#ifdef ADD_EXTRAS
		do_flops();
		do_loads( (nuc+2) % n_isotopes, nuclide_grids, n_gridpoints );	
	#endif
	
	// Fission XS
	xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
	
	#ifdef ADD_EXTRAS
		do_flops();
		do_loads( (nuc+3) % n_isotopes, nuclide_grids, n_gridpoints );	
	#endif
	
	// Nu Fission XS
	xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
	
	#ifdef ADD_EXTRAS
		do_flops();
		do_loads( (nuc+4) % n_isotopes, nuclide_grids, n_gridpoints );	
	#endif
	
	//test
	/*	
	if( omp_get_thread_num() == 0 )
	{
		printf("Lookup: Energy = %lf, nuc = %d\n", p_energy, nuc);
		printf("e_h = %lf e_l = %lf\n", high->energy , low->energy);
		printf("xs_h = %lf xs_l = %lf\n", high->elastic_xs, low->elastic_xs);
		printf("total_xs = %lf\n\n", xs_vector[1]);
	}
	*/
	
}

// Calculates macroscopic cross section based on a given material & energy 
void calculate_macro_xs( double p_energy, int mat, int n_isotopes,
                         int n_gridpoints, int * restrict num_nucs,
                         double ** restrict concs,
                         GridPoint * restrict energy_grid,
                         NuclideGridPoint ** restrict nuclide_grids,
                         int ** restrict mats,
                         double * restrict macro_xs_vector ){
	double xs_vector[5];
	int p_nuc; // the nuclide we are looking up
	int idx = 0;	
	double conc; // the concentration of the nuclide in the material

	// cleans out macro_xs_vector
	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	// binary search for energy on unionized energy grid (UEG)
	idx = grid_search( n_isotopes * n_gridpoints, p_energy,
	                   energy_grid);	
	
	// Once we find the pointer array on the UEG, we can pull the data
	// from the respective nuclide grids, as well as the nuclide
	// concentration data for the material
	// Each nuclide from the material needs to have its micro-XS array
	// looked up & interpolatied (via calculate_micro_xs). Then, the
	// micro XS is multiplied by the concentration of that nuclide
	// in the material, and added to the total macro XS array.
	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		p_nuc = mats[mat][j];
		conc = concs[mat][j];
		calculate_micro_xs( p_energy, p_nuc, n_isotopes,
		                    n_gridpoints, energy_grid,
		                    nuclide_grids, idx, xs_vector );
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
	
	//test
	/*
	for( int k = 0; k < 5; k++ )
		printf("Energy: %lf, Material: %d, XSVector[%d]: %lf\n",
		       p_energy, mat, k, macro_xs_vector[k]);
	*/
}

#if 0
// binary search for energy on unionized energy grid
// returns lower index
int grid_search( int n, double quarry, GridPoint * A)
{
	int min = 0;
	int max = n-1;
	int mid = 0;

	// checks to make sure we don't read off the end of the grid	
	if( quarry < A[0].energy )
		return 0;
	else if( quarry > A[n-1].energy )
		return n-2;
	
	//printf("\nQuarry: %lf\n", quarry);
		
	while( max > min + 1 )
	{
		mid = min + floor( (max-min) / 2.0);
		if( A[mid].energy < quarry )
			min = mid;
		else if( A[mid].energy > quarry )
			max = mid;
		else
			return mid;
	}
	
	//printf("Returning mid: %lf (index %d of %d)\n", A[mid].energy, mid, n);
	return mid;
}
#else

int grid_search( int n, double q, GridPoint * A)
{

  int lowerLimit = 0;
  int upperLimit = n-1;
  int examinationPoint;
  int length = upperLimit - lowerLimit;

  while(length > 1)
  {
   //  (length > 1)  ==> ( length/2 >= 1 )  ==> (examinationPoint != lowerLimit )
    examinationPoint = lowerLimit + (length/2); 
    if(A[examinationPoint].energy > q)
    {
      upperLimit = examinationPoint;
    }
    else{
      lowerLimit = examinationPoint;
    }
    length = upperLimit - lowerLimit;
  }
  return lowerLimit;

}
#endif
