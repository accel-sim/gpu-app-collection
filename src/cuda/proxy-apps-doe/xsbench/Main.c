#include "XSbench_header.h"
#include "cudaHeader.h"
#include <limits.h>

#define ISOTOPES 130
#define ISOTOPES 355

int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	
	unsigned long seed;
	size_t memtotal;
	int n_isotopes; // H-M Large is 355, H-M Small is 68
	int n_gridpoints = 11303;
//	int lookups = 10000;
	int lookups = 50000;
//	int lookups = 100000;
	int i, thread, nthreads, mat;
	double omp_start, omp_end, p_energy;
	int max_procs = omp_get_num_procs();
	char * HM;
	int bgq_mode = 0;
	int kernelId =0;
	// rand() is only used in the serial initialization stages.
	// A custom RNG is used in parallel portions.
//	srand(time(NULL)); //commented this so that same addresses are generated
//	across the runs
	srand(INT_MAX);

	// Process CLI Fields
	// Usage:   ./XSBench <# threads> <H-M Size ("Small or "Large")> <BGQ mode>
	// # threads - The number of threads you wish to run
	// H-M Size -  The problem size (small = 68 nuclides, large = 355 nuclides)
	// BGQ Mode -  Number of ranks - no real effect, save for stamping the
	//             results.txt printout
	// Note - No arguments are required - default parameters will be used if
	//        no arguments are given.

	if( argc == 2 )
	{
		nthreads = atoi(argv[1]);	// first arg sets # of threads
		n_isotopes = ISOTOPES;			// defaults to H-M Large
	}
	else if( argc == 3 )
	{
		nthreads = atoi(argv[1]);	// first arg sets # of threads
		// second arg species small or large H-M benchmark
		if( strcmp( argv[2], "small") == 0 || strcmp( argv[2], "Small" ) == 0)
			n_isotopes = 68;
		else
			n_isotopes = ISOTOPES;
	}
	else if( argc == 4 )
	{
		kernelId = atoi(argv[3]); 
		nthreads = atoi(argv[1]);	// first arg sets # of threads
		// second arg species small or large H-M benchmark
		if( strcmp( argv[2], "small") == 0 || strcmp( argv[2], "Small" ) == 0)
			n_isotopes = 68;
		else
			n_isotopes = ISOTOPES;
	}
  else if(argc == 5)
  {
    n_gridpoints = atoi(argv[4]);
		kernelId = atoi(argv[3]); 
		nthreads = atoi(argv[1]);	// first arg sets # of threads
		// second arg species small or large H-M benchmark
		if( strcmp( argv[2], "small") == 0 || strcmp( argv[2], "Small" ) == 0)
			n_isotopes = 68;
		else
			n_isotopes = ISOTOPES;
  }
	else
	{
		nthreads = max_procs;		// defaults to full CPU usage
		n_isotopes = ISOTOPES;			// defaults to H-M Large
	}

	// Sets H-M size name
	if( n_isotopes == 68 )
		HM = "Small";
	else
		HM = "Large";

	// Set number of OpenMP Threads
	omp_set_num_threads(nthreads); 
		
	// =====================================================================
	// Calculate Estimate of Memory Usage
	// =====================================================================

	size_t single_nuclide_grid = n_gridpoints * sizeof( NuclideGridPoint );
	size_t all_nuclide_grids = n_isotopes * single_nuclide_grid;
	size_t size_GridPoint =sizeof(GridPoint)+n_isotopes*sizeof(int);
	size_t size_UEG = n_isotopes*n_gridpoints * size_GridPoint;
	int mem_tot;
	memtotal = all_nuclide_grids + size_UEG;
	all_nuclide_grids = all_nuclide_grids  / 1000000;//48576;
	size_UEG = size_UEG / 1000000;//48576;
	memtotal = memtotal / 1000000;//48576;
	mem_tot = memtotal;

	// =====================================================================
	// Print-out of Input Summary
	// =====================================================================
	
	logo();
	center_print("INPUT SUMMARY", 79);
	border_print();
	printf("Materials:                    %d\n", 12);
	printf("H-M Benchmark Size:           %s\n", HM);
	printf("Total Isotopes:               %d\n", n_isotopes);
	printf("Gridpoints (per Nuclide):     "); 
  fancy_int(n_gridpoints);
	printf("\nUnionized Energy Gridpoints:  ");
  fancy_int(n_isotopes*n_gridpoints);
	printf("\nXS Lookups:                   "); 
  fancy_int(lookups);
	printf("\nThreads:                      %d\n", nthreads);
	printf("Est. Memory Usage (MB):       "); 
  fancy_int(mem_tot);
  printf("\n");
	if( EXTRA_FLOPS > 0 )
		printf("Extra Flops:                  %d\n", EXTRA_FLOPS);
	if( EXTRA_LOADS > 0 )
		printf("Extra Loads:                  %d\n", EXTRA_LOADS);
	border_print();
	center_print("\nINITIALIZATION", 79);
	border_print();
	
	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// =====================================================================

	// Allocate & fill energy grids
	printf("Generating Nuclide Energy Grids...\n");
	
	NuclideGridPoint ** nuclide_grids = gpmatrix( n_isotopes, n_gridpoints );
	
	generate_grids( nuclide_grids, n_isotopes, n_gridpoints );	
	
	// Sort grids by energy
	sort_nuclide_grids( nuclide_grids, n_isotopes, n_gridpoints );

	// Prepare Unionized Energy Grid Framework
	GridPoint * energy_grid = generate_energy_grid( n_isotopes, n_gridpoints,
	                                                nuclide_grids ); 	

	// Double Indexing. Filling in energy_grid with pointers to the
	// nuclide_energy_grids.
  omp_start = omp_get_wtime();
	set_grid_ptrs( energy_grid, nuclide_grids, n_isotopes, n_gridpoints );
	omp_end = omp_get_wtime();
  printf("Pointer calculation took %f seconds\n", omp_end-omp_start);
  
	// Get material data
	printf("Loading Mats...\n");
	int *num_nucs = load_num_nucs(n_isotopes);
	int **mats = load_mats(num_nucs, n_isotopes);
	double **concs = load_concs(num_nucs);
  double * results = (double*) malloc(N_ELEMENTS*NUM_RESULTS * sizeof(double));

	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================
	
	border_print();
	center_print("SIMULATION", 79);
	border_print();

	omp_start = omp_get_wtime();
	
	#ifdef __PAPI
	int eventset = PAPI_NULL; 
	int num_papi_events;
	counter_init(&eventset, &num_papi_events);
	#endif
	
	// OpenMP compiler directives - declaring variables as shared or private
	#pragma omp parallel default(none) \
	private(i, thread, p_energy, mat, seed) \
	shared( max_procs, n_isotopes, n_gridpoints, \
	energy_grid, nuclide_grids, lookups, nthreads, \
          mats, concs, num_nucs, results)
	{	
		double macro_xs_vector[5];
		thread = omp_get_thread_num();
		seed = (thread+1)*19+17;
		#pragma omp for
		for( i = 0; i < lookups; i++ )
		{
			// Status text
			if( INFO && thread == 0 && i % 1000 == 0 )
				printf("\rCalculating XS's... (%.0lf%% completed)",
						i / ( lookups / (double) nthreads ) * 100.0);
			

#if(STRIP_RANDOM==1)
      p_energy = 0.01 + (((double)(i%10))/10.0) + (((double)(i%1000))/1000.0);
      p_energy -= ((int)(p_energy));
      //p_energy = i/(float)lookups;
      mat =  i %12;
#else
			// Randomly pick an energy and material for the particle
			p_energy = rn(&seed);
      mat = pick_mat(&seed); 		
#endif

		
			// This returns the macro_xs_vector, but we're not going
			// to do anything with it in this program, so return value
			// is written over.
			calculate_macro_xs( p_energy, mat, n_isotopes,
			                    n_gridpoints, num_nucs, concs,
			                    energy_grid, nuclide_grids, mats,
                                macro_xs_vector );
#if(STRIP_RANDOM ==1)
      if( i < NUM_RESULTS)
      {
        memcpy(&results[5*i], &macro_xs_vector[0], 5*sizeof(double));        
      }
#endif
		}
	}
	omp_end = omp_get_wtime();
	
	printf("\n" );
	printf("Simulation complete.\n" );

  double cudaLookupRate = cudaDriver(lookups, n_isotopes, n_gridpoints, 12, num_nucs, energy_grid, concs, nuclide_grids, mats, results, kernelId);


	
	// =====================================================================
	// Print / Save Results and Exit
	// =====================================================================
	
	border_print();
	center_print("RESULTS", 79);
	border_print();

	// Print the results
	printf("Threads:     %d\n", nthreads);
	if( EXTRA_FLOPS > 0 )
	printf("Extra Flops: %d\n", EXTRA_FLOPS);
	if( EXTRA_LOADS > 0 )
	printf("Extra Loads: %d\n", EXTRA_LOADS);
	printf("Runtime:     %.3lf seconds\n", omp_end-omp_start);
	printf("Lookups:     "); fancy_int(lookups);
  double cpuLookupRate =  ((double) lookups / (omp_end-omp_start));
  printf("\n");
	printf("CPU %d threads \tLookups/s:   ", nthreads);	
  fancy_int((int) cpuLookupRate);
  printf("\n");
  printf("CUDA Port      \tLookups/s:   ");
  fancy_int((int)cudaLookupRate);
  printf("\t %.2fX CPU\n", (cudaLookupRate/cpuLookupRate));


	border_print();

	// For bechmarking, output lookup/s data to file
	if( SAVE )
	{
		FILE * out = fopen( "results.txt", "a" );
		fprintf(out, "c%d\t%d\t%.0lf\n", bgq_mode, nthreads,
		       (double) lookups / (omp_end-omp_start));
		fclose(out);
	}
	
	#ifdef __PAPI
	counter_stop(&eventset, num_papi_events);
	#endif

  free(results);
	return 0;
}
