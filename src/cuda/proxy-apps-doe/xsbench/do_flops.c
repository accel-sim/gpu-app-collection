#include "XSbench_header.h"

// Note that this file needs to be compiled separately with the -O0 flag
// to prevent the compiler from optimizing away the flops. The included
// makefile takes this into account.

// Does a extra, arbitrary flops.
void do_flops( void )
{
	double a = 1.33;
	double b = 2.34;
	int i;
	
	for( i = 0; i < EXTRA_FLOPS; i++ )
	{
		a = a * b;
	}
}

// Does extra, random memory loads
void do_loads( int nuc,
               NuclideGridPoint ** restrict nuclide_grids,
		       int n_gridpoints )
{
	int i, idx;
	unsigned long tmp = nuc;
	double load;
	for( i = 0; i < EXTRA_LOADS; i++ )
	{
		idx = rn_int( &tmp ) % n_gridpoints;
		load = nuclide_grids[nuc][idx].total_xs;
	}
}

// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
// This variant is used to find integers, rather than floats.
int rn_int(unsigned long * seed)
{
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (*seed) ) % m;
	*seed = n1;
	return n1;
}
