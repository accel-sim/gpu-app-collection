/**
 * threemm.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define NI 512
#define NJ 512
#define NK 512
#define NL 512
#define NM 512

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


#pragma hmpp <group1> group, target=OpenCL

#pragma hmpp <group1> map, args[loopa::e, loopc::e]
#pragma hmpp <group1> map, args[loopb::f, loopc::f]

#pragma hmpp <group1> loopa codelet, args[a;b;e].io=inout
void threeMMloopa(DATA_TYPE a[NI][NK], DATA_TYPE b[NK][NJ], DATA_TYPE e[NI][NJ])
{
	int i, j, k;

	/* E := A*B */
	#pragma hmppcg grid blocksize 32 X 8

	#pragma hmppcg parallel
	for (i = 0; i < NI; i++)
	{     

		#pragma hmppcg parallel
		for (j = 0; j < NJ; j++)
		{
			e[i][j] = 0;
         
			#pragma hmppcg noParallel
			for (k = 0; k < NK; ++k)
			{
				e[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

#pragma hmpp <group1> loopb codelet, args[f;c;d].io=inout
void threeMMloopb(DATA_TYPE c[NJ][NM], DATA_TYPE d[NM][NL], DATA_TYPE f[NJ][NL])
{
	int i, j, k;
 
	/* F := C*D */
	#pragma hmppcg grid blocksize 32 X 8
  
	#pragma hmppcg parallel
	for (i = 0; i < NJ; i++)
	{
		#pragma hmppcg parallel
		for (j = 0; j < NL; j++)
		{
			f[i][j] = 0;
          
			#pragma hmppcg noParallel
			for (k = 0; k < NM; ++k)
			{
				f[i][j] += c[i][k] * d[k][j];
			}
		}
	}
}

#pragma hmpp <group1> loopc codelet, args[g;e;f].io=inout
void threeMMloopc(DATA_TYPE e[NI][NJ], DATA_TYPE f[NJ][NL], DATA_TYPE g[NI][NL])
{
	int i, j, k;

	/* G := E*F */
	#pragma hmppcg grid blocksize 32 X 8


	#pragma hmppcg parallel
	for (i = 0; i < NI; i++)
	{

		#pragma hmppcg parallel
		for (j = 0; j < NL; j++)
		{
			g[i][j] = 0;
          
			#pragma hmppcg noParallel
			for (k = 0; k < NJ; ++k)
			{
				g[i][j] += e[i][k] * f[k][j];
			}
		}
	}
}


void compareResults(DATA_TYPE G[NI][NL], DATA_TYPE G_outputFromGpu[NI][NL])
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void iNIt_array(DATA_TYPE A[NI][NK], DATA_TYPE B[NK][NJ], DATA_TYPE C[NJ][NM], DATA_TYPE D[NM][NL])
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	/* Array declaration */
	DATA_TYPE A[NI][NK];
	DATA_TYPE B[NK][NJ];
	DATA_TYPE C[NJ][NM];
	DATA_TYPE D[NM][NL];
	DATA_TYPE E[NI][NJ];
	DATA_TYPE E_gpu[NI][NJ];	
	DATA_TYPE F[NJ][NL];
	DATA_TYPE F_gpu[NJ][NL];
	DATA_TYPE G[NI][NL];
	DATA_TYPE G_outputFromGpu[NI][NL];

	/* INItialize array. */
	iNIt_array(A, B, C, D);
    
	#pragma hmpp <group1> allocate

	#pragma hmpp <group1> loopa advancedload, args[a;b;e]
	#pragma hmpp <group1> loopb advancedload, args[f;c;d]
	#pragma hmpp <group1> loopc advancedload, args[g]

	t_start = rtclock();
	#pragma hmpp <group1> loopa callsite, args[a;b;e].advancedload=true, asynchronous
	threeMMloopa(A, B, E_gpu);
	#pragma hmpp <group1> loopa synchronize
	#pragma hmpp <group1> loopb callsite, args[f;c;d].advancedload=true, asynchronous
	threeMMloopb(C, D, F_gpu);
	#pragma hmpp <group1> loopb synchronize
	#pragma hmpp <group1> loopc callsite, args[g;e;f].advancedload=true, asynchronous
	threeMMloopc(E_gpu, F_gpu, G_outputFromGpu);
	#pragma hmpp <group1> loopc synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp <group1> loopa delegatedstore, args[a;b]
	#pragma hmpp <group1> loopb delegatedstore, args[c;d]
	#pragma hmpp <group1> loopc delegatedstore, args[g;e;f]

	#pragma hmpp <group1> release
	
	t_start = rtclock();

	threeMMloopa(A, B, E);
	threeMMloopb(C, D, F);
	threeMMloopc(E, F, G);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(G, G_outputFromGpu);

	return 0;
}

