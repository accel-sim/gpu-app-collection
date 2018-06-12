/**
 * twomm.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 2048
#define NJ 2048
#define NK 2048
#define NL 2048

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


#pragma hmpp <group1> group, target=OpenCL

#pragma hmpp <group1> map, args[loopa::c, loopb::c]

#pragma hmpp <group1> loopa codelet, args[a;b;c].io=inout
void twoMMloopa(DATA_TYPE a[NI][NK], DATA_TYPE b[NK][NJ], DATA_TYPE c[NI][NJ])
{
	int i, j, k;

	#pragma hmppcg grid blocksize 32 X 8
	
	#pragma hmppcg parallel
	for(i = 0; i < NI; i++)
	{
		#pragma hmppcg parallel
		for(j = 0; j < NJ; j++)
		{
			#pragma hmppcg noParallel
			for (k = 0; k < NK; k++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	} 
}

#pragma hmpp <group1> loopb codelet, args[c;d;e].io=inout
void twoMMloopb(DATA_TYPE c[NI][NJ], DATA_TYPE d[NJ][NL], DATA_TYPE e[NI][NL])
{
	int i, j, k;
	
	#pragma hmppcg grid blocksize 32 X 8
	
	#pragma hmppcg parallel
	for(i = 0; i < NI; i++) 
	{
		#pragma hmppcg parallel
		for (j = 0; j < NL; j++)
		{          
			#pragma hmppcg noParallel
			for (k = 0; k < NJ; ++k)
			{
				e[i][j] += c[i][k] * d[k][j]; 
			}
		}
	}
}


void init_array(DATA_TYPE A[NI][NJ], DATA_TYPE B[NI][NJ], DATA_TYPE C[NI][NK], DATA_TYPE C_gpu[NI][NK], DATA_TYPE D[NK][NJ],
		DATA_TYPE E[NJ][NL], DATA_TYPE E_outputFromGpu[NJ][NL])
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j)/NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i][j] = ((DATA_TYPE) i*j + 1)/NJ;
		}
	}
	
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
			C_gpu[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
		}
	}
	
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
		}
	}
	
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			E[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
			E_outputFromGpu[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
		}
	}
}


void compareResults(DATA_TYPE E[NI][NL], DATA_TYPE E_outputFromGpu[NI][NL])
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(E[i][j], E_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char** argv)
{
	double t_start, t_end;
	
	/* Array declaration */
	DATA_TYPE A[NI][NK];
	DATA_TYPE B[NK][NJ];
	DATA_TYPE C[NI][NJ];
	DATA_TYPE C_gpu[NI][NJ];
	DATA_TYPE D[NJ][NL];
	DATA_TYPE E[NI][NL];
	DATA_TYPE E_outputFromGpu[NI][NL];

 	/* Initialize array. */
	init_array(A, B, C, C_gpu, D, E, E_outputFromGpu);
    
	#pragma hmpp <group1> allocate

	#pragma hmpp <group1> loopa advancedload, args[a;b;c]
	#pragma hmpp <group1> loopb advancedload, args[d;e]

	t_start = rtclock();
	#pragma hmpp <group1> loopa callsite, args[a;b;c].advancedload=true, asynchronous
	twoMMloopa(A, B, C_gpu);

	#pragma hmpp <group1> loopa synchronize
	#pragma hmpp <group1> loopb callsite, args[c;d;e].advancedload=true, asynchronous
	twoMMloopb(C_gpu, D, E_outputFromGpu);
	#pragma hmpp <group1> loopb synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp <group1> loopa delegatedstore, args[a;b]
	#pragma hmpp <group1> loopb delegatedstore, args[c;d;e]

	#pragma hmpp <group1> release
    
	t_start = rtclock();
 
	twoMMloopa(A, B, C);
	twoMMloopb(C, D, E);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(E, E_outputFromGpu);

	return 0;
}
