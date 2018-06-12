/**
 * gemm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
#define NI 512
#define NJ 512
#define NK 512

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp gemm codelet, target=OpenCL, args[c].io=inout
void runGemm(DATA_TYPE a[NI][NJ], DATA_TYPE b[NI][NJ], DATA_TYPE c[NI][NJ])
{
	int i, j, k;
	DATA_TYPE p_alpha = 32412;
	DATA_TYPE p_beta = 2123;

	/* C := alpha*A*B + beta*C */
	#pragma hmppcg grid blocksize 32 X 8
	#pragma hmppcg permute i, j, k    
	for (i = 0; i < NI; i++)
	{    
		for (j = 0; j < NJ; j++)
		{
			c[i][j] *= p_beta;

			#pragma hmppcg unroll 8, guarded
			for (k = 0; k < NK; ++k)
			{
				c[i][j] += p_alpha * a[i][k] * b[k][j];
			}
		}
	}
}


void init(DATA_TYPE A[NI][NK], DATA_TYPE B[NK][NJ], DATA_TYPE C[NI][NJ], DATA_TYPE C_outputFromGpu[NI][NJ])
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
			B[i][j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j + 2) / NJ;
			C_outputFromGpu[i][j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void compareResults(DATA_TYPE C[NI][NJ], DATA_TYPE C_outputFromGpu[NI][NJ])
{
	int i, j, fail;
	fail = 0;
	
	// Compare output from CPU and GPU
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}



int main(int argc, char** argv)
{
	double t_start, t_end;
  
	/* Array declaration */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	DATA_TYPE C[NI][NJ];
	DATA_TYPE C_outputFromGpu[NI][NJ];
	DATA_TYPE A[NI][NK];
	DATA_TYPE B[NK][NJ];

	/* Initialize array. */
	init(A, B, C, C_outputFromGpu);
    
	#pragma hmpp gemm allocate
	#pragma hmpp gemm advancedload, args[a;b;c]

	t_start = rtclock();

	#pragma hmpp gemm callsite, args[a;b;c].advancedload=true, asynchronous
	runGemm(A, B, C_outputFromGpu);
    
	#pragma hmpp gemm synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp gemm delegatedstore, args[c]
	#pragma hmpp gemm release
	
	t_start = rtclock();

	runGemm(A, B, C);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(C, C_outputFromGpu);

	return 0;
}

