/**
 * syrk.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N 1024
#define M 1024

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp syrk codelet, target=OpenCL, args[c].io=inout
void runSyrk(DATA_TYPE a[N][M], DATA_TYPE c[N][N])
{
	int i, j, k;
	DATA_TYPE temp;
	DATA_TYPE alpha, beta;
	alpha = 12435;
	beta = 4546;

     
	#pragma hmppcg grid blocksize 32 X 8
  
	#pragma hmppcg parallel
	for (i = 0; i < N; i++)
	{
		#pragma hmppcg parallel
		for (j = 0; j < N; j++)
		{
			c[i][j] *= beta;
		}
	}

	#pragma hmppcg grid blocksize 32 X 8
  
	#pragma hmppcg parallel
	for (i = 0; i < N; i++)
	{
		#pragma hmppcg parallel
		for (j = 0; j < N; j++)
		{      
			#pragma hmppcg noParallel
			for (k = 0; k < M; k++)
			{
				c[i][j] += alpha * a[i][k] * a[j][k];
			}
		}
	}
}


void init_arrays(DATA_TYPE A[N][M], DATA_TYPE C[N][N], DATA_TYPE C_outputFromGpu[N][N])
{
	int i, j;
	
	for (i = 0; i < N; i++)
   	{
		for (j = 0; j < M; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
		
		for (j = 0; j < N; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j + 2) / N;
			C_outputFromGpu[i][j] = ((DATA_TYPE) i*j + 2) / N;
		}
	}
}


void compareResults(DATA_TYPE C[N][N], DATA_TYPE C_outputFromGpu[N][N])
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<N; i++)
	{
		for (j=0; j<M; j++)
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
	DATA_TYPE A[N][M];
	DATA_TYPE C[N][N];
	DATA_TYPE C_outputFromGpu[N][N];

	/* Initialize array. */
	init_arrays(A, C, C_outputFromGpu);

	#pragma hmpp syrk allocate
	#pragma hmpp syrk advancedload, args[a,c]

	t_start = rtclock();
	#pragma hmpp syrk callsite, args[a,c].advancedload=true, asynchronous
	runSyrk(A, C_outputFromGpu);
	#pragma hmpp syrk synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	#pragma hmpp syrk delegatedstore, args[c]
	#pragma hmpp syrk release

	t_start = rtclock();
	runSyrk(A, C);
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(C, C_outputFromGpu);

	return 0;
}

