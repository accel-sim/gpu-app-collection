/**
 * gesummv.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp gesummv codelet, target=CUDA, args[a, b, x1, y1, tmp1].io=inout
void runGesummv(DATA_TYPE a[N][N], DATA_TYPE b[N][N], DATA_TYPE x1[N], DATA_TYPE y1[N], DATA_TYPE tmp1[N])
{
	int i, j;
	
	DATA_TYPE alpha = 43532;
	DATA_TYPE beta = 12313;

	#pragma hmppcg grid blocksize 32 X 8
	#pragma hmppcg permute i, j
    
	#pragma hmppcg parallel
	for (i = 0; i < N; i++)
	{
		tmp1[i] = 0;
		y1[i] = 0;

		#pragma hmppcg noParallel
		for (j = 0; j < N; j++)
		{
			tmp1[i] = a[i][j] * x1[j] + tmp1[i];
			y1[i] = b[i][j] * x1[j] + y1[i];
		}
		y1[i] = alpha * tmp1[i] + beta * y1[i];
	}
}


void init(DATA_TYPE A[N][N], DATA_TYPE x[N])
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		x[i] = ((DATA_TYPE) i) / N;
      	
		for (j = 0; j < N; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
	}
}


void compareResults(DATA_TYPE y[N], DATA_TYPE y_outputFromGpu[N])
{
	int i, fail;
	fail = 0;
	
	for (i=0; i< N; i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char** argv)
{
	double t_start, t_end;
  
	/* Array declaration */
	DATA_TYPE A[N][N];
	DATA_TYPE B[N][N];
	DATA_TYPE x[N];
	DATA_TYPE y[N];
	DATA_TYPE y_outputFromGpu[N];
	DATA_TYPE tmp[N];

	/* Initialize array. */
	init(A, x);
    
	#pragma hmpp gesummv allocate
	#pragma hmpp gesummv advancedload, args[a, b, x1, y1, tmp1]
	t_start = rtclock();
	
	#pragma hmpp gesummv callsite, args[a, b, x1, y1, tmp1].advancedload=true, asynchronous
	runGesummv(A, B, x, y_outputFromGpu, tmp);
    
	#pragma hmpp gesummv synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp gesummv delegatedstore, args[a, b, x1, y1, tmp1]
    
	#pragma hmpp gesummv release
	
	t_start = rtclock();
	
	runGesummv(A, B, x, y, tmp);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(y, y_outputFromGpu);

	return 0;
}
