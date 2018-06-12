/**
 * threedconv.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem dimensions */
#define NI 256
#define NJ 256
#define NK 256

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp conv codelet, target=CUDA, args[A].io=in, args[B].io=inout
void conv3D(DATA_TYPE A[NI][NJ][NK], DATA_TYPE B[NI][NJ][NK])
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	#pragma hmppcg grid blocksize 32 X 8
    #pragma hmppcg permute i, k, j
    
	for (i = 1; i < NI - 1; ++i) // 0
	{
		#pragma hmppcg unroll 4, guarded
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				B[i][j][k] = 0
					+   c11 * A[i - 1][j - 1][k - 1]  +  c13 * A[i + 1][j - 1][k - 1]
					+   c21 * A[i - 1][j - 1][k - 1]  +  c23 * A[i + 1][j - 1][k - 1]
					+   c31 * A[i - 1][j - 1][k - 1]  +  c33 * A[i + 1][j - 1][k - 1]
					+   c12 * A[i + 0][j - 1][k + 0]
					+   c22 * A[i + 0][j + 0][k + 0]
					+   c32 * A[i + 0][j + 1][k + 0]
					+   c11 * A[i - 1][j - 1][k + 1]  +  c13 * A[i + 1][j - 1][k + 1]
					+   c21 * A[i - 1][j + 0][k + 1]  +  c23 * A[i + 1][j + 0][k + 1]
					+   c31 * A[i - 1][j + 1][k + 1]  +  c33 * A[i + 1][j + 1][k + 1];
			}
		}
	}
}

void init(DATA_TYPE A[NI][NJ][NK])
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < 256; ++k)
			{
				A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


void compareResults(DATA_TYPE B[NI][NJ][NK], DATA_TYPE B_outputFromGpu[NI][NJ][NK])
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char *argv[])
{
	double t_start, t_end;
	
	DATA_TYPE A[NI][NJ][NK];
	DATA_TYPE B[NI][NJ][NK];  // CPU target results
	DATA_TYPE B_outputFromGpu[NI][NJ][NK];  // GPU exec results

	//initialize the arrays
	init(A);

	#pragma hmpp conv allocate

	#pragma hmpp conv advancedload, args[A;B]
	
	// Run GPU code
	
	t_start = rtclock();
	
	#pragma hmpp conv callsite, args[A;B].advancedload=true, asynchronous
	conv3D(A, B_outputFromGpu);

	#pragma hmpp conv synchronize

    t_end = rtclock();
    fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	#pragma hmpp conv delegatedstore, args[B]

	#pragma hmpp conv release

	t_start = rtclock();
	
	conv3D(A, B);
	
    t_end = rtclock();
    fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(B, B_outputFromGpu);

	return 0;
}
