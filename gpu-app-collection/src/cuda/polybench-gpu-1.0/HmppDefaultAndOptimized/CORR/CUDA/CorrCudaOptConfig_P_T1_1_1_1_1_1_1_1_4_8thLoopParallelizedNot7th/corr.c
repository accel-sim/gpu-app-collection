/**
 * corr.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define M 2048
#define N 2048

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp corr codelet, target=CUDA, args[pdata;psymmat;pstddev;pmean].io=inout
void runCorr(DATA_TYPE pdata[M+1][N+1], DATA_TYPE psymmat[M+1][M+1], DATA_TYPE pstddev[M+1], DATA_TYPE pmean[M+1], DATA_TYPE pfloat_n, DATA_TYPE peps)
{
	int i, j, j1, j2;

	#define sqrt_of_array_cell(x,j) sqrt(x[j])

	/* Determine mean of column vectors of input data matrix */
	#pragma hmppcg grid blocksize 32 X 8

	for (j = 1; j <= M; j++)
	{
		pmean[j] = 0.0;

		for (i = 1; i <= N; i++)
		{
			pmean[j] += pdata[i][j];
		}
		pmean[j] /= pfloat_n;
	}

	#pragma hmppcg grid blocksize 32 X 8

	/* Determine standard deviations of column vectors of data matrix. */
	for (j = 1; j <= M; j++)
	{
		pstddev[j] = 0.0;

		for (i = 1; i <= N; i++)
		{
			pstddev[j] += (pdata[i][j] - pmean[j]) * (pdata[i][j] - pmean[j]);
		}

		pstddev[j] /= pfloat_n;
		pstddev[j] = sqrt_of_array_cell(pstddev, j);

		/* The following in an inelegant but usual way to handle
		   near-zero std. dev. values, which below would cause a zero-
		   divide. */

		pstddev[j] = pstddev[j] <= peps ? 1.0 : pstddev[j];
	}

	#pragma hmppcg grid blocksize 32 X 8

	/* Center and reduce the column vectors. */
	for (i = 1; i <= N; i++)
	{
		for (j = 1; j <= M; j++)
		{
			pdata[i][j] -= pmean[j];
			pdata[i][j] /= sqrt(pfloat_n) * pstddev[j];
		}
	}

	#pragma hmppcg grid blocksize 32 X 8

	/* Calculate the m * m correlation matrix. */
	#pragma hmppcg noParallel
	for (j1 = 1; j1 <= M-1; j1++)
	{
		psymmat[j1][j1] = 1.0;
        
		for (j2 = j1+1; j2 <= M; j2++)
		{
			psymmat[j1][j2] = 0.0;
            
			#pragma hmppcg tile i:4
			for (i = 1; i <= N; i++)
			{
				psymmat[j1][j2] += (pdata[i][j1] * pdata[i][j2]);
			}
			psymmat[j2][j1] = psymmat[j1][j2];
		}
	}
	psymmat[M][M] = 1.0;
}


void init_arrays(DATA_TYPE data[M][N], DATA_TYPE data_Gpu[M][N])
{
	int i, j;
	
	for (i=0; i < (M+1); i++) 
	{
    		for (j=0; j< (N+1); j++) 
		{
       			data[i][j] = ((DATA_TYPE) i*j)/ (M+1);	
       			data_Gpu[i][j] = ((DATA_TYPE) i*j)/ (M+1);	
       		}
    	}
}

void compareResults(DATA_TYPE symmat[M][N], DATA_TYPE symmat_outputFromGpu[M][N])
{
	int i,j,fail;
	fail = 0;

	for (i=1; i < (M+1); i++)
	{
		for (j=1; j < (N+1); j++)
		{
			if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
				printf("i: %d j: %d\n1: %f 2: %f\n", i, j, symmat[i][j], symmat_outputFromGpu[i][j]);
		
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;
	double t_start, t_end;

	/* Array declaration */
	DATA_TYPE float_n = 321414134.01;
	DATA_TYPE eps = 0.005;
	DATA_TYPE data[M + 1][N + 1];
	DATA_TYPE data_Gpu[M + 1][N + 1];
	DATA_TYPE mean[M + 1];
	DATA_TYPE mean_Gpu[M + 1];
	DATA_TYPE stddev[M + 1];
	DATA_TYPE stddev_Gpu[M + 1];
	DATA_TYPE symmat[M + 1][M + 1];
	DATA_TYPE symmat_outputFromGpu[M + 1][M + 1];

	/* Initialize array. */
	init_arrays(data, data_Gpu);
	
	#pragma hmpp corr allocate
    
	#pragma hmpp corr advancedload, args[pdata;psymmat;pstddev;pmean;pfloat_n;peps]

	t_start = rtclock();
	
	#pragma hmpp corr callsite, args[pdata;psymmat;pstddev;pmean;pfloat_n;peps].advancedload=true, asynchronous
	runCorr(data_Gpu, symmat_outputFromGpu, stddev_Gpu, mean_Gpu, float_n, eps);
    
	#pragma hmpp corr synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp corr delegatedstore, args[pdata;psymmat;pstddev;pmean]
	#pragma hmpp corr release
	
	t_start = rtclock();
	
	runCorr(data, symmat, stddev, mean, float_n, eps);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(symmat, symmat_outputFromGpu);

	return 0;
}
