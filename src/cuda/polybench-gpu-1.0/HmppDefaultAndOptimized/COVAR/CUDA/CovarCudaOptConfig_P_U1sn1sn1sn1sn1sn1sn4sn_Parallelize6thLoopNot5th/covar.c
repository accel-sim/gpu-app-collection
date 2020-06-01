/**
 * covar.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define M 2048
#define N 2048

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp <group1> group, target=CUDA

#pragma hmpp <group1> map, args[loopa::pmean, loopb::pmean]
#pragma hmpp <group1> map, args[loopa::pdata, loopb::pdata, loopc::pdata]

#pragma hmpp <group1> loopa codelet, args[pmean;pdata].io=inout
void covarLoopa(DATA_TYPE pmean[M+1], DATA_TYPE pdata[M+1][N+1], DATA_TYPE pfloat_n)
{
	int i, j, j1, j2;

	/* Determine mean of column vectors of input data matrix */
	#pragma hmppcg grid blocksize 32 X 8
        
	#pragma hmppcg parallel
	for (j = 1; j <= M; j++)
	{
		pmean[j] = 0.0;

		#pragma hmppcg noParallel
		for (i = 1; i <= N; i++)
		{
			pmean[j] += pdata[i][j];
		}
		pmean[j] /= pfloat_n;
	}
}


#pragma hmpp <group1> loopb codelet, args[pdata;pmean].io=inout
void covarLoopb(DATA_TYPE pdata[M+1][N+1], DATA_TYPE pmean[M+1])
{
	int i, j;

	/* Center the column vectors. */
	#pragma hmppcg grid blocksize 32 X 8

	#pragma hmppcg parallel
	for (i = 1; i <= N; i++)
	{
		#pragma hmppcg parallel
		for (j = 1; j <= M; j++)
		{
			pdata[i][j] -= pmean[j];
		}
	}
}

#pragma hmpp <group1> loopc codelet, args[psymmat;pdata].io=inout
void covarLoopc(DATA_TYPE psymmat[M+1][N+1], DATA_TYPE pdata[M+1][N+1])
{
	int i, j1, j2;

	/* Calculate the m * m covariance matrix. */
	#pragma hmppcg grid blocksize 32 X 8
    
	#pragma hmppcg noParallel
	for (j1 = 1; j1 <= M; j1++)
	{      
		#pragma hmppcg parallel
		for (j2 = j1; j2 <= M; j2++)
		{
			psymmat[j1][j2] = 0.0;

			#pragma hmppcg unroll 4, split, guarded
			#pragma hmppcg noParallel
			for (i = 1; i <= N; i++)
			{
				psymmat[j1][j2] += pdata[i][j1] * pdata[i][j2];
			}
			psymmat[j2][j1] = psymmat[j1][j2];
		}
	}
}


void init_arrays(DATA_TYPE data[M+1][N+1], DATA_TYPE data_Gpu[M+1][N+1])
{
	int i, j;

	for (i = 1; i < (M+1); i++)
	{
		for (j = 1; j < (N+1); j++)
		{
			data[i][j] = ((DATA_TYPE) i*j) / M;
			data_Gpu[i][j] = ((DATA_TYPE) i*j) / M;
		}
	}
}


void compareResults(DATA_TYPE symmat[M+1][N+1], DATA_TYPE symmat_outputFromGpu[M+1][N+1])
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
			}			
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;
	double t_start, t_end;

	/* Array declaration */
	DATA_TYPE float_n = 321414134.01;
	DATA_TYPE data[M + 1][N + 1];
	DATA_TYPE data_Gpu[M + 1][N + 1];
	DATA_TYPE symmat[M + 1][M + 1];
	DATA_TYPE symmat_outputFromGpu[M + 1][M + 1];	
	DATA_TYPE mean[M + 1];
	DATA_TYPE mean_Gpu[M + 1];

	/* Initialize array. */
	init_arrays(data, data_Gpu);
    
	#pragma hmpp <group1> allocate
	#pragma hmpp <group1> loopa advancedload, args[pmean;pdata;pfloat_n]
    
	#pragma hmpp <group1> loopc advancedload, args[psymmat]

	t_start = rtclock();
	
	#pragma hmpp <group1> loopa callsite, args[pmean;pdata;pfloat_n].advancedload=true, asynchronous
	covarLoopa(mean_Gpu, data_Gpu, float_n);
	#pragma hmpp <group1> loopa synchronize
	#pragma hmpp <group1> loopb callsite, args[pdata;pmean].advancedload=true, asynchronous
	covarLoopb(data_Gpu, mean_Gpu);
	#pragma hmpp <group1> loopb synchronize
	#pragma hmpp <group1> loopc callsite, args[psymmat;pdata].advancedload=true, asynchronous
	covarLoopc(symmat_outputFromGpu, data_Gpu);
	#pragma hmpp <group1> loopc synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
    
	#pragma hmpp <group1> loopb delegatedstore, args[pmean]
    
	#pragma hmpp <group1> loopc delegatedstore, args[psymmat;pdata]
	#pragma hmpp <group1> release
	
	t_start = rtclock();
	
	covarLoopa(mean, data, float_n);
	covarLoopb(data, mean);
	covarLoopc(symmat, data);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(symmat, symmat_outputFromGpu);

	return 0;
}
