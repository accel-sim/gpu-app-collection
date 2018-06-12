/**
 * fdtd2d.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/* Problem dimensions */
#define NX 2048
#define NY 2048
#define T_MAX 500

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp fdtd codelet, target=CUDA, args[ex,ey,hz].io=inout
void runFdtd(DATA_TYPE fict[T_MAX], DATA_TYPE ex[NX][NY+1], DATA_TYPE ey[NX+1][NY], DATA_TYPE hz[NX][NY])
{
	int t, i, j;
	
	#pragma hmppcg grid blocksize 32 X 8
	
	#pragma hmppcg noParallel
	for(t=0; t< T_MAX; t++)  
	{		
		#pragma hmppcg parallel
		for (j=0; j < NY; j++)
		{
			ey[0][j] = fict[t];
		}
	
		
		#pragma hmppcg parallel
		for (i = 1; i < NX; i++)
		{
			#pragma hmppcg unroll 2, split, guarded
			#pragma hmppcg parallel
       		for (j = 0; j < NY; j++)
			{
       			ey[i][j] = ey[i][j] - 0.5*(hz[i][j] - hz[i-1][j]);
			
			}
		}
	
		#pragma hmppcg parallel
		for (i = 0; i < NX; i++)
		{
			#pragma hmppcg parallel
			for (j = 1; j < NY; j++)
			{
              		ex[i][j] = ex[i][j] - 0.5*(hz[i][j] - hz[i][j-1]);
			}
		}

		
		#pragma hmppcg parallel
       	for (i = 0; i < NX; i++)
		{
			#pragma hmppcg unroll 4, split, guarded
			#pragma hmppcg parallel
			for (j = 0; j < NY; j++)
			{
              		hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
			}
		}
	}
}


void init_arrays(DATA_TYPE _fict_[T_MAX], DATA_TYPE ex[NX][NY+1], DATA_TYPE ex_Gpu[NX][NY+1], DATA_TYPE ey[NX+1][NY], DATA_TYPE ey_Gpu[NX+1][NY], DATA_TYPE hz[NX][NY], DATA_TYPE hz_Gpu[NX][NY])
{
	int i, j;

  	for (i = 0; i < T_MAX; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i][j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ex_Gpu[i][j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i][j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			ey_Gpu[i][j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i][j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
			hz_Gpu[i][j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}



void compareResults(DATA_TYPE hz[NX][NY], DATA_TYPE hz_outputFromGpu[NX][NY])
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz[i][j], hz_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main()
{
	double t_start, t_end;

	DATA_TYPE ex[NX][NY+1];
	DATA_TYPE ex_Gpu[NX][NY+1];
	DATA_TYPE ey[NX+1][NY];
	DATA_TYPE ey_Gpu[NX+1][NY];
	DATA_TYPE fict[T_MAX];
	DATA_TYPE hz[NX][NY];
	DATA_TYPE hz_outputFromGpu[NX][NY];

	#pragma hmpp fdtd allocate

	#pragma hmpp fdtd advancedload, args[fict,ex,ey,hz]
		
	init_arrays(fict, ex, ex_Gpu, ey, ey_Gpu, hz, hz_outputFromGpu);

	t_start = rtclock();
	
	#pragma hmpp fdtd callsite, args[fict,ex,ey,hz].advancedload=true, asynchronous
	runFdtd(fict, ex_Gpu, ey_Gpu, hz_outputFromGpu);

	#pragma hmpp fdtd synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lf\n", t_end - t_start);
	
	#pragma hmpp fdtd delegatedstore, args[ex,ey,hz]

	#pragma hmpp fdtd release

	t_start = rtclock();
	
	runFdtd(fict, ex, ey, hz);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lf\n", t_end - t_start);
	
	compareResults(hz, hz_outputFromGpu);

	return 0;
}

