/**
 * mvt.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define N 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp mvt codelet, target=OpenCL, args[x1,x2].io=inout
void runMvt(DATA_TYPE a[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N]){
	
	int i, j;
	
	#pragma hmppcg grid blocksize 32 X 8
		
	for (i=0; i<N; i++) 
	{		
        for (j=0; j<N; j++) 
		{
			x1[i] = x1[i] + a[i][j] * y1[j];
        }
	}


	#pragma hmppcg grid blocksize 32 X 8
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
			x2[i] = x2[i] + a[j][i] * y2[j];
		}
	}
}


void init_array(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x1_Gpu[N], DATA_TYPE x2[N], DATA_TYPE x2_Gpu[N], DATA_TYPE y1[N], DATA_TYPE y2[N])
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x1_Gpu[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		x2_Gpu[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
	}
}


void compareResults(DATA_TYPE x1[N], DATA_TYPE x1_outputFromGpu[N], DATA_TYPE x2[N], DATA_TYPE x2_outputFromGpu[N])
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}



int main()
{
	double t_start, t_end;
	
	DATA_TYPE a[N][N];
	DATA_TYPE x1[N];
	DATA_TYPE x1_outputFromGpu[N];
	DATA_TYPE x2[N];
	DATA_TYPE x2_outputFromGpu[N];
	DATA_TYPE y1[N];
	DATA_TYPE y2[N];


	//initialize the arrays for running on the CPU and GPU
    	init_array(a, x1, x1_outputFromGpu, x2, x2_outputFromGpu, y1, y2);

	#pragma hmpp mvt allocate

	#pragma hmpp mvt advancedload, args[a,x1,x2,y1,y2]

	t_start = rtclock();
	
	//run the algorithm on the GPU
	#pragma hmpp mvt callsite, args[x1,x2].advancedload=true, asynchronous
	runMvt(a, x1_outputFromGpu, x2_outputFromGpu, y1, y2); // parameters are initialized in decls.h and are initialized with init_array()

	#pragma hmpp mvt synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lf\n", t_end - t_start);
	
	#pragma hmpp mvt delegatedstore, args[x1,x2]

	#pragma hmpp mvt release

	t_start = rtclock();
	
	//run the algorithm on the CPU
	runMvt(a, x1, x2, y1, y2);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lf\n", t_end - t_start);
	
	compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);

	return 0;
}
