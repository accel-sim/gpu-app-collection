/**
 * bicg.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define NX 4096
#define NY 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp bicg codelet, target=OpenCL, args[a;p;q;r;s].io=inout
void runBicg(DATA_TYPE a[NX][NY], DATA_TYPE p[NY], DATA_TYPE q[NX], DATA_TYPE r[NX], DATA_TYPE s[NY])
{
	int i, j, k, l;

	#pragma hmppcg grid blocksize 32 X 8
	for (i = 0; i < NY; i++)
	{
		s[i] = 0;
	}

	#pragma hmppcg grid blocksize 32 X 8
	#pragma hmppcg permute j, i   
	#pragma hmppcg tile i:4
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			s[j] = s[j] + r[i] * a[i][j];
		}
	}

	#pragma hmppcg grid blocksize 32 X 8
	for (i = 0; i < NX; i++)
	{
		q[i] = 0;

		for (j = 0; j < NY; j++)
		{
			q[i] = q[i] + a[i][j] * p[j];
		}
	}
}


void init_array(DATA_TYPE A[NX][NY], DATA_TYPE p[NY], DATA_TYPE r[NX])
{
	int i, j;

  	for (i = 0; i < NX; i++)
	{
		r[i] = i * M_PI;

		for (j = 0; j < NY; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
 	}

	for (i = 0; i < NY; i++)
	{
		p[i] = i * M_PI;
	}
}


void compareResults(DATA_TYPE s[NY], DATA_TYPE s_outputFromGpu[NY], DATA_TYPE q[NX], DATA_TYPE q_outputFromGpu[NX])
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<NX; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<NY; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char** argv)
{
	double t_start, t_end;
	
	/* Array declaration */
	DATA_TYPE A[NX][NY];
	DATA_TYPE p[NY];
	DATA_TYPE q[NX];
	DATA_TYPE q_outputFromGpu[NX];
	DATA_TYPE r[NX];
	DATA_TYPE s[NY];
	DATA_TYPE s_outputFromGpu[NY];

	/* Initialize array. */
	init_array(A, p, r);
    
	#pragma hmpp bicg allocate

	#pragma hmpp bicg advancedload, args[a;p;q;r;s]

	t_start = rtclock();

	#pragma hmpp bicg callsite, args[a;p;q;r;s].advancedload=true, asynchronous
	runBicg(A, p, q_outputFromGpu, r, s_outputFromGpu);
    
	#pragma hmpp bicg synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp bicg delegatedstore, args[a;p;q;r;s]
    
	#pragma hmpp bicg release

	t_start = rtclock();

	runBicg(A, p, q, r, s);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(s, s_outputFromGpu, q, q_outputFromGpu);
	return 0;
}
