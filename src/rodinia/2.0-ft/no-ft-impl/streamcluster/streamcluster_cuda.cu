/***********************************************
	streamcluster_cuda.cu
	: parallelized code of streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
***********************************************/

/* For a given point x, find the cost of the following operation:
 * -- open a facility at x if there isn't already one there,
 * -- for points y such that the assignment distance of y exceeds dist(y, x),
 *    make y a member of x,
 * -- for facilities y such that reassigning y and all its members to x 
 *    would save cost, realize this closing and reassignment.
 * 
 * If the cost of this operation is negative (i.e., if this entire operation
 * saves cost), perform this operation and return the amount of cost saved;
 * otherwise, do nothing.
 */

/* numcenters will be updated to reflect the new number of centers */
/* z is the facility cost, x is the number of this point in the array 
   points */	 

#include "streamcluster_header.cu"

using namespace std;

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
#define PROFILE

/* host memory analogous to device memory */
float *work_mem_h;
static float *coord_h;
float *gl_lower;
Point *p;

/* device memory */
float *work_mem_d;
float *coord_d;
int  *center_table_d;
bool  *switch_membership_d;

static int c;			// counters

/* kernel */
__global__ void
pgain_kernel(	int num,
						int dim,
						long x,
						Point *p,
						int K,
						float *coord_d,
						float *work_mem_d,			
						int *center_table_d,
						bool *switch_membership_d
					)
{	
	/* block ID and global thread ID */
	const int block_id  = blockIdx.x + gridDim.x * blockIdx.y;
	const int thread_id = blockDim.x * block_id + threadIdx.x;
		
	extern __shared__ float coord_s[];							// shared memory for coordinate of point[x]
	
	/* coordinate mapping of point[x] to shared mem */
	if(threadIdx.x == 0)
		for(int i=0; i<dim; i++) { coord_s[i] = coord_d[i*num + x]; }
	__syncthreads();
	
	/* cost between this point and point[x]: euclidean distance multiplied by weight */
	float x_cost = 0.0;
	for(int i=0; i<dim; i++)
		x_cost += (coord_d[(i*num)+thread_id]-coord_s[i]) * (coord_d[(i*num)+thread_id]-coord_s[i]);
	x_cost = x_cost * p[thread_id].weight;
	
	float current_cost = p[thread_id].cost;
	
	/* if computed cost is less then original (it saves), mark it as to reassign */
	float *lower = &work_mem_d[thread_id*(K+1)];
	if ( x_cost < current_cost ) {
		switch_membership_d[thread_id] = 1;
	    lower[K] += x_cost - current_cost;
	}
	/* if computed cost is larger, save the difference */
	else {
	    int assign = p[thread_id].assign;
	    lower[center_table_d[assign]] += current_cost - x_cost;
	}
}

void quit(char *message){
	printf("%s\n", message);
	exit(1);
}

void allocDevMem(int num, int dim, int kmax){
	if( cudaMalloc((void**) &work_mem_d,  kmax * num * sizeof(float))!= cudaSuccess) quit("error allocating device memory");	
	if( cudaMalloc((void**) &center_table_d,  num * sizeof(int))!= cudaSuccess) quit("error allocating device memory");
	if( cudaMalloc((void**) &switch_membership_d,  num * sizeof(bool))!= cudaSuccess) quit("error allocating device memory");
	if( cudaMalloc((void**) &p,  num * sizeof(Point))!= cudaSuccess) quit("error allocating device memory");
	if( cudaMalloc((void**) &coord_d,  num * dim * sizeof(float))!= cudaSuccess) quit("error allocating device memory");
}

void freeDevMem(){	
	cudaFree(work_mem_d);
	cudaFree(center_table_d);
	cudaFree(switch_membership_d);	
	cudaFree(p);
	cudaFree(coord_d);
	cudaFreeHost(work_mem_h);
	free(coord_h);
	free(gl_lower);
}

float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, bool *switch_membership,
							double *serial, double *cpu_gpu_memcpy, double *memcpy_back, double *gpu_malloc, double *kernel)
{	
	cudaSetDevice(0);
#ifdef PROFILE
	double t1 = gettime();
#endif
	int K	= *numcenters ;						// number of centers
	int num    =   points->num;				// number of points
	int dim     =   points->dim;				// number of dimension
	kmax++;
	
	
	
	/***** build center index table *****/
	int count = 0;
	for( int i=0; i<num; i++){
		if( is_center[i] )
			center_table[i] = count++;
	}
	
#ifdef PROFILE
	double t2 = gettime();
	*serial += t2 - t1;
#endif
	
	
	/***** initial memory allocation and preparation for transfer : execute once *****/
	if( c == 0 ) {
#ifdef PROFILE
		double t3 = gettime();
#endif
		allocDevMem(num, dim, kmax);
#ifdef PROFILE
		double t4 = gettime();
		*gpu_malloc += t4 - t3;
#endif
		
		coord_h = (float*) malloc( num * dim * sizeof(float));								// coordinates (host)
		gl_lower = (float*) malloc( kmax * sizeof(float) );
		cudaMallocHost( (void**)&work_mem_h,  kmax * num * sizeof(float) );
		
		/* prepare mapping for point coordinates */
		for(int i=0; i<dim; i++){
			for(int j=0; j<num; j++)
				coord_h[ (num*i)+j ] = points->p[j].coord[i];
		}
#ifdef PROFILE		
		double t5 = gettime();
		*serial += t5 - t4;
#endif
		/* copy coordinate to device memory */
		cudaMemcpy( switch_membership_d,  switch_membership,  num*sizeof(bool),  cudaMemcpyHostToDevice);
		cudaMemcpy( coord_d,  coord_h,  num*dim*sizeof(float),  cudaMemcpyHostToDevice);
#ifdef PROFILE
		double t6 = gettime();
		*cpu_gpu_memcpy += t6 - t5;
#endif
	}	
	
#ifdef PROFILE
	double t7 = gettime();
#endif
	
	
	/***** memory transfer from host to device *****/
	/* copy to device memory */
	cudaMemcpy( center_table_d,  center_table,  num*sizeof(int),  cudaMemcpyHostToDevice);
	cudaMemcpy( p,  points->p,  num * sizeof(Point),  cudaMemcpyHostToDevice);
	/* initialize device memory */
	cudaMemset( switch_membership_d, 0, num 		* sizeof(bool) );
	cudaMemset( work_mem_d, 				0, kmax * num * sizeof(float) );
	
#ifdef PROFILE
	double t8 = gettime();
	*cpu_gpu_memcpy += t8 - t7;
#endif
	
	
	/***** kernel execution *****/
	/* Determine the number of thread blocks in the x- and y-dimension */
	int num_blocks 	 = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
	int num_blocks_y  = (int) ((float) (num_blocks + MAXBLOCKS - 1)    / (float) MAXBLOCKS);
	int num_blocks_x  = (int) ((float) (num_blocks+num_blocks_y - 1)   / (float) num_blocks_y);	
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);
	size_t smSize = dim * sizeof(float);
#ifdef PROFILE
	double t9 = gettime();
#endif
	pgain_kernel<<< grid_size, THREADS_PER_BLOCK,  smSize>>>(	
																											num,								// in:	# of data
																											dim,									// in:	dimension of point coordinates
																											x,										// in:	point to open a center at
																											p,										// out:	data point array
																											K,										// in:	number of centers
																											coord_d,							// in:	array of point coordinates
																											work_mem_d,					// out:	cost and lower field array
																											center_table_d,				// in:	center index table
																											switch_membership_d		// out:  changes in membership
																										  );
	cudaThreadSynchronize();
	
#ifdef PROFILE
	double t10 = gettime();
	*kernel += t10 - t9;
#endif
	
	
	/***** copy back to host for CPU side work *****/
	cudaMemcpy(work_mem_h, work_mem_d, (K+1) *num*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(switch_membership, switch_membership_d, num * sizeof(bool), cudaMemcpyDeviceToHost);

#ifdef PROFILE
	double t11 = gettime();
	*memcpy_back += t11 - t10;
#endif
	
	
	/****** cpu side work *****/
	int numclose = 0;
	float gl_cost = z;
	
	/* compute the number of centers to close if we are to open i */
	for(int i=0; i < num; i++){
		if( is_center[i] ) {
			float low = z;
			
		    for( int j = 0; j < num; j++ )
				low += work_mem_h[ j*(K+1) + center_table[i] ];
				
		    gl_lower[center_table[i]] = low;
				
		    if ( low > 0 ) {
				numclose++;				
				work_mem_h[i*(K+1)+K] -= low;
		    }
		}
		gl_cost += work_mem_h[i*(K+1)+K];
	}
	
	
	/* if opening a center at x saves cost (i.e. cost is negative) do so
		otherwise, do nothing */
	if ( gl_cost < 0 ) {
		for(int i=0; i<num; i++){
		
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
		    if ( switch_membership[i] || close_center ) {
				points->p[i].cost = points->p[i].weight * dist(points->p[i], points->p[x], points->dim);
				points->p[i].assign = x;
		    }
	    }
		
		for(int i=0; i<num; i++){
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
				is_center[i] = false;
		}
		
		is_center[x] = true;
		*numcenters = *numcenters +1 - numclose;
	}
	else
		gl_cost = 0;  // the value we'

#ifdef PROFILE
	double t12 = gettime();
	*serial += t12 - t11;
#endif
	c++;
	return -gl_cost;
}
