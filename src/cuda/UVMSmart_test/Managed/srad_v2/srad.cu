// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
    	int rows, cols, size_I, size_R, niter = 10, iter;
    	float *I, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

	float *J_shared;
    	float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;
	
	unsigned int r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
			fprintf(stderr, "rows and cols must be multiples of 16\n");
			exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
		
	}
    	else{
		usage(argc, argv);
    	}



	size_I = cols * rows;
    	size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;

	//Allocate device memory
    	cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& N_C, sizeof(float)* size_I);
	
	//Allocate managed memory
    	cudaMallocManaged((void**)& J_shared, sizeof(float)* size_I);
	
	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    	for (int k = 0;  k < size_I; k++ ) {
     		J_shared[k] = (float)exp(I[k]) ;
    	}
	printf("Start the SRAD main loop\n");

#ifdef PREF
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);
#endif
	for (iter=0; iter< niter; iter++) {     
		sum=0; sum2=0;
        	for (int i=r1; i<=r2; i++) {
            		for (int j=c1; j<=c2; j++) {
                		tmp   = J_shared[i * cols + j];
                		sum  += tmp ;
                		sum2 += tmp*tmp;
            		}
        	}
        	meanROI = sum / size_R;
        	varROI  = (sum2 / size_R) - meanROI*meanROI;
        	q0sqr   = varROI / (meanROI*meanROI);

#ifdef PREF
		int device = -1;
		cudaGetDevice(&device);
		cudaMemPrefetchAsync(J_shared, sizeof(float)* size_I, device, stream1);
#endif
		//Currently the input size must be divided by 16 - the block size
		int block_x = cols/BLOCK_SIZE ;
    		int block_y = rows/BLOCK_SIZE ;

    		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(block_x , block_y);

		//Run kernels
#ifdef PREF
		srad_cuda_1<<<dimGrid, dimBlock, 0, stream2>>>(E_C, W_C, N_C, S_C, J_shared, C_cuda, cols, rows, q0sqr); 
		srad_cuda_2<<<dimGrid, dimBlock, 0, stream2>>>(E_C, W_C, N_C, S_C, J_shared, C_cuda, cols, rows, lambda, q0sqr); 
#else
		srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_shared, C_cuda, cols, rows, q0sqr); 
		srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_shared, C_cuda, cols, rows, lambda, q0sqr); 
#endif

		// Wait for GPU to finish before accessing on host
		cudaDeviceSynchronize();

	}

    cudaThreadSynchronize();

#define OUTPUT

#ifdef OUTPUT
    //Printing output	
    printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
	for ( int j = 0 ; j < cols ; j++){
        	printf("%.5f ", J_shared[i * cols + j]); 
	}	
   	printf("\n"); 
    }
#endif 

	printf("Computation Done\n");

	free(I);
	free(c);

    	cudaFree(C_cuda);
	cudaFree(E_C);
	cudaFree(W_C);
	cudaFree(N_C);
	cudaFree(S_C);

	cudaFree(J_shared);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

