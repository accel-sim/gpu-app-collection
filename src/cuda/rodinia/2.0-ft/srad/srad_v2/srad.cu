// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <srad.h>


// includes, kernels
#include <srad_kernel.cu>

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);

#undef TIMER

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
//	CUT_CHECK_DEVICE();
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
#endif

	unsigned int r1, r2, c1, c2;
	float *c;
    
	const char* infile;
	const char* goldfile;
	if (argc == 9) {
		infile = argv[1];  //matrix input file
		r1   = atoi(argv[2]);  //y1 position of the speckle
		r2   = atoi(argv[3]);  //y2 position of the speckle
		c1   = atoi(argv[4]);  //x1 position of the speckle
		c2   = atoi(argv[5]);  //x2 position of the speckle
		lambda = atof(argv[6]); //Lambda value
		niter = atoi(argv[7]); //number of iterations
		goldfile = argv[8];
	}
	else {
		printf("Wrong Usage: infile r1 r2 c1 c2 lambda niter\n");
		exit(1);
	}

	FILE* ifile = fopen(infile, "r");
	if (!ifile) {
		printf("Could not open input file %s\n", infile);
		exit(1);
	}
	fscanf(ifile, "%d", &cols);
	fscanf(ifile, "%d", &rows);
	printf("Matrix size: %dx%d\n", cols, rows);
	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);



#ifdef CPU

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;

	//Generate a random matrix
	//random_matrix(I, rows, cols);
	//read matrix from file
	for (int i=0; i<rows*cols; i++){
		fscanf(ifile, "%f", &I[i]);
	}
	printf("\n");

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }


#ifdef TIMER

    unsigned int timer_1 = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer_1));
    CUT_SAFE_CALL( cutStartTimer( timer_1));

#endif


 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

#ifdef CPU
        
		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
	}

#endif // CPU



#ifdef GPU

    float *J_cuda;
    float *C_cuda;
	
	float *E_C, *W_C, *N_C, *S_C;

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);
    
	//Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& N_C, sizeof(float)* size_I);

	//Copy data from main memory to device memory
	cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

	//Copy data from device memory to main memory
    cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

#endif   
}

    cudaThreadSynchronize();

#ifdef TIMER
		CUT_SAFE_CALL( cutStopTimer( timer_1 ));
		printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer_1));
		CUT_SAFE_CALL( cutDeleteTimer( timer_1 ));
#endif


#ifdef OUTPUT
   //Printing output	
		printf("Printing Output:\n"); 
   for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
 #endif 

#ifdef CHECKSUM 
   unsigned long long int checksum = 0; 
   double checksum2=0;
   for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         checksum += ((unsigned int*)J)[i * cols + j]; 
		 checksum2+= J[i*cols+j];
		}	
   }
   printf("checksum = %llu\n", checksum); 
   printf("checksum2 = %f\n", checksum2); 
#endif
	
	FILE* gf = fopen(goldfile, "r");
	printf("Reading gold checksum2 from: %s\n", goldfile);
	if (!gf) {
		printf("Failed to open file %s\n", goldfile);
		exit(1);
	}

	float gold;
	fscanf(gf, "%f", &gold);
	printf("gold checksum2 = %f\n", gold);
	fclose(gf);
	double error = gold-checksum2;
	if (error < 0) error = -error;
	if (error < (cols*rows*0.0001)) {
		printf("\nPASSED\n");
	}
	else {
		printf("\nFAILED\n");
	}

	free(I);
	free(J);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#endif 
	free(c);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}
