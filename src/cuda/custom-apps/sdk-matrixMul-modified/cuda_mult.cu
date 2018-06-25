#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>
//#include <helper_cuda.h>

// the number of thread
#define BLOCK_SIZE 32

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void mult(float *A, float *B, float *C, int size){
  	int i, j, k;
  	float sum = 0.0;

	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			for(k = 0; k < size; k++){
				sum += A[i * size + k] * B[k * size + j];
			}
		
			C[i * size + j] = sum;
			sum = 0.0;
		}
	}

}

__global__ void mult_gpu(float *A, float *B, float *C, int wA, int wB){
   // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    //float Csub[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    //float Csub[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //float Csub[4] = {0, 0, 0, 0};
    //float Csub[2] = {0, 0};
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
/*
	As[ty + 8][tx] = A[a + wA * (ty + 8) + tx];
        Bs[ty + 8][tx] = B[b + wB * (ty + 8) + tx];

	As[ty + 16][tx] = A[a + wA * (ty + 16) + tx];
	Bs[ty + 16][tx] = B[b + wB * (ty + 16) + tx];

	As[ty + 24][tx] = A[a + wA * (ty + 24) + tx];
        Bs[ty + 24][tx] = B[b + wB * (ty + 24) + tx];

	As[ty + 32][tx] = A[a + wA * (ty + 32) + tx];
        Bs[ty + 32][tx] = B[b + wB * (ty + 32) + tx];

        As[ty + 40][tx] = A[a + wA * (ty + 40) + tx];
        Bs[ty + 40][tx] = B[b + wB * (ty + 40) + tx];

        As[ty + 48][tx] = A[a + wA * (ty + 48) + tx];
        Bs[ty + 48][tx] = B[b + wB * (ty + 48) + tx];

	As[ty + 56][tx] = A[a + wA * (ty + 56) + tx];
        Bs[ty + 56][tx] = B[b + wB * (ty + 56) + tx];

	As[ty + 64][tx] = A[a + wA * (ty + 64) + tx];
        Bs[ty + 64][tx] = B[b + wB * (ty + 64) + tx];

        As[ty + 72][tx] = A[a + wA * (ty + 72) + tx];
        Bs[ty + 72][tx] = B[b + wB * (ty + 72) + tx];

        As[ty + 80][tx] = A[a + wA * (ty + 80) + tx];
        Bs[ty + 80][tx] = B[b + wB * (ty + 80) + tx];

        As[ty + 88][tx] = A[a + wA * (ty + 88) + tx];
        Bs[ty + 88][tx] = B[b + wB * (ty + 88) + tx];

        As[ty + 96][tx] = A[a + wA * (ty + 96) + tx];
        Bs[ty + 96][tx] = B[b + wB * (ty + 96) + tx];

        As[ty + 104][tx] = A[a + wA * (ty + 104) + tx];
        Bs[ty + 104][tx] = B[b + wB * (ty + 104) + tx];

        As[ty + 112][tx] = A[a + wA * (ty + 112) + tx];
        Bs[ty + 112][tx] = B[b + wB * (ty + 112) + tx];

	As[ty + 120][tx] = A[a + wA * (ty + 120) + tx];
        Bs[ty + 120][tx] = B[b + wB * (ty + 120) + tx];
*/


	
        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
	    Csub += As[ty][k] * Bs[k][tx];
/*
            Csub[0] += As[ty][k] * Bs[k][tx];
	    Csub[1] += As[ty + 8][k] * Bs[k][tx];

	    Csub[2] += As[ty + 16][k] * Bs[k][tx];
            Csub[3] += As[ty + 24][k] * Bs[k][tx];

	    Csub[4] += As[ty + 32][k] * Bs[k][tx];
            Csub[5] += As[ty + 40][k] * Bs[k][tx];

            Csub[6] += As[ty + 48][k] * Bs[k][tx];
            Csub[7] += As[ty + 56][k] * Bs[k][tx];

	    Csub[8] += As[ty + 64][k] * Bs[k][tx];
            Csub[9] += As[ty + 72][k] * Bs[k][tx];

            Csub[10] += As[ty + 80][k] * Bs[k][tx];
            Csub[11] += As[ty + 88][k] * Bs[k][tx];

            Csub[12] += As[ty + 96][k] * Bs[k][tx];
            Csub[13] += As[ty + 104][k] * Bs[k][tx];

            Csub[14] += As[ty + 112][k] * Bs[k][tx];
            Csub[15] += As[ty + 120][k] * Bs[k][tx];
*/

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

/*
    C[c + wB * ty + tx] = Csub[0];
    C[c + wB * (ty + 8) + tx] = Csub[1];

    C[c + wB * (ty + 16) + tx] = Csub[2];
    C[c + wB * (ty + 24) + tx] = Csub[3];

    C[c + wB * (ty + 32) + tx] = Csub[4];
    C[c + wB * (ty + 40) + tx] = Csub[5];

    C[c + wB * (ty + 48) + tx] = Csub[6];
    C[c + wB * (ty + 56) + tx] = Csub[7];

    C[c + wB * (ty + 64) + tx] = Csub[8];
    C[c + wB * (ty + 72) + tx] = Csub[9];

    C[c + wB * (ty + 80) + tx] = Csub[10];
    C[c + wB * (ty + 88) + tx] = Csub[11];

    C[c + wB * (ty + 96) + tx] = Csub[12];
    C[c + wB * (ty + 104) + tx] = Csub[13];

    C[c + wB * (ty + 112) + tx] = Csub[14];
    C[c + wB * (ty + 120) + tx] = Csub[15];
*/


}

int main(int argc, char *argv[]){
  	int i;
  	float *A, *B, *C, *D;
  	float *A_dev, *B_dev, *C_dev;
  	double start_timer, end_timer;

	int MROW = atoi(argv[1]);
	int MSIZE = MROW * MROW;

    	A = (float*)malloc(sizeof(float)*MSIZE);
    	cudaMalloc(&A_dev, MSIZE*sizeof(float));
    	B = (float*)malloc(sizeof(float)*MSIZE);
    	cudaMalloc(&B_dev, MSIZE*sizeof(float));
    	C = (float*)malloc(sizeof(float)*MSIZE);
    	cudaMalloc(&C_dev, MSIZE*sizeof(float));
   	D = (float*)malloc(sizeof(float)*MSIZE);

	srand(time(NULL)); 
  	// Init matrix
    	for(i = 0; i < MSIZE; i++){
      		//A[i] = (i%MROW)+1;
		A[i] = ((double) rand() / (RAND_MAX)) + 1;
      		//B[i] = (i%MCOL)+1;
		B[i] = ((double) rand() / (RAND_MAX)) + 1;
      		C[i] = 0;
      		D[i] = 0;
    	}
 	
	//transfer data to device
    	cudaMemcpy(A_dev, A, MSIZE*sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(B_dev, B, MSIZE*sizeof(float), cudaMemcpyHostToDevice);
  	
  	cudaDeviceSynchronize();
  
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE/1);
  	dim3 grid(MROW / BLOCK_SIZE, MROW / BLOCK_SIZE);
  	printf("block:%d, thread:%d\n", (MROW / BLOCK_SIZE) * (MROW / BLOCK_SIZE), BLOCK_SIZE * BLOCK_SIZE);
  	start_timer = my_timer();
    	mult_gpu<<<grid, threads, 0>>>(A_dev, B_dev, C_dev, MROW, MROW);
  	cudaDeviceSynchronize();
  	end_timer = my_timer();
  	printf("The GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);
  
  	//transfer data back to host
    	cudaMemcpy(C, C_dev, MSIZE*sizeof(int), cudaMemcpyDeviceToHost);
  	cudaDeviceSynchronize();
#if 1
  	start_timer = my_timer();
    	mult(A, B, D, MROW);
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//Verification
    FILE *fcpu = fopen("cpu_result.txt", "w");
    FILE *fgpu = fopen("gpu_result.txt", "w");
  	printf("Verifying\n");
    	for(i = 0; i < MSIZE; i++) {
      		if(abs(C[i] - D[i]) > 1e-2){
        		printf("Error:%f, %f\n", C[i], D[i]);
			//break;
      		}
            fprintf(fcpu, "%f\n",D[i]);
            fprintf(fgpu, "%f\n",C[i]);
        }
    fclose(fcpu);
    fclose(fgpu);
#endif

    	free(A);
    	cudaFree(A_dev);
    	free(B);
    	cudaFree(B_dev);
    	free(C);
    	cudaFree(C_dev);
    	free(D);
  return 0;
}
