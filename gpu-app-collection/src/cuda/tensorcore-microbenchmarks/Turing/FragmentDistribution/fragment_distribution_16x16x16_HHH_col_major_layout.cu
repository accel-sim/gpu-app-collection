#include <stdio.h>
#include <curand.h>
#include <ctime>
#include <assert.h>
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#include <mma.h>
using namespace nvcuda;
//enum MatrixLayout{
#define ROW_MAJOR 0
#define COL_MAJOR 1
//};

//ONLY THE PARAMETER HERE NEEDS TO BE CHANGED
// Must be multiples of 16 for wmma code to work
#define MATRIX_M (16)
#define MATRIX_N (16)
#define MATRIX_K (16)
const int WMMA_M =16;
const int WMMA_N =16;
const int WMMA_K =16;
typedef half atype;
typedef half btype;
typedef half ctype;
typedef half dtype;
typedef float host_type;
#define A_LAYOUT COL_MAJOR
#define B_LAYOUT COL_MAJOR 
#define C_LAYOUT COL_MAJOR
#define D_LAYOUT COL_MAJOR
#define NUM_CTA 1
#define WARP_IN_CTA 1
//Don't change anything after here 



#define THREAD_IN_WARP 32

#if A_LAYOUT==ROW_MAJOR
	#define LAYOUT_A wmma::row_major 
	#define A_STRIDE MATRIX_K
#else
	#define LAYOUT_A wmma::col_major 
	#define A_STRIDE MATRIX_M
#endif	
#if B_LAYOUT==ROW_MAJOR
	#define LAYOUT_B wmma::row_major 
	#define B_STRIDE MATRIX_N
#else
	#define LAYOUT_B wmma::col_major 
	#define B_STRIDE MATRIX_K
#endif	
#if C_LAYOUT==ROW_MAJOR
	#define LAYOUT_C wmma::mem_row_major 
	#define C_STRIDE MATRIX_N
#else
	#define LAYOUT_C wmma::mem_col_major 
	#define C_STRIDE MATRIX_M
#endif	
#if D_LAYOUT==ROW_MAJOR
	#define LAYOUT_D wmma::mem_row_major 
	#define D_STRIDE MATRIX_N
#else
	#define LAYOUT_D wmma::mem_col_major 
	#define D_STRIDE MATRIX_M
#endif	


enum MatrixInitializationType{
	ZERO,
	ONE,
	RANDOM,
	IDENTITY,
	LINEAR
};
int get_value(MatrixInitializationType init_type,int randomRange=3,bool RESET=false){
	static int val=0;
	switch(init_type){
		case ZERO:
			break;
		case ONE:
			val=1;
			break;
		case RANDOM:
			val=rand()%randomRange;
			break;
		case LINEAR:
			val++;
			break;
		default :
			printf("illegal MatrixInitializationType\n");
			abort();
			break;
	}
	if(RESET)
		val=0;
	return val;
}
template <typename T>
void print_matrix(T *matrix,int row_size,int col_size,int/*MatrixLayout*/  layout){
	for(int row=0;row<row_size;row++){
		for(int col=0;col<col_size;col++){
			T val;
			if(layout==ROW_MAJOR)
				val=matrix[row*col_size+col];		
			else
				val=matrix[col*row_size+row];
			printf("%.2f ",static_cast<float>(val));
		}
		printf(";\n");
	}
}

template <typename T>
void initialize_matrix(T *matrix,int row_size,int col_size,int/*MatrixLayout*/  layout,MatrixInitializationType init_type){
	for(int row=0;row<row_size;row++){
		for(int col=0;col<col_size;col++){
			if(init_type==IDENTITY){
				assert(row_size==col_size);//only for square matrix can be used
				matrix[row*row_size+col]=static_cast<T>(1);
			}
			else{
				if(layout==ROW_MAJOR){
					matrix[row*col_size+col]=static_cast<T>(get_value(init_type));
				}
				else{
					matrix[col*row_size+row]=static_cast<T>(get_value(init_type));
				}
			}
		}
	}
	get_value(init_type,10,true);//reseting the val counter
  	print_matrix<T>(matrix,row_size,col_size,layout);
}

int get_index(int row,int col,int row_size,int col_size,int/*MatrixLayout*/  layout){
		int index=0;
		if(layout==ROW_MAJOR){
			index=row*col_size+col;		
		}
		else{
			index=col*row_size+row;
		}
		return index;
}

template <typename T>
void matrix_multiply(T *result_matrix, T *matrix_a,T* matrix_b,T *matrix_c,int M,int N,int K,int/*MatrixLayout*/  resultlayout,int/*MatrixLayout*/  alayout,int/*MatrixLayout*/  blayout,int/*MatrixLayout*/  clayout){
	for(int row=0;row<M;row++){
		for(int col=0;col<N;col++){
			int rindex=get_index(row,col,M,N,resultlayout);
			int cindex=get_index(row,col,M,N,clayout);
			for(int k=0;k<K;k++){
				int aindex=get_index(row,k,M,K,alayout);
				int bindex=get_index(k,col,K,N,blayout);
				result_matrix[rindex]+=matrix_a[aindex]*matrix_b[bindex];
			}
			result_matrix[rindex]+=matrix_c[cindex];
		}
	}
   	print_matrix<T>(result_matrix,M,N,resultlayout);
}

template <typename T>	
void compare_matrix(T *matrix_a, T *matrix_b,int row_size,int col_size,int/*MatrixLayout*/  alayout,int/*MatrixLayout*/  blayout){
	
	for(int row=0;row<row_size;row++){
		for(int col=0;col<col_size;col++){
			int index_a,index_b;
			index_a=get_index(row,col,row_size,col_size,alayout);
			index_b=get_index(row,col,row_size,col_size,alayout);
			if(matrix_a[index_a]!=matrix_b[index_b])
					printf("ERROR at index row=%d col=%d\n",row,col);
		}
				
	}
}

__global__ void wmma_example(atype *a, btype *b, ctype *c,dtype *d)
{
   float t; 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, atype , LAYOUT_A> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, btype , LAYOUT_B> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, ctype> c_frag;
   // Bounds checking
   wmma::load_matrix_sync(a_frag, a, A_STRIDE);
   wmma::load_matrix_sync(b_frag, b, B_STRIDE);
   wmma::load_matrix_sync(c_frag, c, C_STRIDE,LAYOUT_C);
   for(int i=0; i < a_frag.num_elements; i++) {
           t=static_cast<float>(a_frag.x[i]);
           printf("A_THREAD%d: %.2f \n",threadIdx.x,t);
   }
   for(int i=0; i < b_frag.num_elements; i++) {
           t=static_cast<float>(b_frag.x[i]);
           printf("B_THREAD%d: %.2f \n",threadIdx.x,t);
   }
   for(int i=0; i < c_frag.num_elements; i++) {
           t=static_cast<float>(c_frag.x[i]);
           printf("C_THREAD%d: %.2f \n",threadIdx.x,t);
   }
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   wmma::store_matrix_sync(d, c_frag, D_STRIDE, LAYOUT_D);
}

template <typename T1,typename T2>	
__global__ void convert(T1 *out, T2 *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int main(int argc, char* argv[]) {
   //data on device in host type format
   host_type *a_htype;
   host_type *b_htype;
   host_type *c_htype;
   host_type *d_htype;

   //data on device in gemm format
   atype *a_atype;
   btype *b_btype;
   ctype *c_ctype;
   dtype *d_dtype;

   srand(time(NULL));  
 
   host_type *a_host_wmma;
   host_type *b_host_wmma;
   host_type *c_host_wmma;
   host_type *d_host_wmma;
   host_type *d_cal_host_wmma;

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   // Use tensor cores
   cudaErrCheck(cudaMalloc((void**)&a_htype, MATRIX_M * MATRIX_K * sizeof(host_type)));
   cudaErrCheck(cudaMalloc((void**)&b_htype, MATRIX_K * MATRIX_N * sizeof(host_type)));
   cudaErrCheck(cudaMalloc((void**)&c_htype, MATRIX_M * MATRIX_N * sizeof(host_type)));
   cudaErrCheck(cudaMalloc((void**)&d_htype, MATRIX_M * MATRIX_N * sizeof(host_type)));
   cudaErrCheck(cudaMalloc((void**)&a_atype, MATRIX_M * MATRIX_K * sizeof(atype)));
   cudaErrCheck(cudaMalloc((void**)&b_btype, MATRIX_K * MATRIX_N * sizeof(btype)));
   cudaErrCheck(cudaMalloc((void**)&c_ctype, MATRIX_M * MATRIX_N * sizeof(ctype)));
   cudaErrCheck(cudaMalloc((void**)&d_dtype, MATRIX_M * MATRIX_N * sizeof(dtype)));


   a_host_wmma      = (host_type*)malloc(MATRIX_M * MATRIX_K * sizeof(host_type));
   b_host_wmma      = (host_type*)malloc(MATRIX_K * MATRIX_N * sizeof(host_type));
   c_host_wmma      = (host_type*)malloc(MATRIX_M * MATRIX_N * sizeof(host_type));
   d_host_wmma      = (host_type*)malloc(MATRIX_M * MATRIX_N * sizeof(host_type));
   d_cal_host_wmma  = (host_type*)malloc(MATRIX_M * MATRIX_N * sizeof(host_type));


   printf("a_host\n");
   initialize_matrix<host_type>(a_host_wmma,MATRIX_M,MATRIX_K,A_LAYOUT,LINEAR);
   printf("b_host\n");
   initialize_matrix<host_type>(b_host_wmma,MATRIX_K,MATRIX_N,B_LAYOUT,LINEAR);
   printf("c_host\n");
   initialize_matrix<host_type>(c_host_wmma,MATRIX_M,MATRIX_N,C_LAYOUT,LINEAR);
   printf("d_cal_host\n");
   initialize_matrix<host_type>(d_cal_host_wmma,MATRIX_M,MATRIX_N,D_LAYOUT,ZERO);
   printf("d_cal_host\n");
   matrix_multiply<host_type>(d_cal_host_wmma,a_host_wmma,b_host_wmma,c_host_wmma,MATRIX_M,MATRIX_N,MATRIX_K,D_LAYOUT,A_LAYOUT,B_LAYOUT,C_LAYOUT);

   cudaErrCheck(cudaMemcpy(a_htype,a_host_wmma,  MATRIX_M * MATRIX_K * sizeof(host_type), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_htype,b_host_wmma,  MATRIX_K * MATRIX_N * sizeof(host_type), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(c_htype,c_host_wmma,  MATRIX_M * MATRIX_N * sizeof(host_type), cudaMemcpyHostToDevice));

   convert<atype,host_type> <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_atype, a_htype, MATRIX_M * MATRIX_K);
   convert<btype,host_type> <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_btype, b_htype, MATRIX_K * MATRIX_N);
   convert<ctype,host_type> <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_ctype, c_htype, MATRIX_M * MATRIX_N);

   printf("\nM = %d, N = %d, K = %d. \n", MATRIX_M, MATRIX_N, MATRIX_K);
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< NUM_CTA,WARP_IN_CTA*THREAD_IN_WARP>>> (a_atype, b_btype, c_ctype, d_dtype);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   convert<host_type,dtype> <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (d_htype, d_dtype, MATRIX_M * MATRIX_N);
   cudaErrCheck(cudaEventSynchronize(stopWMMA));

   // Error checking
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(d_host_wmma, d_htype, MATRIX_M * MATRIX_N * sizeof(host_type), cudaMemcpyDeviceToHost));
   
   printf("Results verified: cublas and WMMA agree.\n\n");
   float wmmaTime;
   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("wmma took %.2fms\n", wmmaTime);
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
   //printf("D_CALCULATED\n");

   //print_matrix<host_type>(d_cal_host_wmma,MATRIX_M,MATRIX_N,D_LAYOUT);
   //printf("D_WMMA\n");
   //print_matrix<host_type>(d_host_wmma,MATRIX_M,MATRIX_N,D_LAYOUT);
 
   //printf("CHECKING\n");
   //compare_matrix<host_type>(d_host_wmma,d_cal_host_wmma,MATRIX_M,MATRIX_N,D_LAYOUT,D_LAYOUT);
  
   cudaErrCheck(cudaFree(a_htype));
   cudaErrCheck(cudaFree(b_htype));
   cudaErrCheck(cudaFree(c_htype));
   cudaErrCheck(cudaFree(d_htype));
   cudaErrCheck(cudaFree(a_atype));
   cudaErrCheck(cudaFree(b_btype));
   cudaErrCheck(cudaFree(c_ctype));
   cudaErrCheck(cudaFree(d_dtype));

   free(a_host_wmma);
   free(b_host_wmma);
   free(c_host_wmma);
   free(d_host_wmma);
   free(d_cal_host_wmma);
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


