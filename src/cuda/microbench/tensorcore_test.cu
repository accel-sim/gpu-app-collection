#include<iostream>
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
#define ROW_MAJOR 0
#define COL_MAJOR 1

//ONLY THE PARAMETER HERE NEEDS TO BE CHANGED
// Must be multiples of 16 for wmma code to work
#define MATRIX_M (16)
#define MATRIX_N (16)
#define MATRIX_K (16)
const int WMMA_M =16;
const int WMMA_N =16;
const int WMMA_K =16;

//Don't change anything after here 
#define NUM_CTA 1
#define WARP_IN_CTA 1
#define THREAD_IN_WARP 32
#define RANDOM_RANGE 10

void get_stride(int &stride_a,int &stride_b, int &stride_c,int &stride_d,int layout_a,int layout_b,int layout_c,int layout_d){
	if(layout_a==ROW_MAJOR)
		stride_a=MATRIX_K;
	else
		stride_a=MATRIX_M;

	if(layout_b==ROW_MAJOR)
		stride_b=MATRIX_N;
	else
		stride_b=MATRIX_K;

	if(layout_c==ROW_MAJOR)
		stride_c=MATRIX_N;
	else
		stride_c=MATRIX_M;

	if(layout_d==ROW_MAJOR)
		stride_d=MATRIX_N;
	else
		stride_d=MATRIX_M;
	return;
}
void get_stride_in_matmul(int &stride_a,int &stride_b, int &stride_c,int layout_a,int layout_b,int layout_c){
	if(layout_a==ROW_MAJOR)
		stride_a=MATRIX_K;
	else
		stride_a=MATRIX_M;

	if(layout_b==ROW_MAJOR)
		stride_b=MATRIX_N;
	else
		stride_b=MATRIX_K;

	if(layout_c==ROW_MAJOR)
		stride_c=MATRIX_N;
	else
		stride_c=MATRIX_M;
	return;
}

enum MatrixInitializationType{
	ZERO,
	ONE,
	RANDOM,
	IDENTITY
};

int get_value(MatrixInitializationType init_type,int randomRange=RANDOM_RANGE,bool RESET=false){
	static int val=0;
	switch(init_type){
		case ZERO:
			val=0;
			break;
		case ONE:
			val=1;
			break;
		case RANDOM:
			val=rand()%randomRange;
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
	#ifdef DEBUG
	for(int row=0;row<row_size;row++){
		for(int col=0;col<col_size;col++){
			T val;
			if(layout==ROW_MAJOR)
				val=matrix[row*col_size+col];		
			else
				val=matrix[col*row_size+row];
			printf("%.1f ",static_cast<float>(val));
		}
		printf(";\n");
	}
	#endif
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
	get_value(init_type,RANDOM_RANGE,true);//reseting the val counter
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
int compare_matrix(T *matrix_a, T *matrix_b,int row_size,int col_size,int/*MatrixLayout*/  alayout,int/*MatrixLayout*/  blayout){
	int equal=1;	
	for(int row=0;row<row_size;row++){
		for(int col=0;col<col_size;col++){
			int index_a,index_b;
			index_a=get_index(row,col,row_size,col_size,alayout);
			index_b=get_index(row,col,row_size,col_size,alayout);
			double err= abs(static_cast<double>(matrix_a[index_a])-static_cast<double>(matrix_b[index_b]));
			//assuming matrix_b is the true vale
			double max_error_tolerance=(5.0*static_cast<double>(matrix_b[index_b]))/100.0;
			printf("err=%f max_error_tolerance=%f\n",err,max_error_tolerance);
			if(err>max_error_tolerance){
				printf("ERROR at index row=%d col=%d %f %f\n",row,col,matrix_a[index_a],matrix_b[index_b]);
			 	equal=0;	
			}
		}
				
	}
	return equal;
}
template <typename aT,typename bT,typename cT,typename dT,typename layout_a,typename layout_b>
__global__ void wmma_example(aT *a, bT *b, cT *c,dT *d,wmma::layout_t layout_c,wmma::layout_t layout_d,int stride_a,int stride_b,int stride_c,int stride_d)
{
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, aT, layout_a> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bT , layout_b> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, cT> c_frag;
   // Bounds checking
   wmma::load_matrix_sync(a_frag, a, stride_a);
   wmma::load_matrix_sync(b_frag, b, stride_b);
   wmma::load_matrix_sync(c_frag, c, stride_c,layout_c);
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   wmma::store_matrix_sync(d, c_frag, stride_d, layout_d);
}

template <typename T1,typename T2>	
__global__ void convert(T1 *out, T2 *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

template <typename atype,typename btype,typename ctype,typename alayout_config,typename blayout_config>	
__global__ void generic_matmul(atype *a, btype *b, ctype *c,int layout_a,int layout_b,int layout_c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   wmma::layout_t clayout_config=(layout_c==ROW_MAJOR)?(wmma::mem_row_major):(wmma::mem_col_major);

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, alayout_config> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, blayout_config> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, ctype> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, ctype> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
	 int a_offset = (layout_a==COL_MAJOR) ? (aRow + aCol*lda) : (aRow*lda + aCol);
	 int b_offset = (layout_b==COL_MAJOR) ? (bRow + bCol*ldb) : (bRow*ldb + bCol);
         wmma::load_matrix_sync(a_frag, a + a_offset, lda);
         wmma::load_matrix_sync(b_frag, b + b_offset, ldb);
 
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;
   if (cRow < M && cCol < N) {
      int c_offset = (layout_c==COL_MAJOR) ? (cRow + cCol*ldc) : (cRow*ldc + cCol);
      wmma::load_matrix_sync(c_frag, c + c_offset, ldc, clayout_config);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + c_offset, c_frag, ldc, clayout_config);
   }
}
template <typename atype,typename btype,typename ctype,typename dtype,typename host_type,typename layout_a_config,typename layout_b_config>	
void check_configuration(int layout_a,int layout_b,int layout_c,int layout_d){
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

  	int stride_a,stride_b,stride_c,stride_d; 
  	std::string row_str="ROW";
	std::string col_str="COL";
	wmma::layout_t layout_c_config;  
	wmma::layout_t layout_d_config;  
	
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

	std::string tensorcore_mode;
	assert(sizeof(ctype)==sizeof(dtype));
	if(sizeof(ctype)==sizeof(half))
		tensorcore_mode="PURE FP16";
	else if (sizeof(ctype)==sizeof(float))
		tensorcore_mode="MIXED";
	else{
		printf("WRONG MODE\n");
		abort();
	}
		
  	printf("a_host\n");
  	initialize_matrix<host_type>(a_host_wmma,MATRIX_M,MATRIX_K,layout_a,RANDOM);
  	printf("b_host\n");
  	initialize_matrix<host_type>(b_host_wmma,MATRIX_K,MATRIX_N,layout_b,RANDOM);
  	printf("c_host\n");
  	initialize_matrix<host_type>(c_host_wmma,MATRIX_M,MATRIX_N,layout_c,RANDOM);
  	printf("d_cal_host\n");
  	initialize_matrix<host_type>(d_cal_host_wmma,MATRIX_M,MATRIX_N,layout_d,ZERO);

  	cudaErrCheck(cudaMemcpy(a_htype,a_host_wmma,  MATRIX_M * MATRIX_K * sizeof(host_type), cudaMemcpyHostToDevice));
  	cudaErrCheck(cudaMemcpy(b_htype,b_host_wmma,  MATRIX_K * MATRIX_N * sizeof(host_type), cudaMemcpyHostToDevice));
  	cudaErrCheck(cudaMemcpy(c_htype,c_host_wmma,  MATRIX_M * MATRIX_N * sizeof(host_type), cudaMemcpyHostToDevice));

  	convert<atype,host_type> <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_atype, a_htype, MATRIX_M * MATRIX_K);
  	convert<btype,host_type> <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_btype, b_htype, MATRIX_K * MATRIX_N);
  	convert<ctype,host_type> <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_ctype, c_htype, MATRIX_M * MATRIX_N);

	std::string str_layout_a,str_layout_b,str_layout_c,str_layout_d;
	str_layout_a	=(layout_a==ROW_MAJOR)?row_str:col_str;
	str_layout_b	=(layout_b==ROW_MAJOR)?row_str:col_str;
	str_layout_c	=(layout_c==ROW_MAJOR)?row_str:col_str;
	str_layout_d	=(layout_d==ROW_MAJOR)?row_str:col_str;
	
  	printf("\nM = %d, N = %d, K = %d. \n", MATRIX_M, MATRIX_N, MATRIX_K);
  	printf("MATRIX_A    MATRIX_B    MATRIX_C    MATRIX_D\n");
  	printf("  %s          %s          %s          %s   CONFIGURATION\n",str_layout_a.c_str(),str_layout_b.c_str(),str_layout_c.c_str(),str_layout_d.c_str());

  	get_stride(stride_a,stride_b,stride_c,stride_d,layout_a,layout_b,layout_c,layout_d); 
	layout_c_config=(layout_c==ROW_MAJOR)?wmma::mem_row_major:wmma::mem_col_major;
	layout_d_config=(layout_d==ROW_MAJOR)?wmma::mem_row_major:wmma::mem_col_major;

  	wmma_example<atype,btype,ctype,dtype,layout_a_config,layout_b_config> <<< NUM_CTA,WARP_IN_CTA*THREAD_IN_WARP>>> (a_atype, b_btype, c_ctype, d_dtype,layout_c_config,layout_d_config,stride_a,stride_b,stride_c,stride_d);
  	convert<host_type,dtype> <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (d_htype, d_dtype, MATRIX_M * MATRIX_N);

  	printf("\nChecking results...\n");
  	cudaErrCheck(cudaMemcpy(d_host_wmma, d_htype, MATRIX_M * MATRIX_N * sizeof(host_type), cudaMemcpyDeviceToHost));

  	printf("D_CAL_HOST\n");
  	matrix_multiply<host_type>(d_cal_host_wmma,a_host_wmma,b_host_wmma,c_host_wmma,MATRIX_M,MATRIX_N,MATRIX_K,layout_d,layout_a,layout_b,layout_c);
  	printf("D_WMMA\n");
  	print_matrix<host_type>(d_host_wmma,MATRIX_M,MATRIX_N,layout_d);
  	printf("CHECKING\n");
  	if(compare_matrix<host_type>(d_host_wmma,d_cal_host_wmma,MATRIX_M,MATRIX_N,layout_d,layout_d)){
  	     printf("%s MODE CONFIGURATION %s_%s_%s_%s COMPLETED SUCCESSFULLY\n",tensorcore_mode.c_str(),str_layout_a.c_str(),str_layout_b.c_str(),str_layout_c.c_str(),str_layout_d.c_str());
  	}
  	else{
  	     printf("ERROR\n");
  	     abort();
  	}

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
}
//FUNCTION HAS BEEN TESTED ON 2 CONFIGURATION ONLY: <ROW ROW ROW> AND  <COL COL COL> 
//AND TESTED ONLY FOR SQUARE MATRIX OF SIZE 16, 32, 64, 128, 256, 512
template <typename atype,typename btype,typename ctype,typename host_type,typename layout_a_config,typename layout_b_config>	
void check_generic_matmul(int matrix_m,int matrix_n,int matrix_k,int layout_a,int layout_b,int layout_c){
  	//data on device in host type format
  	host_type *a_htype;
  	host_type *b_htype;
  	host_type *c_htype;

  	//data on device in gemm format
  	atype *a_atype;
  	btype *b_btype;
  	ctype *c_ctype;

  	srand(time(NULL));  
 
  	host_type *a_host_wmma;
  	host_type *b_host_wmma;
  	host_type *c_host_wmma;
  	host_type *c_cal_host_wmma;
   	dim3 gridDim;
   	dim3 blockDim;

  	std::string row_str="ROW";
	std::string col_str="COL";
	//wmma::layout_t layout_c_config;  
  	
  	
  	// Use tensor cores
  	cudaErrCheck(cudaMalloc((void**)&a_htype, matrix_m * matrix_k * sizeof(host_type)));
  	cudaErrCheck(cudaMalloc((void**)&b_htype, matrix_k * matrix_n * sizeof(host_type)));
  	cudaErrCheck(cudaMalloc((void**)&c_htype, matrix_m * matrix_n * sizeof(host_type)));
  	cudaErrCheck(cudaMalloc((void**)&a_atype, matrix_m * matrix_k * sizeof(atype)));
  	cudaErrCheck(cudaMalloc((void**)&b_btype, matrix_k * matrix_n * sizeof(btype)));
  	cudaErrCheck(cudaMalloc((void**)&c_ctype, matrix_m * matrix_n * sizeof(ctype)));


  	a_host_wmma      = (host_type*)malloc(matrix_m * matrix_k * sizeof(host_type));
  	b_host_wmma      = (host_type*)malloc(matrix_k * matrix_n * sizeof(host_type));
  	c_host_wmma      = (host_type*)malloc(matrix_m * matrix_n * sizeof(host_type));
  	c_cal_host_wmma  = (host_type*)malloc(matrix_m * matrix_n * sizeof(host_type));

	std::string tensorcore_mode;
	if(sizeof(ctype)==sizeof(half))
		tensorcore_mode="PURE FP16";
	else if (sizeof(ctype)==sizeof(float))
		tensorcore_mode="MIXED";
	else{
		printf("WRONG MODE\n");
		abort();
	}

  	printf("a_host\n");
  	initialize_matrix<host_type>(a_host_wmma,matrix_m,matrix_k,layout_a,RANDOM);
  	printf("b_host\n");
  	initialize_matrix<host_type>(b_host_wmma,matrix_k,matrix_n,layout_b,RANDOM);
  	printf("c_host\n");
  	initialize_matrix<host_type>(c_host_wmma,matrix_m,matrix_n,layout_c,RANDOM);
  	printf("c_cal_host\n");
  	initialize_matrix<host_type>(c_cal_host_wmma,matrix_m,matrix_n,layout_c,ZERO);
  	printf("D_CAL_HOST\n");
  	matrix_multiply<host_type>(c_cal_host_wmma,a_host_wmma,b_host_wmma,c_host_wmma,matrix_m,matrix_n,matrix_k,layout_c,layout_a,layout_b,layout_c);

  	cudaErrCheck(cudaMemcpy(a_htype,a_host_wmma,  matrix_m * matrix_k * sizeof(host_type), cudaMemcpyHostToDevice));
  	cudaErrCheck(cudaMemcpy(b_htype,b_host_wmma,  matrix_k * matrix_n * sizeof(host_type), cudaMemcpyHostToDevice));
  	cudaErrCheck(cudaMemcpy(c_htype,c_host_wmma,  matrix_m * matrix_n * sizeof(host_type), cudaMemcpyHostToDevice));

  	convert<atype,host_type> <<< (matrix_m * matrix_k + 255) / 256, 256 >>> (a_atype, a_htype, matrix_m * matrix_k);
  	convert<btype,host_type> <<< (matrix_k * matrix_n + 255) / 256, 256 >>> (b_btype, b_htype, matrix_k * matrix_n);
  	convert<ctype,host_type> <<< (matrix_m * matrix_n + 255) / 256, 256 >>> (c_ctype, c_htype, matrix_m * matrix_n);

	std::string str_layout_a,str_layout_b,str_layout_c;
	str_layout_a	=(layout_a==ROW_MAJOR)?row_str:col_str;
	str_layout_b	=(layout_b==ROW_MAJOR)?row_str:col_str;
	str_layout_c	=(layout_c==ROW_MAJOR)?row_str:col_str;

  	printf("\nM = %d, N = %d, K = %d. \n", matrix_m,matrix_n, matrix_k);
  	printf("MATRIX_A    MATRIX_B    MATRIX_C    \n");
  	printf("  %s          %s          %s        CONFIGURATION\n",str_layout_a.c_str(),str_layout_b.c_str(),str_layout_c.c_str());

  	//get_stride_in_matmul(stride_a,stride_b,stride_c,layout_a,layout_b,layout_c); 
	//layout_c_config=(layout_c==ROW_MAJOR)?wmma::mem_row_major:wmma::mem_col_major;
  
	float alpha=1.0f;
	float beta =1.0f;
   	
	// blockDim.x must be a multple of warpSize
   	// 128x4 means we have 16 warps and a block computes a 64x64 output tile
   	blockDim.x = 64;
   	blockDim.y = 2;

   	gridDim.x = (matrix_m+ (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   	gridDim.y = (matrix_n+ WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   	printf("GRID:X=%d,Y=%d\n",gridDim.x,gridDim.y);
   	printf("BLOCK:X=%d,Y=%d\n",blockDim.x,blockDim.y);

	generic_matmul<atype,btype,ctype,layout_a_config,layout_b_config> <<< gridDim,blockDim >>> (a_atype, b_btype, c_ctype,layout_a,layout_b,layout_c, matrix_m , matrix_n, matrix_k, alpha, beta);
  	convert<host_type,ctype> <<< (matrix_m * matrix_n + 255) / 256, 256 >>> (c_htype, c_ctype, matrix_m * matrix_n);

  	printf("\nChecking results...\n");
  	cudaErrCheck(cudaMemcpy(c_host_wmma, c_htype, matrix_m* matrix_n* sizeof(host_type), cudaMemcpyDeviceToHost));

  	printf("D_WMMA\n");
  	print_matrix<host_type>(c_host_wmma,matrix_m,matrix_n,layout_c);
  	printf("CHECKING\n");
  	if(compare_matrix<host_type>(c_host_wmma,c_cal_host_wmma,matrix_m,matrix_n,layout_c,layout_c)){
  	     printf("%s MODE M=%d N=%d K=%d CONFIGURATION %s_%s_%s COMPLETED SUCCESSFULLY\n",tensorcore_mode.c_str(),matrix_m,matrix_n,matrix_k,str_layout_a.c_str(),str_layout_b.c_str(),str_layout_c.c_str());
  	}
  	else{
  	     printf("ERROR\n");
  	     abort();
  	}

  	cudaErrCheck(cudaFree(a_htype));
  	cudaErrCheck(cudaFree(b_htype));
  	cudaErrCheck(cudaFree(c_htype));
  	cudaErrCheck(cudaFree(a_atype));
  	cudaErrCheck(cudaFree(b_btype));
  	cudaErrCheck(cudaFree(c_ctype));

  	free(a_host_wmma);
  	free(b_host_wmma);
  	free(c_host_wmma);
  	free(c_cal_host_wmma);
  	cudaErrCheck(cudaDeviceReset());

}
int main(int argc, char* argv[]) {
	//////MIXED_PRECISION MODE
	////check_configuration<atype,btype,ctype,dtype,host_type,layout_a_config,layout_b_config>(layout_a,layout_b,layout_c,layout_d)
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,COL_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,COL_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,float,float,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,COL_MAJOR,COL_MAJOR);

	//check_configuration<half,half,half,half,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	//PURE FP16 MODE
	//check_configuration<atype,btype,ctype,dtype,host_type,layout_a_config,layout_b_config>(layout_a,layout_b,layout_c,layout_d)
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::row_major>(ROW_MAJOR,ROW_MAJOR,COL_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::row_major,wmma::col_major>(ROW_MAJOR,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::row_major>(COL_MAJOR,ROW_MAJOR,COL_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,ROW_MAJOR,COL_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,COL_MAJOR,ROW_MAJOR);
	check_configuration<half,half,half,half,float,wmma::col_major,wmma::col_major>(COL_MAJOR,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	
	//check_generic_matmul<half,half,float,float,wmma::col_major,wmma::col_major>(16,16,16,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::col_major,wmma::col_major>(32,32,32,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::col_major,wmma::col_major>(64,64,64,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::col_major,wmma::col_major>(128,128,128,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::col_major,wmma::col_major>(256,256,256,COL_MAJOR,COL_MAJOR,COL_MAJOR);
	check_generic_matmul<half,half,float,float,wmma::col_major,wmma::col_major>(512,512,512,COL_MAJOR,COL_MAJOR,COL_MAJOR);

	//check_generic_matmul<half,half,float,float,wmma::row_major,wmma::row_major>(16,16,16,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::row_major,wmma::row_major>(32,32,32,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::row_major,wmma::row_major>(64,64,64,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::row_major,wmma::row_major>(128,128,128,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	//check_generic_matmul<half,half,float,float,wmma::row_major,wmma::row_major>(256,256,256,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	check_generic_matmul<half,half,float,float,wmma::row_major,wmma::row_major>(512,512,512,ROW_MAJOR,ROW_MAJOR,ROW_MAJOR);
	return 0;
}
