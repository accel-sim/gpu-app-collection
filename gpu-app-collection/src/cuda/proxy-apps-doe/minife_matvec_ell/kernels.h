#include "ld_functions.h"
#include "shfl.h"
template <int warpSize, typename T>
__device__ __inline__ T warpReduceSum(T val) {
  if(warpSize>16) val+=__shfl_down(val,16,warpSize);
  if(warpSize>8) val+=__shfl_down(val,8,warpSize);
  if(warpSize>4) val+=__shfl_down(val,4,warpSize);
  if(warpSize>2) val+=__shfl_down(val,2,warpSize);
  if(warpSize>1) val+=__shfl_down(val,1,warpSize);
  return val;
}
/********************
 * CSR KERNEL
 * ******************/
template<typename MatrixType,
         typename VectorType>
__global__ void matvec_csr_kernel(MatrixType A, VectorType X, VectorType Y){
  typedef typename MatrixType::ScalarType ScalarType;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;

  //1 warp per row...
  for(int row_idx=blockIdx.x*blockDim.x+threadIdx.x;row_idx<A.num_rows;row_idx+=blockDim.x*gridDim.x)
  {
    ScalarType sum=0;
    int col_start=A.d_row_offsets[row_idx];
    int col_end=A.d_row_offsets[row_idx+1];
    for(int j=col_start;j<col_end;j++) {
        GlobalOrdinalType c=A.d_cols[j];
        ScalarType x=A.d_coefs[j];
        ScalarType a=X.d_coefs[c];
        sum+=a*x;
    }
    Y.d_coefs[row_idx]=sum;
  }
}

template<typename MatrixType, typename VectorType>
void matvec_csr(MatrixType A, VectorType x, VectorType y){
  int threads=256;
  int blocks=min((A.num_rows+threads-1)/threads,4096);

  matvec_csr_kernel<<<blocks,threads>>>(A, x, y);
  cudaCheckError();
}

/********************
 * CSR VECTOR KERNEL
 * ******************/
template<int warpSize, typename MatrixType,
         typename VectorType>
__global__ void matvec_csr_vector_kernel(MatrixType A, VectorType X, VectorType Y){
  typedef typename MatrixType::ScalarType ScalarType;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;

  //1 warp per row...
  for(int row_idx=blockIdx.y*blockDim.y+threadIdx.y;row_idx<A.num_rows;row_idx+=blockDim.y*gridDim.y)
  {
    ScalarType sum=0;
    int col_start=A.d_row_offsets[row_idx];
    int col_end=A.d_row_offsets[row_idx+1];
  
    for(int j=col_start+threadIdx.x;j<col_end;j+=warpSize)
    {
      if(j<col_end) {
        GlobalOrdinalType c=A.d_cols[j];
        ScalarType x=A.d_coefs[j];
        ScalarType a=X.d_coefs[c];
        sum+=a*x;
      }
    }
    sum=warpReduceSum<warpSize>(sum);
    
    if(threadIdx.x==0) {
      Y.d_coefs[row_idx]=sum;
    }
  }
}

/********************
 * CSR VECTOR KERNEL
 * ******************/
template<int warpSize, typename MatrixType, typename VectorType>
void matvec_csr_vector(MatrixType A, VectorType x, VectorType y){
  dim3 BLOCK_SIZE;
  BLOCK_SIZE.x=warpSize;
  BLOCK_SIZE.y=256/warpSize;

  dim3 BLOCKS;

  BLOCKS.x=1;
  BLOCKS.y=min((A.num_rows+BLOCK_SIZE.y-1)/BLOCK_SIZE.y,4096);

  matvec_csr_vector_kernel<warpSize><<<BLOCKS,BLOCK_SIZE>>>(A, x, y);
  cudaCheckError();

}

/********************
 * ELL KERNEL
 * ******************/
template<typename MatrixType, typename VectorType>
__global__ void matvec_ell_kernel(MatrixType A, VectorType X, VectorType Y) {
  typedef typename MatrixType::ScalarType ScalarType;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;

  for(int row_idx=blockIdx.x*blockDim.x+threadIdx.x;row_idx<A.num_rows;row_idx+=blockDim.x*gridDim.x)
  {
    ScalarType sum=0;
    GlobalOrdinalType offset = row_idx;
    for(int j=0;j<A.num_nnz_per_row;++j)
    {
      GlobalOrdinalType c=A.d_cols[offset];
      if(c!=-1) {
        ScalarType a=A.d_coefs[offset];
        ScalarType x=X.d_coefs[c];
        sum+=a*x;
      }
      offset+=A.pitch;
    }
    Y.d_coefs[row_idx]=sum;
  }
}

template<typename MatrixType, typename VectorType>
void matvec_ell(MatrixType A, VectorType x, VectorType y){
  int threads=256;
  int blocks=min((A.num_rows+threads-1)/threads,4096);

  matvec_ell_kernel<<<blocks,threads>>>(A, x, y);
  cudaCheckError();

}

/********************
 * ELL CV KERNEL
 * ******************/
template<typename MatrixType, typename VectorType>
__global__ void matvec_ell_cv_kernel(MatrixType A, VectorType X, VectorType Y) {
  typedef typename MatrixType::ScalarType ScalarType;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;

  for(int row_idx=blockIdx.x*blockDim.x+threadIdx.x;row_idx<A.num_rows;row_idx+=blockDim.x*gridDim.x)
  {
    ScalarType sum=0;
    GlobalOrdinalType offset = row_idx;
    for(int j=0;j<A.num_nnz_per_row;++j)
    {
      GlobalOrdinalType c=ld_cv(A.d_cols+offset);
      if(c!=-1) {
        ScalarType a=ld_cv(A.d_coefs+offset);
        ScalarType x=X.d_coefs[c];
        sum+=a*x;
      }
      offset+=A.pitch;
    }
    Y.d_coefs[row_idx]=sum;
  }
}

template<typename MatrixType, typename VectorType>
void matvec_ell_cv(MatrixType A, VectorType x, VectorType y){
  int threads=256;
  int blocks=min((A.num_rows+threads-1)/threads,4096);

  matvec_ell_cv_kernel<<<blocks,threads>>>(A, x, y);
  cudaCheckError();

}



