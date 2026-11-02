




#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}
  
 
//TEXTURE memory
/* tex_x_float is unused in this variant; the texture version lives in
   src/cuda_tex. Texture references were removed in CUDA 12. */

//constant memory
__constant__ int jds_ptr_int[5000];
__constant__ int sh_zcnt_int[5000];

__global__ void spmv_jds(float *dst_vector,
							   const float *d_data,const int *d_index, const int *d_perm,
							   const float *x_vec,const int *d_nzcnt,const int dem);
							   
