
#ifdef GLOBAL_ATOMIC_EXTS_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#endif

#ifdef LOCAL_ATOMIC_EXTS_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#endif

#ifdef BYTE_ADDRESSABLE_STORE_EXTS_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#endif

#ifndef HMPPCG_WARP_SIZE
#define HMPPCG_WARP_SIZE 1
#endif
#ifndef __HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE
#define __HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE( var ) \
        var ## _aarray_desc->wholesize_
#endif //__HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_SIZE
#define __HMPPCG_ALLOCATABLE_ARRAY_SIZE( var, d ) \
        var ## _aarray_desc->sizes_[d]
#endif //__HMPPCG_ALLOCATABLE_ARRAY_SIZE

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_LBOUND
#define __HMPPCG_ALLOCATABLE_ARRAY_LBOUND( var, d ) \
        var ## _aarray_desc->lbounds_[d]
#endif //__HMPPCG_ALLOCATABLE_ARRAY_LBOUND

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_UBOUND
#define __HMPPCG_ALLOCATABLE_ARRAY_UBOUND( var, d ) \
        (var ## _aarray_desc->sizes_[d] + var ## _aarray_desc->lbounds_[d] - 1)
#endif //__HMPPCG_ALLOCATABLE_ARRAY_UBOUND

#define HMPP_INT_POW_FUNC(func_ext_name, func_type)                    \
  func_type hmpp_pow ##func_ext_name ( func_type base, func_type exp ) \
  {                                                                    \
    if(exp < 0)                                                        \
      return 0;                                                        \
    func_type result = 1;                                              \
    while (exp)                                                        \
    {                                                                  \
      if (exp & 1)                                                     \
        result *= base;                                                \
      exp >>= 1;                                                       \
      base *= base;                                                    \
    }                                                                  \
      return result;                                                   \
  }

HMPP_INT_POW_FUNC( i64, long );
HMPP_INT_POW_FUNC( i32, int );
HMPP_INT_POW_FUNC( i16, short );
HMPP_INT_POW_FUNC( i8,  char );
HMPP_INT_POW_FUNC( ui64, unsigned long );
HMPP_INT_POW_FUNC( ui32, unsigned int );
HMPP_INT_POW_FUNC( ui16, unsigned short );
HMPP_INT_POW_FUNC( ui8,  unsigned char );
// kernel : hmpp_codelet__conv3D_loop0_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__conv3D_loop0_(  __global float* A, __global float* B)
{
  int k_1;
  int j_1;
  k_1 = (get_global_id(0));
  j_1 = (get_global_id(1));
  bool __hmppcg_guard = (!((k_1 <= (int) 253) & (j_1 <= (int) 253)));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, i_1;
    for (i_1 = (int) 0, __hmppcg_end = (int) 253; i_1 <= __hmppcg_end; i_1 += (int) 1)
    {
      float tmp_1;
      float tmp_2;
      float tmp_3;
      float tmp_4;
      float tmp_5;
      float tmp_6;
      tmp_1 = (float) ((int) 7);
      tmp_2 = (float) ((int) 10);
      tmp_3 = (float) ((int) 2);
      tmp_4 = (float) ((int) 5);
      tmp_5 = (float) ((int) -8);
      tmp_6 = (float) ((int) 4);
      B[((((i_1 + (int) 1) * (int) 256) + (j_1 + (int) 1)) * (int) 256) + (k_1 + (int) 1)] = ((((((((((((((tmp_3 * (A[(((i_1 * (int) 256) + j_1) * (int) 256) + k_1])) + (tmp_6 * (A[((((i_1 + (int) 2) * (int) 256) + j_1) * (int) 256) + k_1]))) + (tmp_4 * (A[(((i_1 * (int) 256) + j_1) * (int) 256) + k_1]))) + (tmp_1 * (A[((((i_1 + (int) 2) * (int) 256) + j_1) * (int) 256) + k_1]))) + (tmp_5 * (A[(((i_1 * (int) 256) + j_1) * (int) 256) + k_1]))) + (tmp_2 * (A[((((i_1 + (int) 2) * (int) 256) + j_1) * (int) 256) + k_1]))) + (((float) ((int) -3)) * (A[((((i_1 + (int) 1) * (int) 256) + j_1) * (int) 256) + (k_1 + (int) 1)]))) + (((float) ((int) 6)) * (A[((((i_1 + (int) 1) * (int) 256) + (j_1 + (int) 1)) * (int) 256) + (k_1 + (int) 1)]))) + (((float) ((int) -9)) * (A[((((i_1 + (int) 1) * (int) 256) + (j_1 + (int) 2)) * (int) 256) + (k_1 + (int) 1)]))) + (tmp_3 * (A[(((i_1 * (int) 256) + j_1) * (int) 256) + (k_1 + (int) 2)]))) + (tmp_6 * (A[((((i_1 + (int) 2) * (int) 256) + j_1) * (int) 256) + (k_1 + (int) 2)]))) + (tmp_4 * (A[(((i_1 * (int) 256) + (j_1 + (int) 1)) * (int) 256) + (k_1 + (int) 2)]))) + (tmp_1 * (A[((((i_1 + (int) 2) * (int) 256) + (j_1 + (int) 1)) * (int) 256) + (k_1 + (int) 2)]))) + (tmp_5 * (A[(((i_1 * (int) 256) + (j_1 + (int) 2)) * (int) 256) + (k_1 + (int) 2)]))) + (tmp_2 * (A[((((i_1 + (int) 2) * (int) 256) + (j_1 + (int) 2)) * (int) 256) + (k_1 + (int) 2)]));
    } 
  }
  }
} 

