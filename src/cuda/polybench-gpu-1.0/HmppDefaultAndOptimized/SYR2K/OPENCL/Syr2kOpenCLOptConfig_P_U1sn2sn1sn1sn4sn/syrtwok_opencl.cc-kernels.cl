
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
// kernel : hmpp_codelet__runSyrTwoK_loop0_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__runSyrTwoK_loop0_(  __global float* c)
{
  float tmp_2;
  int j_1;
  int i_1;
  j_1 = (get_global_id(0));
  i_1 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_1 <= (int) 1023) & (i_1 <= (int) 2047)));
  if(!__hmppcg_guard) 
  {
  ;
  tmp_2 = (float) ((int) 4546);
  c[(i_1 * (int) 2048) + j_1] = (c[(i_1 * (int) 2048) + j_1]) * tmp_2;
  c[(i_1 * (int) 2048) + (j_1 + (int) 1024)] = (c[(i_1 * (int) 2048) + (j_1 + (int) 1024)]) * tmp_2;
  }
} 

// kernel : hmpp_codelet__runSyrTwoK_loop1_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__runSyrTwoK_loop1_(  __global float* a, __global float* b, __global float* c)
{
  int j_2;
  int i_2;
  j_2 = (get_global_id(0));
  i_2 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_2 <= (int) 2047) & (i_2 <= (int) 2047)));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, k_1;
    for (k_1 = (int) 0, __hmppcg_end = (int) 511; k_1 <= __hmppcg_end; k_1 += (int) 1)
    {
      float tmp_1__0;
      float tmp_3;
      tmp_3 = (float) ((int) 12435);
      tmp_1__0 = tmp_3;
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (a[(i_2 * (int) 2048) + k_1])) * (b[(j_2 * (int) 2048) + k_1]));
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (b[(i_2 * (int) 2048) + k_1])) * (a[(j_2 * (int) 2048) + k_1]));
      float tmp_1__1;
      tmp_1__1 = tmp_3;
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (a[(i_2 * (int) 2048) + (k_1 + (int) 512)])) * (b[(j_2 * (int) 2048) + (k_1 + (int) 512)]));
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (b[(i_2 * (int) 2048) + (k_1 + (int) 512)])) * (a[(j_2 * (int) 2048) + (k_1 + (int) 512)]));
      float tmp_1__2;
      tmp_1__2 = tmp_3;
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (a[(i_2 * (int) 2048) + (k_1 + (int) 1024)])) * (b[(j_2 * (int) 2048) + (k_1 + (int) 1024)]));
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (b[(i_2 * (int) 2048) + (k_1 + (int) 1024)])) * (a[(j_2 * (int) 2048) + (k_1 + (int) 1024)]));
      float tmp_1__3;
      tmp_1__3 = tmp_3;
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (a[(i_2 * (int) 2048) + (k_1 + (int) 1536)])) * (b[(j_2 * (int) 2048) + (k_1 + (int) 1536)]));
      c[(i_2 * (int) 2048) + j_2] = (c[(i_2 * (int) 2048) + j_2]) + ((tmp_3 * (b[(i_2 * (int) 2048) + (k_1 + (int) 1536)])) * (a[(j_2 * (int) 2048) + (k_1 + (int) 1536)]));
    } 
  }
  }
} 

