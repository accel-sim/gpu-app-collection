
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
// kernel : hmpp_codelet__threeMMloopa_loop0_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__threeMMloopa_loop0_(  __global float* a, __global float* b, __global float* e)
{
  int j_3;
  int i_3;
  j_3 = (get_global_id(0));
  i_3 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_3 <= (int) 511) & (i_3 <= (int) 511)));
  if(!__hmppcg_guard) 
  {
  ;
  e[(i_3 * (int) 512) + j_3] = (int) 0;
  {
    int __hmppcg_end, k_3;
    for (k_3 = (int) 0, __hmppcg_end = (int) 511; k_3 <= __hmppcg_end; k_3 += (int) 1)
    {
      e[(i_3 * (int) 512) + j_3] = (e[(i_3 * (int) 512) + j_3]) + ((a[(i_3 * (int) 512) + k_3]) * (b[(k_3 * (int) 512) + j_3]));
    } 
  }
  }
} 

// kernel : hmpp_codelet__threeMMloopb_loop0_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__threeMMloopb_loop0_(  __global float* c, __global float* d, __global float* f)
{
  int j_4;
  int i_4;
  j_4 = (get_global_id(0));
  i_4 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_4 <= (int) 511) & (i_4 <= (int) 511)));
  if(!__hmppcg_guard) 
  {
  ;
  f[(i_4 * (int) 512) + j_4] = (int) 0;
  {
    int __hmppcg_end, k_4;
    for (k_4 = (int) 0, __hmppcg_end = (int) 511; k_4 <= __hmppcg_end; k_4 += (int) 1)
    {
      f[(i_4 * (int) 512) + j_4] = (f[(i_4 * (int) 512) + j_4]) + ((c[(i_4 * (int) 512) + k_4]) * (d[(k_4 * (int) 512) + j_4]));
    } 
  }
  }
} 

// kernel : hmpp_codelet__threeMMloopc_loop0_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__threeMMloopc_loop0_(  __global float* e_11, __global float* f_11, __global float* g)
{
  int j_5;
  int i_5;
  j_5 = (get_global_id(0));
  i_5 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_5 <= (int) 511) & (i_5 <= (int) 511)));
  if(!__hmppcg_guard) 
  {
  ;
  g[(i_5 * (int) 512) + j_5] = (int) 0;
  {
    int __hmppcg_end, k_5;
    for (k_5 = (int) 0, __hmppcg_end = (int) 511; k_5 <= __hmppcg_end; k_5 += (int) 1)
    {
      g[(i_5 * (int) 512) + j_5] = (g[(i_5 * (int) 512) + j_5]) + ((e_11[(i_5 * (int) 512) + k_5]) * (f_11[(k_5 * (int) 512) + j_5]));
    } 
  }
  }
} 

