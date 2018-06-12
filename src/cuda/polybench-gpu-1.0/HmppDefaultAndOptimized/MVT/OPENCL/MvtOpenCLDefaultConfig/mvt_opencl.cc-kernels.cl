
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
// kernel : hmpp_codelet__runMvt_loop0_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runMvt_loop0_(  __global float* a, __global float* x1, __global float* y1)
{
  int i_1;
  i_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_1 <= (int) 4095));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, j_1;
    for (j_1 = (int) 0, __hmppcg_end = (int) 4095; j_1 <= __hmppcg_end; j_1 += (int) 1)
    {
      x1[i_1] = (x1[i_1]) + ((a[(i_1 * (int) 4096) + j_1]) * (y1[j_1]));
    } 
  }
  }
} 

// kernel : hmpp_codelet__runMvt_loop1_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runMvt_loop1_(  __global float* a, __global float* x2, __global float* y2)
{
  int i_2;
  i_2 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_2 <= (int) 4095));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, j_2;
    for (j_2 = (int) 0, __hmppcg_end = (int) 4095; j_2 <= __hmppcg_end; j_2 += (int) 1)
    {
      x2[i_2] = (x2[i_2]) + ((a[(j_2 * (int) 4096) + i_2]) * (y2[j_2]));
    } 
  }
  }
} 

