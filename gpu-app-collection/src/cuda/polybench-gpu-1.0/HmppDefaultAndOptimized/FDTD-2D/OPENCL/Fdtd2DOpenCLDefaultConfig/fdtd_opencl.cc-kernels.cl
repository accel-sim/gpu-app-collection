#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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
// kernel : hmpp_codelet__runFdtd_loop0_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runFdtd_loop0_(  __private int t_11, __global float* ey, __global float* fict)
{
  int j_1;
  j_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j_1 <= (int) 2047));
  if(!__hmppcg_guard) 
  {
  ;
  ey[((int) 0 * (int) 2048) + j_1] = fict[t_11];
  }
} 

// kernel : hmpp_codelet__runFdtd_loop1_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__runFdtd_loop1_(  __global float* ey, __global float* hz)
{
  int j_2;
  int i_1;
  j_2 = (get_global_id(0));
  i_1 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_2 <= (int) 2047) & (i_1 <= (int) 2046)));
  if(!__hmppcg_guard) 
  {
  ;
  ey[((i_1 + (int) 1) * (int) 2048) + j_2] = (ey[((i_1 + (int) 1) * (int) 2048) + j_2]) - ((double) 5.000000000000000E-01 * ((hz[((i_1 + (int) 1) * (int) 2048) + j_2]) - (hz[(i_1 * (int) 2048) + j_2])));
  }
} 

// kernel : hmpp_codelet__runFdtd_loop2_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__runFdtd_loop2_(  __global float* ex, __global float* hz)
{
  int j_3;
  int i_2;
  j_3 = (get_global_id(0));
  i_2 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_3 <= (int) 2046) & (i_2 <= (int) 2047)));
  if(!__hmppcg_guard) 
  {
  ;
  ex[(i_2 * (int) 2049) + (j_3 + (int) 1)] = (ex[(i_2 * (int) 2049) + (j_3 + (int) 1)]) - ((double) 5.000000000000000E-01 * ((hz[(i_2 * (int) 2048) + (j_3 + (int) 1)]) - (hz[(i_2 * (int) 2048) + j_3])));
  }
} 

// kernel : hmpp_codelet__runFdtd_loop3_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__runFdtd_loop3_(  __global float* ex, __global float* ey, __global float* hz)
{
  int j_4;
  int i_3;
  j_4 = (get_global_id(0));
  i_3 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_4 <= (int) 2047) & (i_3 <= (int) 2047)));
  if(!__hmppcg_guard) 
  {
  ;
  hz[(i_3 * (int) 2048) + j_4] = (hz[(i_3 * (int) 2048) + j_4]) - ((double) 7.000000000000000E-01 * ((((ex[(i_3 * (int) 2049) + (j_4 + (int) 1)]) - (ex[(i_3 * (int) 2049) + j_4])) + (ey[((i_3 + (int) 1) * (int) 2048) + j_4])) - (ey[(i_3 * (int) 2048) + j_4])));
  }
} 

