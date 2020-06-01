
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
// kernel : hmpp_codelet__runBicg_loop0_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runBicg_loop0_(  __global float* s)
{
  int i_1;
  i_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_1 <= (int) 4095));
  if(!__hmppcg_guard) 
  {
  ;
  s[i_1] = (int) 0;
  }
} 

// kernel : hmpp_codelet__runBicg_loop1_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runBicg_loop1_(  __global float* a, __global float* r, __global float* s)
{
  int j_1;
  j_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j_1 <= (int) 4095));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, outer_i_2;
    for (outer_i_2 = (int) 0, __hmppcg_end = (int) 1023; outer_i_2 <= __hmppcg_end; outer_i_2 += (int) 1)
    {
      {
        int __hmppcg_end, i_4;
        for (i_4 = (int) 0, __hmppcg_end = (int) 3; i_4 <= __hmppcg_end; i_4 += (int) 1)
        {
          s[j_1] = (s[j_1]) + ((r[i_4 + ((int) (outer_i_2 * (int) 4))]) * (a[((i_4 + ((int) (outer_i_2 * (int) 4))) * (int) 4096) + j_1]));
        } 
      }
    } 
  }
  }
} 

// kernel : hmpp_codelet__runBicg_loop2_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runBicg_loop2_(  __global float* a, __global float* p, __global float* q)
{
  int i_3;
  i_3 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_3 <= (int) 4095));
  if(!__hmppcg_guard) 
  {
  ;
  q[i_3] = (int) 0;
  {
    int __hmppcg_end, j_2;
    for (j_2 = (int) 0, __hmppcg_end = (int) 4095; j_2 <= __hmppcg_end; j_2 += (int) 1)
    {
      q[i_3] = (q[i_3]) + ((a[(i_3 * (int) 4096) + j_2]) * (p[j_2]));
    } 
  }
  }
} 

