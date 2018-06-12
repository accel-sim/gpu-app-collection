
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
// kernel : hmpp_codelet__runGesummv_loop0_
__kernel __attribute__((reqd_work_group_size(32, 4, 1))) void hmpp_codelet__runGesummv_loop0_(  __global float* a, __global float* b, __global float* tmp1, __global float* x1, __global float* y1)
{
  int i_2;
  int outer_i_2;
  i_2 = (get_global_id(0));
  outer_i_2 = (get_global_id(1));
  bool __hmppcg_guard = (!((i_2 <= (int) 3) & (outer_i_2 <= (int) 1023)));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, outer_j_2;
    for (outer_j_2 = (int) 0, __hmppcg_end = (int) 1365; outer_j_2 <= __hmppcg_end; outer_j_2 += (int) 1)
    {
      {
        int __hmppcg_end, j_2;
        for (j_2 = (int) 0, __hmppcg_end = ((((outer_j_2 * (int) 3) + (int) 2) > (int) 4095 ? (int) 4095 : ((outer_j_2 * (int) 3) + (int) 2))) - (outer_j_2 * (int) 3); j_2 <= __hmppcg_end; j_2 += (int) 1)
        {
          if ((j_2 + ((int) (outer_j_2 * (int) 3))) == (int) 0)
          {
            tmp1[i_2 + ((int) (outer_i_2 * (int) 4))] = (int) 0;
            y1[i_2 + ((int) (outer_i_2 * (int) 4))] = (int) 0;
          } 
          tmp1[i_2 + ((int) (outer_i_2 * (int) 4))] = ((a[((i_2 + ((int) (outer_i_2 * (int) 4))) * (int) 4096) + (j_2 + ((int) (outer_j_2 * (int) 3)))]) * (x1[j_2 + ((int) (outer_j_2 * (int) 3))])) + (tmp1[i_2 + ((int) (outer_i_2 * (int) 4))]);
          y1[i_2 + ((int) (outer_i_2 * (int) 4))] = ((b[((i_2 + ((int) (outer_i_2 * (int) 4))) * (int) 4096) + (j_2 + ((int) (outer_j_2 * (int) 3)))]) * (x1[j_2 + ((int) (outer_j_2 * (int) 3))])) + (y1[i_2 + ((int) (outer_i_2 * (int) 4))]);
          if ((j_2 + ((int) (outer_j_2 * (int) 3))) == (int) 4095)
          {
            y1[i_2 + ((int) (outer_i_2 * (int) 4))] = (((float) ((int) 43532)) * (tmp1[i_2 + ((int) (outer_i_2 * (int) 4))])) + (((float) ((int) 12313)) * (y1[i_2 + ((int) (outer_i_2 * (int) 4))]));
          } 
        } 
      }
    } 
  }
  }
} 

