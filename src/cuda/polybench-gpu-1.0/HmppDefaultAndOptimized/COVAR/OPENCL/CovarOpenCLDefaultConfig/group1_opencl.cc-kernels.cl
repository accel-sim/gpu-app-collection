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
// kernel : hmpp_codelet__covarLoopa_loop0_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__covarLoopa_loop0_(  __private float pfloat_n, __global float* pdata, __global float* pmean)
{
  int j_2;
  j_2 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j_2 <= (int) 2047));
  if(!__hmppcg_guard) 
  {
  ;
  pmean[j_2 + (int) 1] = (double) 0.000000000000000E+00;
  {
    int __hmppcg_end, i_3;
    for (i_3 = (int) 0, __hmppcg_end = (int) 2047; i_3 <= __hmppcg_end; i_3 += (int) 1)
    {
      pmean[j_2 + (int) 1] = (pmean[j_2 + (int) 1]) + (pdata[((i_3 + (int) 1) * (int) 2049) + (j_2 + (int) 1)]);
    } 
  }
  pmean[j_2 + (int) 1] = (pmean[j_2 + (int) 1]) / pfloat_n;
  }
} 

// kernel : hmpp_codelet__covarLoopb_loop0_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__covarLoopb_loop0_(  __global float* pdata_21, __global float* pmean_11)
{
  int j_3;
  int i_4;
  j_3 = (get_global_id(0));
  i_4 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_3 <= (int) 2047) & (i_4 <= (int) 2047)));
  if(!__hmppcg_guard) 
  {
  ;
  pdata_21[((i_4 + (int) 1) * (int) 2049) + (j_3 + (int) 1)] = (pdata_21[((i_4 + (int) 1) * (int) 2049) + (j_3 + (int) 1)]) - (pmean_11[j_3 + (int) 1]);
  }
} 

// kernel : hmpp_codelet__covarLoopc_loop0_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__covarLoopc_loop0_(  __global float* pdata_11, __global float* psymmat)
{
  int j1_2;
  j1_2 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j1_2 <= (int) 2047));
  if(!__hmppcg_guard) 
  {
  ;
  {
    int __hmppcg_end, j2_2;
    for (j2_2 = (int) 0, __hmppcg_end = (int) 2047 - j1_2; j2_2 <= __hmppcg_end; j2_2 += (int) 1)
    {
      psymmat[((j1_2 + (int) 1) * (int) 2049) + (j2_2 + ((int) (j1_2 + (int) 1)))] = (double) 0.000000000000000E+00;
      {
        int __hmppcg_end, i_5;
        for (i_5 = (int) 0, __hmppcg_end = (int) 2047; i_5 <= __hmppcg_end; i_5 += (int) 1)
        {
          psymmat[((j1_2 + (int) 1) * (int) 2049) + (j2_2 + ((int) (j1_2 + (int) 1)))] = (psymmat[((j1_2 + (int) 1) * (int) 2049) + (j2_2 + ((int) (j1_2 + (int) 1)))]) + ((pdata_11[((i_5 + (int) 1) * (int) 2049) + (j1_2 + (int) 1)]) * (pdata_11[((i_5 + (int) 1) * (int) 2049) + (j2_2 + ((int) (j1_2 + (int) 1)))]));
        } 
      }
      psymmat[((j2_2 + ((int) (j1_2 + (int) 1))) * (int) 2049) + (j1_2 + (int) 1)] = psymmat[((j1_2 + (int) 1) * (int) 2049) + (j2_2 + ((int) (j1_2 + (int) 1)))];
    } 
  }
  }
} 

