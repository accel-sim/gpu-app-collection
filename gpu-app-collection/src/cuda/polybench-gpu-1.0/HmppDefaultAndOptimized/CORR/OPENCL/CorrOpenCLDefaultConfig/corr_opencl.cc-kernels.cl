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
// kernel : hmpp_codelet__runCorr_loop0_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runCorr_loop0_(  __private float pfloat_n, __global float* pdata, __global float* pmean)
{
  int j_1;
  j_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j_1 <= (int) 2047));
  if(!__hmppcg_guard) 
  {
  ;
  pmean[j_1 + (int) 1] = (double) 0.000000000000000E+00;
  {
    int __hmppcg_end, i_1;
    for (i_1 = (int) 0, __hmppcg_end = (int) 2047; i_1 <= __hmppcg_end; i_1 += (int) 1)
    {
      pmean[j_1 + (int) 1] = (pmean[j_1 + (int) 1]) + (pdata[((i_1 + (int) 1) * (int) 2049) + (j_1 + (int) 1)]);
    } 
  }
  pmean[j_1 + (int) 1] = (pmean[j_1 + (int) 1]) / pfloat_n;
  }
} 

// kernel : hmpp_codelet__runCorr_loop1_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runCorr_loop1_(  __private float peps, __private float pfloat_n, __global float* pdata, __global float* pmean, __global float* pstddev)
{
  int j_2;
  j_2 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j_2 <= (int) 2047));
  if(!__hmppcg_guard) 
  {
  ;
  pstddev[j_2 + (int) 1] = (double) 0.000000000000000E+00;
  {
    int __hmppcg_end, i_2;
    for (i_2 = (int) 0, __hmppcg_end = (int) 2047; i_2 <= __hmppcg_end; i_2 += (int) 1)
    {
      float tmp_1;
      tmp_1 = (pdata[((i_2 + (int) 1) * (int) 2049) + (j_2 + (int) 1)]) - (pmean[j_2 + (int) 1]);
      pstddev[j_2 + (int) 1] = (pstddev[j_2 + (int) 1]) + (tmp_1 * tmp_1);
    } 
  }
  pstddev[j_2 + (int) 1] = (pstddev[j_2 + (int) 1]) / pfloat_n;
  pstddev[j_2 + (int) 1] = sqrt((double) (pstddev[j_2 + (int) 1]));
  {
    int lazy_1;
    double res_1;
    lazy_1 = ((pstddev[j_2 + (int) 1]) <= peps) != (int) 0;
    if (lazy_1)
    {
      res_1 = (double) 1.000000000000000E+00;
    } 
    else
    {
      res_1 = pstddev[j_2 + (int) 1];
    } 
    pstddev[j_2 + (int) 1] = res_1;
  } 
  }
} 

// kernel : hmpp_codelet__runCorr_loop2_
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void hmpp_codelet__runCorr_loop2_(  __private float pfloat_n, __global float* pdata, __global float* pmean, __global float* pstddev)
{
  int j_3;
  int i_3;
  j_3 = (get_global_id(0));
  i_3 = (get_global_id(1));
  bool __hmppcg_guard = (!((j_3 <= (int) 2047) & (i_3 <= (int) 2047)));
  if(!__hmppcg_guard) 
  {
  ;
  pdata[((i_3 + (int) 1) * (int) 2049) + (j_3 + (int) 1)] = (pdata[((i_3 + (int) 1) * (int) 2049) + (j_3 + (int) 1)]) - (pmean[j_3 + (int) 1]);
  pdata[((i_3 + (int) 1) * (int) 2049) + (j_3 + (int) 1)] = (pdata[((i_3 + (int) 1) * (int) 2049) + (j_3 + (int) 1)]) / ((sqrt((double) (pfloat_n))) * (pstddev[j_3 + (int) 1]));
  }
} 

// kernel : hmpp_codelet__runCorr_loop3_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runCorr_loop3_(  __global float* pdata, __global float* psymmat)
{
  int j1_1;
  j1_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j1_1 <= (int) 2046));
  if(!__hmppcg_guard) 
  {
  ;
  psymmat[((j1_1 + (int) 1) * (int) 2049) + (j1_1 + (int) 1)] = (double) 1.000000000000000E+00;
  {
    int __hmppcg_end, j2_1;
    for (j2_1 = (int) 0, __hmppcg_end = (int) 2046 - j1_1; j2_1 <= __hmppcg_end; j2_1 += (int) 1)
    {
      psymmat[((j1_1 + (int) 1) * (int) 2049) + (j2_1 + ((int) (j1_1 + (int) 2)))] = (double) 0.000000000000000E+00;
      {
        int __hmppcg_end, i_4;
        for (i_4 = (int) 0, __hmppcg_end = (int) 2047; i_4 <= __hmppcg_end; i_4 += (int) 1)
        {
          psymmat[((j1_1 + (int) 1) * (int) 2049) + (j2_1 + ((int) (j1_1 + (int) 2)))] = (psymmat[((j1_1 + (int) 1) * (int) 2049) + (j2_1 + ((int) (j1_1 + (int) 2)))]) + ((pdata[((i_4 + (int) 1) * (int) 2049) + (j1_1 + (int) 1)]) * (pdata[((i_4 + (int) 1) * (int) 2049) + (j2_1 + ((int) (j1_1 + (int) 2)))]));
        } 
      }
      psymmat[((j2_1 + ((int) (j1_1 + (int) 2))) * (int) 2049) + (j1_1 + (int) 1)] = psymmat[((j1_1 + (int) 1) * (int) 2049) + (j2_1 + ((int) (j1_1 + (int) 2)))];
    } 
  }
  }
} 

// kernel : hmpp_codelet__runCorr_loop4_
__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void hmpp_codelet__runCorr_loop4_(  __global float* psymmat)
{
  int i_5;
  i_5 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_5 <= (long) 0));
  if(!__hmppcg_guard) 
  {
  ;
  psymmat[((int) 2048 * (int) 2049) + (int) 2048] = (double) 1.000000000000000E+00;
  }
} 

