
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
// kernel : hmpp_codelet__runGramSchmidt_loop0_
__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void hmpp_codelet__runGramSchmidt_loop0_(  __private int k_11, __global float* pA, __global float* pR)
{
  float priv_priv_pnrm_1;
  int i_5;
  i_5 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_5 <= (long) 0));
  if(!__hmppcg_guard) 
  {
  ;
  priv_priv_pnrm_1 = (int) 0;
  {
    int __hmppcg_end, i_1;
    for (i_1 = (int) 0, __hmppcg_end = (int) 2047; i_1 <= __hmppcg_end; i_1 += (int) 1)
    {
      priv_priv_pnrm_1 = priv_priv_pnrm_1 + ((pA[(i_1 * (int) 2048) + k_11]) * (pA[(i_1 * (int) 2048) + k_11]));
    } 
  }
  pR[(k_11 * (int) 2048) + k_11] = sqrt(priv_priv_pnrm_1);
  }
} 

// kernel : hmpp_codelet__runGramSchmidt_loop1_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runGramSchmidt_loop1_(  __private int k_12, __global float* pA, __global float* pQ, __global float* pR)
{
  int i_2;
  i_2 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(i_2 <= (int) 2047));
  if(!__hmppcg_guard) 
  {
  ;
  pQ[(i_2 * (int) 2048) + k_12] = (pA[(i_2 * (int) 2048) + k_12]) / (pR[(k_12 * (int) 2048) + k_12]);
  }
} 

// kernel : hmpp_codelet__runGramSchmidt_loop2_
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void hmpp_codelet__runGramSchmidt_loop2_(  __private int k_13, __global float* pA, __global float* pQ, __global float* pR)
{
  int j_1;
  j_1 = (get_global_id(1) * get_global_size(0) + get_global_id(0));
  bool __hmppcg_guard = (!(j_1 <= ((int) 2046 - k_13)));
  if(!__hmppcg_guard) 
  {
  ;
  pR[(k_13 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))] = (int) 0;
  {
    int __hmppcg_end, i_3;
    for (i_3 = (int) 0, __hmppcg_end = (int) 2047; i_3 <= __hmppcg_end; i_3 += (int) 1)
    {
      pR[(k_13 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))] = (pR[(k_13 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))]) + ((pQ[(i_3 * (int) 2048) + k_13]) * (pA[(i_3 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))]));
    } 
  }
  {
    int __hmppcg_end, i_4;
    for (i_4 = (int) 0, __hmppcg_end = (int) 2047; i_4 <= __hmppcg_end; i_4 += (int) 1)
    {
      pA[(i_4 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))] = (pA[(i_4 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))]) - ((pQ[(i_4 * (int) 2048) + k_13]) * (pR[(k_13 * (int) 2048) + (j_1 + ((int) (k_13 + (int) 1)))]));
    } 
  }
  }
} 

