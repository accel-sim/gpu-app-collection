/*
# Copyright (c) 2011-2012 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.   
*/

#pragma once


#if defined(_WIN64) || defined(__LP64__)
        // 64-bit pointer operand constraint for inlined asm
        #define _ASM_PTR_ "l"
#else
        // 32-bit pointer operand constraint for inlined asm
        #define _ASM_PTR_ "r"
#endif

__device__   __inline__ double ld_cg(const double* address){
  double reg;
  asm("ld.global.cg.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ float ld_cg(const float* address){
  float reg;
  asm("ld.global.cg.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__  __inline__ int ld_cg(const int* address){
  int reg;
  asm("ld.global.cg.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ double ld_ca(const double* address){
  double reg;
  asm("ld.global.ca.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ float ld_ca(const float* address){
  float reg;
  asm("ld.global.ca.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__  __inline__ int ld_ca(const int* address){
  int reg;
  asm("ld.global.ca.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ double ld_cs(const double* address){
  double reg;
  asm("ld.global.cs.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ float ld_cs(const float* address){
  float reg;
  asm("ld.global.cs.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__  __inline__ int ld_cs(const int* address){
  int reg;
  asm("ld.global.cs.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ float ld_cv(const float* address)
{
  float reg;
  asm("ld.volatile.global.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ double ld_cv(const double* address)
{
  double reg;
  asm("ld.volatile.global.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
  return reg;
}

__device__   __inline__ int ld_cv(const int* address)
{
  int reg;
  asm("ld.volatile.global.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
  return reg;
}

