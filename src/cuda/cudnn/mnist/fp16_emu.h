/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// Conversion from/to 16-bit floating point (half-precision).

#if !defined(_FP16_EMU_H_)
#define _FP16_EMU_H_

#include <driver_types.h>
#include <cuda_fp16.h>

// Necessary to ensure visibility of CUDART_VERSION macro
#include <cuda_runtime_api.h>

// Definition of '__half_raw' was not provided before CUDA 9.0.
// '__half_raw' is our type where the unsigned 16-bit integer 
// data member 'x' can be accessed in both CUDA 9.0 and 8.0.
#if CUDART_VERSION < 9000 
typedef __half __half_raw;
#endif

// Internally, in CUDNN we use half1 struct as the FP16 type.
typedef __half half1;

#define HLF_EPSILON 4.887581E-04
#define HLF_MIN     6.103516E-05
#define HLF_MAX     6.550400E+04

half1 cpu_float2half_rn(float f);

float cpu_half2float(half1 h);

static __inline__ __device__ __host__ half1 habs(half1 h)
{
    __half_raw hr = reinterpret_cast<__half_raw&>(h);
    hr.x &= 0x7fffU;
    return reinterpret_cast<half1&>(hr);
}

static __inline__ __device__ __host__ half1 hneg(half1 h)
{
    __half_raw hr = reinterpret_cast<__half_raw&>(h);
    hr.x ^= 0x8000U;
    return reinterpret_cast<half1&>(hr);
}

static __inline__ __device__ __host__ int ishnan(half1 h)
{
    // When input is NaN, exponent is all ones and mantissa is non-zero.
    __half_raw hr = reinterpret_cast<__half_raw&>(h);
    return (hr.x & 0x7c00U) == 0x7c00U && (hr.x & 0x03ffU) != 0;
}

static __inline__ __device__ __host__ int ishinf(half1 h)
{
    // When input is +/- inf, exponent is all ones and mantissa is zero.
    __half_raw hr = reinterpret_cast<__half_raw&>(h);
    return (hr.x & 0x7c00U) == 0x7c00U && (hr.x & 0x03ffU) == 0;
}

static __inline__ __device__ __host__ int ishequ(half1 x, half1 y)
{
    __half_raw xr = reinterpret_cast<__half_raw&>(x);
    __half_raw yr = reinterpret_cast<__half_raw&>(y);
    return ishnan(x) == 0 && ishnan(y) == 0 && xr.x == yr.x;
}

// Returns 0.0000 in FP16 binary form
static __inline__ __device__ __host__ half1 hzero()
{
    __half_raw hr;
    hr.x = 0x0000U;
    return reinterpret_cast<half1&>(hr);
}

// Returns 1.0000 in FP16 binary form
static __inline__ __device__ __host__ half1 hone()
{
    __half_raw hr;
    hr.x = 0x3c00U;
    return reinterpret_cast<half1&>(hr);
}

// Returns quiet NaN, the most significant fraction bit #9 is set
static __inline__ __device__ __host__ half1 hnan()
{
    __half_raw hr;
    hr.x = 0x7e00U;
    return reinterpret_cast<half1&>(hr);
}

// Largest positive FP16 value, corresponds to 6.5504e+04
static __inline__ __device__ __host__ half1 hmax()
{
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    __half_raw hr;
    hr.x = 0x7bffU;
    return reinterpret_cast<half1&>(hr);
}

// Smallest positive (normalized) FP16 value, corresponds to 6.1035e-05
static __inline__ __device__ __host__ half1 hmin()
{
    // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
    __half_raw hr;
    hr.x = 0x0400U;
    return reinterpret_cast<half1&>(hr);
}

#endif  // _FP16_EMU_H_

