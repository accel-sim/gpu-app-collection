***     simpleTemplates    ***
*** NVIDIA CUDA SDK Sample ***
***       readme.txt       ***

/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

---------------------------
-- C++ Templates in CUDA --
---------------------------

Templates are a powerful C++ feature that are made available in CUDA.  They enable 
compile-time polymorphism.  This is useful not only for writing generic functions that 
can be applied to many types, but also for generating multiple versions of a function
based on compile time decisions.

----------------------------------------------
-- Problem: C++ Templates and Shared Memory --
----------------------------------------------

A problem arises in CUDA when we mix functions templatized on type with 
dynamically allocated shared memory. 

For example if you have:

template <type T>
__global__ void foo(T *odata, T* idata)
{
    extern __shared__ T sdata[];
    // ... do stuff with odata, idata, and sdata
}

Then in a host function, try to call foo twice:

foo<int><<<blocks, threads, mem>>>(d_odata, d_idata);
foo<float><<<blocks, threads, mem>>>(d_odata, d_idata);

You get this error:
    "declaration is incompatible with previous "sdata" (declared at line 3)
    extern __declspec(__shared__) T sdata[];

This is because  the CUDA compiler generates two extern arrays of the same name, one 
with type int and one with type float. In general, template functions fail to compile 
when the type of an unsized shared memory array is templatized and the template function 
is called more than once, with a different type for the template parameter that defines 
the shared memory array type. 

---------------------------------------
-- Solution: Template Specialization --
---------------------------------------

This problem has an interesting workaround: we just use C++ template specialization 
to get get the compiler to generate different shared memory variables for different 
types!

First we define a simple wrapper template class, "SharedMem" that just has a method 
getPointer() that returns a pointer to the shared memory array. This template class 
is then specialized for each type we need to support, and getPointer() declares an 
unsized shared memory array and returns the array pointer. 

The key is that each specialized version renames the array so that there are no 
multiple definition conflicts at compile time.

// non-specialized class template
template <class T>
class SharedMem
{
public:
    // Ensure that we won't compile any un-specialized types
    T* getPointer() { error };
};

// specialization for int
template <>
class SharedMem <int>
{
public:
    int* getPointer() { extern __shared__ int s_int[]; return s_int; }
};

// specialization for float
template <>
class SharedMem <float>
{
public:
    float* getPointer() { extern __shared__ float s_float[]; return s_float; }
};


Then to use SharedMem, in our kernel we just instantiate a SharedMem object and get its 
pointer:

template<class T>
__global__ void foo( T* g_idata, T* g_odata)
{
    // shared memory
    // the size is determined by the host application
    
    SharedMem<T> shared;
    T* sdata = shared.getPointer();

    // .. the rest of the code remains unchanged!
}


THe simpleTemplates project includes a header, sharedmem.cuh, which has the above class 
template and specializations for all basic types.  This file can be used in your own 
CUDA projects.