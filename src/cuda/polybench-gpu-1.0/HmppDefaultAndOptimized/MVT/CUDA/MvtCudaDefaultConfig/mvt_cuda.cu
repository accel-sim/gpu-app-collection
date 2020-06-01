// ** Original codelet code **
//
// #pragma hmppcg cpiparam __arg0 IN a%hmpp_codelet__runMvt: (1, 0)
// #pragma hmppcg cpiparam __arg1 INOUT x1%hmpp_codelet__runMvt: (1, 1)
// #pragma hmppcg cpiparam __arg2 INOUT x2%hmpp_codelet__runMvt: (1, 2)
// #pragma hmppcg cpiparam __arg3 IN y1%hmpp_codelet__runMvt: (1, 3)
// #pragma hmppcg cpiparam __arg4 IN y2%hmpp_codelet__runMvt: (1, 4)
// 
// #pragma hmppcg cpicall hmpp_codelet__runMvt(__arg0, __arg1, __arg2, __arg3, __arg4): 1
// 
// 
// /* begin of extracted source code for directive set "mvt" */
// 
// 
// # 25 "mvt.c"
// typedef float  DATA_TYPE;
// 
// 
// # 30 "mvt.c"
// void hmpp_codelet__runMvt(DATA_TYPE a[4096][4096], DATA_TYPE x1[4096], DATA_TYPE x2[4096], DATA_TYPE y1[4096], DATA_TYPE y2[4096])
// {
//   int  i, j;
// 
// #pragma hmppcg grid blocksize 32 X 8
// # 9 "<preprocessor>"
// # 36 "mvt.c"
//   for (i = 0 ; i < 4096 ; i++)
//     {
//       for (j = 0 ; j < 4096 ; j++)
//         {
//           x1[i] = x1[i] + a[i][j] * y1[j];
//         }
//     }
// 
// 
// #pragma hmppcg grid blocksize 32 X 8
// # 21 "<preprocessor>"
// # 47 "mvt.c"
//   for (i = 0 ; i < 4096 ; i++)
//     {
//       for (j = 0 ; j < 4096 ; j++)
//         {
//           x2[i] = x2[i] + a[j][i] * y2[j];
//         }
//     }
// }
// 
// 
// /* end of extracted source code for directive set "mvt" */
// 
// 
//
// ** End of original codelet codelet **



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#ifdef _MSC_VER
#  define HMPPCG_RESTRICT
typedef __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#  ifdef _WIN64
typedef int64_t intptr_t;
#  else
typedef int32_t intptr_t;
#  endif
#else
#  if defined(__GNUC__) || defined(__RESTRICT)
#    define HMPPCG_RESTRICT __restrict
#  else
#    define HMPPCG_RESTRICT
#  endif
#  include <stdint.h>
#endif

// Dynamic array
typedef struct __hmppcg_array_struct
{
  void *array;
  size_t *size;
  size_t elsize;
} __hmppcg_array_t;

// Data section
typedef struct __hmppcg_DataSection
{
  size_t from;
  size_t to;
  size_t step;
} __hmppcg_DataSection;


#include <cuda.h>

#if CUDART_VERSION < 2000
#error Bad CUDA Runtime version. CUDA Toolkit 2.0+ required.
#endif

#define HMPP_CONSTMEM_OFFSET 0

#include <map>
#include <string>
// ----------------------------------------------------------------------------
// HMPP CUDA support classes
// ----------------------------------------------------------------------------

#ifndef __HMPP_CUDADATA_H__
#define __HMPP_CUDADATA_H__

#ifndef HMPPCG_WARP_SIZE
#define HMPPCG_WARP_SIZE 32
#endif

enum CopyKind
{
  HostToHost  = 0,
  HostToDevice = 1,
  DeviceToHost = 2,
  DeviceToDevice = 3,
};

inline int hmppcg_check_status(const char *file,int line,cudaError_t status)
{
  if(status != cudaSuccess)
  {
    fprintf(stderr, "%s:%d CUDA Error: %s\n", file, line,
            cudaGetErrorString(status));
    return -1;
  }
  return 0;
}


#define CHECK_STATUS(X) hmppcg_check_status(__FILE__,__LINE__,(X))

#define HMPP_CHECK_GRID_BOUNDARY(x) \
   if(x>65535){\
     fprintf(stderr, "%s:%d Grid Dimension Error: '%s' exceeds the 65535 dimension limit. Please modify the grid size configuration (see the hmppcg grid blocksize pragma) or switch to 2D gridification\n", __FILE__,__LINE__, #x);\
     exit(-1) ;\
   }

#define HMPP_CHECK_BLOCK_BOUNDARY(x) \
  if(x > devProp.maxThreadsPerBlock){		\
    fprintf(stderr, "%s:%d Number of threads per block exceeds for the HWA: it is '%d' and HWA supports up to '%d'. Please modify the block size configuration (see the hmppcg grid blocksize pragma)\n", __FILE__,__LINE__, x, devProp.maxThreadsPerBlock); \
    exit(-1) ;								\
  }

// ----------------------------------------------------------------------------
// class DefaultPolicy
// ----------------------------------------------------------------------------

struct DefaultPolicy
{
public:

  DefaultPolicy()
  {
  }

  virtual ~DefaultPolicy()
  {
  }

  int deviceAlloc(void **ptr,size_t size)
  {
    if( CHECK_STATUS(cudaStreamCreate(&stream_)) != 0 ) return -1;
    if( CHECK_STATUS(cudaMalloc(ptr,size)) != 0 ) return -1;
#if CUDA_VERSION >= 3020
    if( CHECK_STATUS(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync)) != 0)
      return -1;
#else
    if( CHECK_STATUS(cudaEventCreateWithFlags(&event, cudaEventBlockingSync)) != 0)
      return -1;
#endif
    return 0;
  }

  int deviceFree(void *ptr)
  {
    if( CHECK_STATUS(cudaStreamDestroy(stream_)) != 0) return -1;
    if( CHECK_STATUS(cudaFree(ptr)) != 0) return -1;
    if( CHECK_STATUS(cudaEventDestroy(event)) != 0) return -1;
    return 0;
  }

  int deviceMemcpy(void *dst,const void *src,size_t size,CopyKind kind,bool async)
  {
    static cudaMemcpyKind cudaKind[]
      = {cudaMemcpyHostToHost,
         cudaMemcpyHostToDevice,
         cudaMemcpyDeviceToHost,
         cudaMemcpyDeviceToDevice };

    if(async)
    {
      return CHECK_STATUS(cudaMemcpyAsync(dst,src,size,cudaKind[kind],stream_));
    }
    else
    {
      return CHECK_STATUS(cudaMemcpy(dst,src,size,cudaKind[kind]));
    }
  }

  int makeStreamWait(cudaStream_t wstream)
  {
    int status;
    status = CHECK_STATUS(cudaEventRecord(event, stream_));
    if (status != 0)
      return status;
#if CUDA_VERSION >= 3020
    return CHECK_STATUS(cudaStreamWaitEvent(wstream, event, 0));
#else
    return CHECK_STATUS(cudaEventSynchronize(event));
#endif
  }

  int waitOnEvent(cudaEvent_t wevent)
  {
#if CUDA_VERSION >= 3020
    return CHECK_STATUS(cudaStreamWaitEvent(stream_, wevent, 0));
#else
    return CHECK_STATUS(cudaEventSynchronize(wevent));
#endif
  }


  int deviceWait()
  {
    return CHECK_STATUS(cudaStreamSynchronize(stream_));
  }

private:
  cudaStream_t stream_;
  cudaEvent_t event;
};

// ----------------------------------------------------------------------------
// class ConstantPolicy
// ----------------------------------------------------------------------------

#ifndef HMPP_CONSTMEM_SIZE
#define HMPP_CONSTMEM_SIZE 2048
#endif

__constant__ int64_t hmpp_constmem[HMPP_CONSTMEM_SIZE / 8];

/// Shared memory array is aligned on 64 bit thanks to that (to avoid an nvcc compilation error)
extern __shared__ int64_t hmpp_sharedmem[];

struct ConstantPolicy
{
public:
  ConstantPolicy()
  {
    static bool initialized = false;
    if(!initialized)
    {
      next_offset_ = HMPP_CONSTMEM_OFFSET;
      initialized = true;
    }
    offset_ = -1;
  }

  virtual ~ConstantPolicy()
  {

  }

  void setStaticOffset(int offset)
  {
    offset_ = offset;

    while(offset_  %  8)
        offset_ ++;
  }

  int deviceAlloc(void **ptr, size_t size)
  {
#if CUDA_VERSION >= 3020
    if( CHECK_STATUS(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync)) != 0) return -1;
#else
    if( CHECK_STATUS(cudaEventCreateWithFlags(&event, cudaEventBlockingSync)) != 0) return -1;
#endif
    if(offset_ != -1)
    {
      if((offset_ + size) >= HMPP_CONSTMEM_SIZE)
        return -1;

      (*ptr) = (void *)offset_;
      return 0;
    }

    if((next_offset_ + size) >= HMPP_CONSTMEM_SIZE)
      return -1;

    (*ptr) = (void *)next_offset_;
    next_offset_ += size;
    return 0;
  }

  int deviceFree(void *ptr)
  {
    return 0;
  }

  int deviceMemcpy(void *dst,const void *src,size_t size,CopyKind kind,bool async)
  {
    size_t offset;

    switch(kind)
    {
    case HostToDevice:
      offset = (size_t)dst;
      return CHECK_STATUS(cudaMemcpyToSymbol(hmpp_constmem,src,size,offset,cudaMemcpyHostToDevice));
    case DeviceToHost:
      offset = (size_t)src;
      return CHECK_STATUS(cudaMemcpyFromSymbol(dst,hmpp_constmem,size,offset,cudaMemcpyDeviceToHost));
    default:
      return -1;
    }
  }

  int makeStreamWait(cudaStream_t wstream)
  {
    int status;
    /* stream 0 at the moment */
    status = CHECK_STATUS(cudaEventRecord(event, 0));
    if (status != 0)
      return status;
#if CUDA_VERSION >= 3020
    return CHECK_STATUS(cudaStreamWaitEvent(wstream, event, 0));
#else
    return CHECK_STATUS(cudaEventSynchronize(event));
#endif
  }

  int waitOnEvent(cudaEvent_t wevent)
  {
    /* stream 0 at the moment */
#if CUDA_VERSION >= 3020
    return CHECK_STATUS(cudaStreamWaitEvent(0, wevent, 0));
#else
    return CHECK_STATUS(cudaEventSynchronize(wevent));
#endif
  }

  int deviceWait()
  {
    return 0;
  }

private:
  static size_t next_offset_;
  int offset_;
  cudaEvent_t event;
};

size_t ConstantPolicy::next_offset_;


// ----------------------------------------------------------------------------
// class Lazy
// ----------------------------------------------------------------------------

template <typename Policy>
struct Lazy
{
  char * value;
  bool valid;
  bool allocated;
  void ** devaddr;
  Policy * policy;
  size_t size;


  Lazy(size_t elem_size)
  {
    value = new char[elem_size];
  }


  ~Lazy()
  {
    delete[] value;
  }


  int requireDeviceAlloc()
  {
    if(!allocated)
    {
      allocated = true;
      return policy->deviceAlloc(devaddr,size);
    }

    else
    {
      return 0;
    }
  }

};


// ----------------------------------------------------------------------------
// class Element
// ----------------------------------------------------------------------------

template <typename T,typename Policy>
struct Element
{
  Element(void * const * device_addr, size_t offset, Policy *policy, Lazy<Policy> * lazy)
    : device_addr_(device_addr) , offset_(offset), policy_(policy), lazy_(lazy)
  {

  }

  Element &operator=(const T & value)
  {
    if(lazy_)
    {
      *((T *)(lazy_->value)) = value;
      lazy_->valid = true;
      return *this;
    }

    if(lazy_)
      lazy_->requireDeviceAlloc();

    policy_->deviceMemcpy(((char*)(*device_addr_)) + offset_,(const char*)&value,ElemSize,HostToDevice,false);
    return *this;
  }

  Element &operator=(const Element & src)
  {
    if(src.lazy_ &&  src.lazy_->valid)
    {
      lazy_->valid = true;
      *((T *)(lazy_->value)) =  *((T *)(src.lazy_->value));
      return *this;
    }

    if(lazy_)
      lazy_->requireDeviceAlloc();
    if(src.lazy_)
      src.lazy_->requireDeviceAlloc();

    policy_->deviceMemcpy(((char*)(*device_addr_)) + offset_,((const char*)(*src.device_addr_)) + src.offset_,
                        ElemSize,DeviceToDevice,false);

    if(lazy_)
    {
      lazy_->valid = false;
    }
    return *this;
  }

  operator T()
  {
    if(lazy_ && lazy_->valid)
      return *((T *)(lazy_->value));

    T res;
    if(lazy_)
      lazy_->requireDeviceAlloc();

    policy_->deviceMemcpy(&res,((const char*)(*device_addr_)) + offset_,ElemSize,DeviceToHost,false);

    if(lazy_)
    {
      *((T *)(lazy_->value)) = res;
      lazy_->valid = true;
    }
    return res;
  }

  typedef T Type;
  enum { ElemSize = sizeof(T) };

private:
  size_t offset_;
  void *const* device_addr_;
  Policy *policy_;

public:
  Lazy<Policy> * lazy_;
};


enum DataFlags
{
    DEFAULT = 0x0,
    LAZY    = 0x1
};

// ----------------------------------------------------------------------------
// class Data
// ----------------------------------------------------------------------------

template <typename T,typename Policy>
class Data
{
public:
  typedef T Type;
  typedef Element<T,Policy> ElementType;

  enum { ElemSize = sizeof(T) };

  Data(const char * name, unsigned int flags = DEFAULT)
    : name_(name), flags_(flags),
      dim_(0), sizes_(0), size_(0),
      host_addr_(0), device_addr_(0)
  {
    policy_ = new Policy;

    if(flags_ & LAZY)
    {
      lazy_ = new Lazy<Policy>(ElemSize);
      lazy_->valid = false;
      lazy_->devaddr = 0;
      lazy_->policy = policy_;
    }
    else
      lazy_ = 0;

  }

  ~Data()
  {
    free();
    delete policy_;
    if(lazy_)
      delete lazy_;
  }

  int allocate(unsigned int dim,
               size_t idx0 = 0, size_t idx1 = 0, size_t idx2 = 0, size_t idx3 = 0,
               size_t idx4 = 0, size_t idx5 = 0, size_t idx6 = 0, size_t idx7 = 0,
               size_t idx8 = 0, size_t idx9 = 0, size_t idxA = 0, size_t idxB = 0)
  {
    const size_t sizes[] = { idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idxA, idxB };
    return allocate2(dim,sizes);
  }

  int allocate3(unsigned int dim_p, const size_t * sizes_p)
  {
    size_t sizes[2];

    sizes[0] = 1;
    sizes[1] = 0;
    for(int d = 0 ; d < dim_p ; d++)
    {
      sizes[0] *= sizes_p[d];
    }

    return allocate2(1, sizes);
  }

  int allocate2(unsigned int dim, const size_t * sizes)
  {
    dim_ = dim;
    sizes_ = new size_t[dim];
    dimSizes_ = new size_t[dim];

    size_ = ElemSize;
    for(int d=0;d<dim;d++)
    {
      sizes_[d] = sizes[d];
      size_ *= sizes_[d];

      size_t size = 1;
      for(int d2=d+1;d2<dim;d2++)
        size*=sizes[d2];
      dimSizes_[d] = size;
    }

    if(lazy_)
    {
      lazy_->allocated = false;
      lazy_->devaddr = &device_addr_;
      lazy_->size = size_;
      return 0;
    }
    else
      return policy_->deviceAlloc(&device_addr_,size_);
  }

  int free()
  {
    if(sizes_)
    {
      delete [] sizes_;
      delete [] dimSizes_;
      sizes_ = 0;
      dim_ = 0;
      size_ = 0;
    }

    if(device_addr_)
    {
      if(policy_->deviceFree(device_addr_) != 0)
       return -1;
      device_addr_ = 0;
    }
    return 0;
  }

  int download(void * host_addr,bool async)
  {
    if(lazy_ && lazy_->valid)
    {
      *((T *)host_addr) = *((T *)(lazy_->value));
      return 0;
    }

    if(lazy_)
    {
      lazy_->requireDeviceAlloc();
    }

    int sts = policy_->deviceMemcpy(host_addr,device_addr_,size_,DeviceToHost,async);

    if(lazy_)
    {
      lazy_->valid = true;
      *((T *)(lazy_->value)) = *((T *)host_addr);
    }

    return sts;
  }

  int upload(const void * host_addr,bool async)
  {
    if(lazy_)
    {
      lazy_->valid = true;
      *((T *)(lazy_->value)) = * ((T *)host_addr);
      lazy_->requireDeviceAlloc();
    }

    return policy_->deviceMemcpy(device_addr_,host_addr,size_,HostToDevice,async);
  }

  int downloadSection(void *host_addr,const __hmppcg_DataSection *sections,bool async)
  {
    return sectionCopy(host_addr,device_addr_,sections,DeviceToHost,async);
  }

  int uploadSection(const void *host_addr,const __hmppcg_DataSection *sections,bool async)
  {
    return sectionCopy(device_addr_,host_addr,sections,HostToDevice,async);
  }

  int makeStreamWait(cudaStream_t wstream)
  {
    if(lazy_)
      lazy_->requireDeviceAlloc();
    return policy_->makeStreamWait(wstream);
  }

  int waitOnEvent(cudaEvent_t wevent)
  {
    return policy_->waitOnEvent(wevent);
  }

  int waitTransfer()
  {
    return policy_->deviceWait();
  }

  ElementType operator()(size_t idx0 = 0, size_t idx1 = 0, size_t idx2 = 0, size_t idx3 = 0,
                         size_t idx4 = 0, size_t idx5 = 0, size_t idx6 = 0, size_t idx7 = 0,
                         size_t idx8 = 0, size_t idx9 = 0, size_t idxA = 0, size_t idxB = 0)
  {
    size_t sizes[] = { idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idxA, idxB };
    return at(sizes);
  }

  ElementType at(size_t *idx)
  {
    size_t offset = idx[0];
    return ElementType(&device_addr_,offset*ElemSize,policy_,lazy_);
  }

  template <typename Y>
  Element<Y,Policy> at(size_t offset)
  {
    return Element<Y,Policy>(&device_addr_,offset,policy_,lazy_);
  }

  ElementType operator=(const T & value)
  {
    ElementType res(&device_addr_,0,policy_,lazy_);
    res = value;
    return res;
  }

  ElementType operator=(const Data &data)
  {
    return operator=(data.value());
  }

  T value() const
  {
    ElementType res(&device_addr_,0,policy_,lazy_);
    return (T)res;
  }

  operator T()
  {
    return value();
  }

  T *getDeviceAddr()
  {
    if(lazy_)
      lazy_->requireDeviceAlloc();

    if(lazy_ && lazy_->valid)
    {
      policy_->deviceMemcpy(device_addr_,lazy_->value,size_,HostToDevice,false);
    }

    return (T*)device_addr_;
  }

  void invalidateLazy()
  {
    if(lazy_)
    {
      lazy_->valid = false;
    }
  }

private:

  Data(const Data &data) {}

  int sectionCopy(char *dst,const char *src,int offset,int cur, const __hmppcg_DataSection *sections,int lastdense,CopyKind kind,bool async)
  {
    int d;
    int size = 1;
    for(d=cur+1;d<dim_;d++)
      size *= sizes_[d];

    if(cur<(lastdense-1))
    {
      int x;
      for(x=sections[cur].from;x<=sections[cur].to;x+=sections[cur].step)
        if(sectionCopy(dst,src,offset+x*size,cur+1,sections,lastdense,kind,async) != 0)
          return -1;
    }
    else
    {
      int step = sections[cur].step;
      if(step == 1)
      {
        int start = (offset + sections[cur].from * size) * ElemSize;
        int total = (sections[cur].to - sections[cur].from + 1) * size * ElemSize;
        return policy_->deviceMemcpy(dst+start,src+start,total,kind,async);
      }
      else
      {
        int x;
        for(x=sections[cur].from;x<=sections[cur].to;x+=step)
        {
          int off = (offset + x * size) * ElemSize;
          if(policy_->deviceMemcpy(dst+off,src+off,size * ElemSize,kind,async) != 0)
            return -1;
        }
      }
    }
    return 0;
  }

  int sectionCopy(void *dst,const void *src, const __hmppcg_DataSection *sections,CopyKind kind,bool async)
  {
    int i;
    int lastdense = dim_;
    for (i = dim_ - 1 ; i >= 0 ; i --)
    {
      if ((sections[i].from == 0) && (sections[i].to == sizes_[i] - 1) && (sections[i].step == 1))
        lastdense = i;
      else
        break;
    }
    return sectionCopy((char*)dst,(const char*)src,0,0,sections,lastdense,kind,async);
  }

  const char * name_;
  size_t flags_;
  void *device_addr_;
  void *host_addr_;
  size_t dim_;
  size_t *sizes_;
  size_t *dimSizes_;
  size_t size_;

  Lazy<Policy> * lazy_;

public:
  Policy *policy_;
};

// ---------------------------------------------------------------------------
// User data
// ---------------------------------------------------------------------------
class UserData{
public:
  virtual ~UserData(){}
  UserData(){}
};

#define __HMPPCG_COMPLEX_FLOAT_DEFINED
typedef float2 __hmppcg_complex_float;

#define __HMPPCG_COMPLEX_DOUBLE_DEFINED
typedef double2 __hmppcg_complex_double;


// ---------------------------------------------------------------------------
// Allocatable Arrays
// ---------------------------------------------------------------------------
template <const size_t nb_dims> struct AArrayDesc {
  int lbounds_[nb_dims];
  size_t sizes_[nb_dims];
  size_t wholesize_;
};

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATE
#define __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATE( var, type, nb_dims, ... )                 \
        { int alloc_ranges[] = { __VA_ARGS__ };                                        \
          int hmppcg_alloc_i;                                                          \
          var ## _aarray_desc.wholesize_ = 1;                                          \
          for(hmppcg_alloc_i=0; hmppcg_alloc_i<nb_dims; hmppcg_alloc_i++){             \
            int hmppcg_alloc_first = alloc_ranges[2*hmppcg_alloc_i];                   \
            int hmppcg_alloc_last  = alloc_ranges[2*hmppcg_alloc_i + 1];               \
            int hmppcg_alloc_size  = hmppcg_alloc_last - hmppcg_alloc_first + 1;       \
            var ## _aarray_desc.lbounds_[hmppcg_alloc_i] = hmppcg_alloc_first;         \
            var ## _aarray_desc.sizes_[hmppcg_alloc_i] = hmppcg_alloc_size;            \
            var ## _aarray_desc.wholesize_ *= hmppcg_alloc_size;                       \
          }                                                                            \
          if((hmppcg_status_ = var.allocate2(nb_dims, var ## _aarray_desc.sizes_)))    \
            return;                                                                    \
        }
#endif

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_DEALLOCATE
#define __HMPPCG_ALLOCATABLE_ARRAY_DEALLOCATE( var ) \
        {                                            \
          var.free();                                \
        }
#endif

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATED
#define __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATED( var ) \
        (var.getDeviceAddr() != NULL)
#endif //__HMPPCG_ALLOCATABLE_ARRAY_ALLOCATED

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE
#define __HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE( var ) \
        var ## _aarray_desc.wholesize_
#endif //__HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_SIZE
#define __HMPPCG_ALLOCATABLE_ARRAY_SIZE( var, d ) \
        var ## _aarray_desc.sizes_[d]
#endif //__HMPPCG_ALLOCATABLE_ARRAY_SIZE

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_LBOUND
#define __HMPPCG_ALLOCATABLE_ARRAY_LBOUND( var, d ) \
        var ## _aarray_desc.lbounds_[d]
#endif //__HMPPCG_ALLOCATABLE_ARRAY_LBOUND

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_UBOUND
#define __HMPPCG_ALLOCATABLE_ARRAY_UBOUND( var, d ) \
        (var ## _aarray_desc.sizes_[d] + var ## _aarray_desc.lbounds_[d] - 1)
#endif //__HMPPCG_ALLOCATABLE_ARRAY_UBOUND

#ifndef __HMPP_INT_POW_FUNC
#define __HMPP_INT_POW_FUNC(func_ext_name, func_type)                             \
  __device__ func_type hmpp_pow ##func_ext_name ( func_type base, func_type exp ) \
  {                                                                               \
    if(exp < 0)                                                                   \
      return 0;                                                                   \
    func_type result = 1;                                                         \
    while (exp)                                                                   \
    {                                                                             \
      if (exp & 1)                                                                \
        result *= base;                                                           \
      exp >>= 1;                                                                  \
      base *= base;                                                               \
    }                                                                             \
    return result;                                                                \
  }
#endif

__HMPP_INT_POW_FUNC( i64, int64_t );
__HMPP_INT_POW_FUNC( i32, int32_t );
__HMPP_INT_POW_FUNC( i16, int16_t );
__HMPP_INT_POW_FUNC( i8,  int8_t );

#ifndef __HMPP_UINT_POW_FUNC
#define __HMPP_UINT_POW_FUNC(func_ext_name, func_type)                            \
  __device__ func_type hmpp_pow ##func_ext_name ( func_type base, func_type exp ) \
  {                                                                               \
    func_type result = 1;                                                         \
    while (exp)                                                                   \
    {                                                                             \
      if (exp & 1)                                                                \
        result *= base;                                                           \
      exp >>= 1;                                                                  \
      base *= base;                                                               \
    }                                                                             \
    return result;                                                                \
  }
#endif

__HMPP_UINT_POW_FUNC( ui64, uint64_t );
__HMPP_UINT_POW_FUNC( ui32, uint32_t );
__HMPP_UINT_POW_FUNC( ui16, uint16_t );
__HMPP_UINT_POW_FUNC( ui8,  uint8_t );

#endif // __HMPP_CUDADATA_H__

#ifndef __HMPPCG_COMPLEX_DOUBLE_DEFINED
#define __HMPPCG_COMPLEX_DOUBLE_DEFINED
typedef struct 
{
  double x;
  double y;
}__hmppcg_complex_double;
#endif /* __HMPPCG_COMPLEX_DOUBLE_DEFINED */

#ifndef __HMPPCG_COMPLEX_FLOAT_DEFINED
#define __HMPPCG_COMPLEX_FLOAT_DEFINED
typedef struct 
{
  float x;
  float y;
}__hmppcg_complex_float;
#endif /* __HMPPCG_COMPLEX_FLOAT_DEFINED */

template <const unsigned int blockDimX__, const unsigned int blockDimY__>
__global__ void hmpp_codelet__runMvt_loop0_(  float * HMPPCG_RESTRICT a, float * HMPPCG_RESTRICT x1, float * HMPPCG_RESTRICT y1)
{
  int32_t i_1;
  i_1 = (blockDimX__ * blockDimY__ * blockIdx.x  +  threadIdx.y * blockDimX__  +  threadIdx.x);
  bool __hmppcg_guard = (!(i_1 <= 4095));
  if(__hmppcg_guard) { goto __hmppcg_label1; };
  {
    int32_t __hmppcg_end, j_1;
    for (j_1 = 0, __hmppcg_end = 4095; j_1 <= __hmppcg_end; j_1 += 1)
    {
      x1[i_1] = (x1[i_1]) + ((a[(i_1 * 4096) + j_1]) * (y1[j_1]));
    } 
  }
  __hmppcg_label1:;
} 

template <const unsigned int blockDimX__, const unsigned int blockDimY__>
__global__ void hmpp_codelet__runMvt_loop1_(  float * HMPPCG_RESTRICT a, float * HMPPCG_RESTRICT x2, float * HMPPCG_RESTRICT y2)
{
  int32_t i_2;
  i_2 = (blockDimX__ * blockDimY__ * blockIdx.x  +  threadIdx.y * blockDimX__  +  threadIdx.x);
  bool __hmppcg_guard = (!(i_2 <= 4095));
  if(__hmppcg_guard) { goto __hmppcg_label3; };
  {
    int32_t __hmppcg_end, j_2;
    for (j_2 = 0, __hmppcg_end = 4095; j_2 <= __hmppcg_end; j_2 += 1)
    {
      x2[i_2] = (x2[i_2]) + ((a[(j_2 * 4096) + i_2]) * (y2[j_2]));
    } 
  }
  __hmppcg_label3:;
} 

void hmpp_codelet__runMvt(  int &hmppcg_status_, void * __h, const cudaDeviceProp &devProp, cudaStream_t kernel_stream, cudaEvent_t kernel_event, Data<float,DefaultPolicy> & a, Data<float,DefaultPolicy> & x1, Data<float,DefaultPolicy> & x2, Data<float,DefaultPolicy> & y1, Data<float,DefaultPolicy> & y2)

{
  if(1LL)
  {
    unsigned int gridDimX__ = 16LL;
    HMPP_CHECK_GRID_BOUNDARY(gridDimX__);
    unsigned int gridDimY__ = 1LL;
    HMPP_CHECK_GRID_BOUNDARY(gridDimY__);
    dim3 dim_grid(gridDimX__, gridDimY__);
    const unsigned int blockDimX__ = 32LL;
    const unsigned int blockDimY__ = 8LL;
    HMPP_CHECK_BLOCK_BOUNDARY(blockDimX__*blockDimY__);
  #if CUDA_VERSION >= 3020
    a.makeStreamWait(kernel_stream);
    x1.makeStreamWait(kernel_stream);
    y1.makeStreamWait(kernel_stream);
  #else
    if ((hmppcg_status_ = CHECK_STATUS(cudaThreadSynchronize()))) return;
  #endif
    dim3 dim_block(blockDimX__, blockDimY__);
    hmpp_codelet__runMvt_loop0_<blockDimX__, blockDimY__><<<dim_grid, dim_block, 0LL, kernel_stream>>>(a.getDeviceAddr(), x1.getDeviceAddr(), y1.getDeviceAddr());
    if ((hmppcg_status_ = CHECK_STATUS(cudaGetLastError()))) return;
  #if CUDA_VERSION >= 3020
    if((hmppcg_status_ = CHECK_STATUS(cudaEventRecord(kernel_event, kernel_stream)))) return;
    a.waitOnEvent(kernel_event);
    x1.waitOnEvent(kernel_event);
    y1.waitOnEvent(kernel_event);
  #else
    if ((hmppcg_status_ = CHECK_STATUS(cudaThreadSynchronize()))) return;
  #endif
    
  };
  if(1LL)
  {
    unsigned int gridDimX__ = 16LL;
    HMPP_CHECK_GRID_BOUNDARY(gridDimX__);
    unsigned int gridDimY__ = 1LL;
    HMPP_CHECK_GRID_BOUNDARY(gridDimY__);
    dim3 dim_grid(gridDimX__, gridDimY__);
    const unsigned int blockDimX__ = 32LL;
    const unsigned int blockDimY__ = 8LL;
    HMPP_CHECK_BLOCK_BOUNDARY(blockDimX__*blockDimY__);
  #if CUDA_VERSION >= 3020
    a.makeStreamWait(kernel_stream);
    x2.makeStreamWait(kernel_stream);
    y2.makeStreamWait(kernel_stream);
  #else
    if ((hmppcg_status_ = CHECK_STATUS(cudaThreadSynchronize()))) return;
  #endif
    dim3 dim_block(blockDimX__, blockDimY__);
    hmpp_codelet__runMvt_loop1_<blockDimX__, blockDimY__><<<dim_grid, dim_block, 0LL, kernel_stream>>>(a.getDeviceAddr(), x2.getDeviceAddr(), y2.getDeviceAddr());
    if ((hmppcg_status_ = CHECK_STATUS(cudaGetLastError()))) return;
  #if CUDA_VERSION >= 3020
    if((hmppcg_status_ = CHECK_STATUS(cudaEventRecord(kernel_event, kernel_stream)))) return;
    a.waitOnEvent(kernel_event);
    x2.waitOnEvent(kernel_event);
    y2.waitOnEvent(kernel_event);
  #else
    if ((hmppcg_status_ = CHECK_STATUS(cudaThreadSynchronize()))) return;
  #endif
    
  };
} 


// HMPP_API
#ifdef __cplusplus
#define HMPP_EXTERN extern "C"
#else
#define HMPP_EXTERN
#endif

#ifdef _WIN32
#define HMPP_EXPORT __declspec(dllexport)
#define HMPP_INLINE __inline
#else
#define HMPP_EXPORT
#define HMPP_INLINE inline
#endif

#define HMPP_API HMPP_EXTERN HMPP_EXPORT

// HMPPCG_POP_HASH
#define HMPPCG_POP_HASH(major,minor) (((major)<<16)|(minor))



// ---------------------------------------------------------------------------
// HMPP handle
// ---------------------------------------------------------------------------
typedef struct hmpp_handle_struct
{
  Data<float,DefaultPolicy> * __arg0;
  Data<float,DefaultPolicy> * __arg1;
  Data<float,DefaultPolicy> * __arg2;
  Data<float,DefaultPolicy> * __arg3;
  Data<float,DefaultPolicy> * __arg4;
  cudaDeviceProp devProp;
  cudaStream_t kernel_stream;
  cudaEvent_t kernel_event;
  std::map<std::string,UserData*> map_user_data;
} hmpp_handle_t;


// ---------------------------------------------------------------------------
// hmpp_createInstance()
// ---------------------------------------------------------------------------
HMPP_API hmpp_handle_t * hmpp_createInstance()
{
  hmpp_handle_t * __h = new hmpp_handle_t;
  if(!__h) return 0;
  if(CHECK_STATUS(cudaStreamCreate(&__h->kernel_stream)) != 0) return NULL;
  #if CUDA_VERSION >= 3020
  if(CHECK_STATUS(cudaEventCreateWithFlags(&__h->kernel_event, cudaEventDisableTiming | cudaEventBlockingSync)) != 0) return NULL;
  #else
  if(CHECK_STATUS(cudaEventCreateWithFlags(&__h->kernel_event, cudaEventBlockingSync)) != 0) return NULL;
  #endif
  __h->__arg0 = NULL;
  __h->__arg1 = NULL;
  __h->__arg2 = NULL;
  __h->__arg3 = NULL;
  __h->__arg4 = NULL;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&(__h->devProp), device);
  return __h;
}

// ---------------------------------------------------------------------------
// hmpp_freeInstance()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_freeInstance(hmpp_handle_t * __h)
{
  delete __h->__arg0;
  delete __h->__arg1;
  delete __h->__arg2;
  delete __h->__arg3;
  delete __h->__arg4;
  cudaStreamDestroy(__h->kernel_stream);
  cudaEventDestroy(__h->kernel_event);
  __h->kernel_stream = 0;
  for(std::map<std::string,UserData*>::const_iterator it = __h->map_user_data.begin(); it != __h->map_user_data.end(); it++) { delete it->second; }
  delete(__h);
  return 0;
}

// ---------------------------------------------------------------------------
// hmpp_allocateOnHWA()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_allocateOnHWA(hmpp_handle_t * __h, int major, int minor, const size_t * size, size_t elsize, int dim)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      __h->__arg0 = new Data<float,DefaultPolicy>("__arg0", DEFAULT);
      return __h->__arg0->allocate2(dim, size);
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      __h->__arg1 = new Data<float,DefaultPolicy>("__arg1", DEFAULT);
      return __h->__arg1->allocate2(dim, size);
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      __h->__arg2 = new Data<float,DefaultPolicy>("__arg2", DEFAULT);
      return __h->__arg2->allocate2(dim, size);
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      __h->__arg3 = new Data<float,DefaultPolicy>("__arg3", DEFAULT);
      return __h->__arg3->allocate2(dim, size);
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      __h->__arg4 = new Data<float,DefaultPolicy>("__arg4", DEFAULT);
      return __h->__arg4->allocate2(dim, size);
    }
    default: return -1;
  }
}

HMPP_API int hmpp_allocateOutputOnHWA(hmpp_handle_t * __h, int major, int minor, const size_t * size, size_t elsize, int dim)
 { return hmpp_allocateOnHWA(__h, major, minor, size, elsize, dim); }

HMPP_API int hmpp_allocateInputOnHWA(hmpp_handle_t * __h, int major, int minor, const size_t * size, size_t elsize, int dim)
 { return hmpp_allocateOnHWA(__h, major, minor, size, elsize, dim); }

HMPP_API int hmpp_allocateInOutOnHWA(hmpp_handle_t * __h, int major, int minor, const size_t * size, size_t elsize, int dim)
 { return hmpp_allocateOnHWA(__h, major, minor, size, elsize, dim); }



// ---------------------------------------------------------------------------
// hmpp_readDataFromHWA()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_readDataFromHWA(hmpp_handle_t * __h, int major, int minor, void * data, const size_t * size, size_t elsize, int dim, int async)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      return __h->__arg0->download(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      return __h->__arg1->download(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      return __h->__arg2->download(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      return __h->__arg3->download(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      return __h->__arg4->download(data,async!=0);
    }
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// hmpp_writeDataToHWA()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_writeDataToHWA(hmpp_handle_t * __h, int major, int minor, const void * data, const size_t * size, size_t elsize, int dim, int async)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      return __h->__arg0->upload(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      return __h->__arg1->upload(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      return __h->__arg2->upload(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      return __h->__arg3->upload(data,async!=0);
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      return __h->__arg4->upload(data,async!=0);
    }
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// hmpp_readDataSectionFromHWA()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_readDataSectionFromHWA(hmpp_handle_t * __h, int major, int minor, void * data, const __hmppcg_DataSection *section, const size_t * size, size_t elsize, int dim, int async)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      return __h->__arg0->downloadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      return __h->__arg1->downloadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      return __h->__arg2->downloadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      return __h->__arg3->downloadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      return __h->__arg4->downloadSection(data,section,async!=0);
    }
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// hmpp_writeDataSectionToHWA()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_writeDataSectionToHWA(hmpp_handle_t * __h, int major, int minor, const void * data, const __hmppcg_DataSection *section, const size_t * size, size_t elsize, int dim, int async)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      return __h->__arg0->uploadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      return __h->__arg1->uploadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      return __h->__arg2->uploadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      return __h->__arg3->uploadSection(data,section,async!=0);
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      return __h->__arg4->uploadSection(data,section,async!=0);
    }
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// hmpp_waitForWriteTransfer()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_waitForWriteTransfer(hmpp_handle_t * __h, int major, int minor)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      return __h->__arg0->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      return __h->__arg1->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      return __h->__arg2->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      return __h->__arg3->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      return __h->__arg4->waitTransfer();
    }
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// hmpp_waitForReadTransfer()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_waitForReadTransfer(hmpp_handle_t * __h, int major, int minor)
{
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // a@hmpp_codelet__runMvt
    {
      return __h->__arg0->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,1): // x1@hmpp_codelet__runMvt
    {
      return __h->__arg1->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,2): // x2@hmpp_codelet__runMvt
    {
      return __h->__arg2->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,3): // y1@hmpp_codelet__runMvt
    {
      return __h->__arg3->waitTransfer();
    }
    case HMPPCG_POP_HASH(1,4): // y2@hmpp_codelet__runMvt
    {
      return __h->__arg4->waitTransfer();
    }
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// hmpp_codeletsAreReentrant()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_codeletsAreReentrant()
{
  return 0;
}

// ---------------------------------------------------------------------------
// hmpp_start()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_start(hmpp_handle_t * __h, int __id, int __async)
{
  int status = 0;
  switch(__id) { 
    case 1: // hmpp_codelet__runMvt(__arg0,__arg1,__arg2,__arg3,__arg4)
      hmpp_codelet__runMvt(status, __h, __h->devProp, __h->kernel_stream, __h->kernel_event,   (*__h->__arg0), (*__h->__arg1), (*__h->__arg2), (*__h->__arg3), (*__h->__arg4));
      return status;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// hmpp_wait()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_wait(hmpp_handle_t * __h,int codelet_id)
{
  return CHECK_STATUS(cudaStreamSynchronize(__h->kernel_stream));
}

// ---------------------------------------------------------------------------
// hmpp_version()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_version()
{
#ifndef HMPP_RUNTIME_TARGET_VERSION
#define HMPP_RUNTIME_TARGET_VERSION(major,minor)((major << 16) | (minor << 8))
#endif
  return HMPP_RUNTIME_TARGET_VERSION(2,5);
}

//


