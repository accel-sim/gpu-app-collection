// ** Original codelet code **
//
// #pragma hmppcg cpiparam __arg0 IN fict%hmpp_codelet__runFdtd: (1, 0)
// #pragma hmppcg cpiparam __arg1 INOUT ex%hmpp_codelet__runFdtd: (1, 1)
// #pragma hmppcg cpiparam __arg2 INOUT ey%hmpp_codelet__runFdtd: (1, 2)
// #pragma hmppcg cpiparam __arg3 INOUT hz%hmpp_codelet__runFdtd: (1, 3)
// 
// #pragma hmppcg cpicall hmpp_codelet__runFdtd(__arg0, __arg1, __arg2, __arg3): 1
// 
// 
// /* begin of extracted source code for directive set "fdtd" */
// 
// 
// # 28 "fdtd2d.c"
// typedef float  DATA_TYPE;
// 
// 
// # 32 "fdtd2d.c"
// void hmpp_codelet__runFdtd(DATA_TYPE fict[500], DATA_TYPE ex[2048][2048 + 1], DATA_TYPE ey[2048 + 1][2048], DATA_TYPE hz[2048][2048])
// {
//   int  t, i, j;
// 
// #pragma hmppcg grid blocksize 32 X 8
// # 9 "<preprocessor>"
// # 39 "fdtd2d.c"
// #pragma hmppcg noParallel
// # 12 "<preprocessor>"
// # 40 "fdtd2d.c"
//   for (t = 0 ; t < 500 ; t++)
//     {
// #pragma hmppcg parallel
// # 17 "<preprocessor>"
// # 43 "fdtd2d.c"
//       for (j = 0 ; j < 2048 ; j++)
//         {
//           ey[0][j] = fict[t];
//         }
// 
// #pragma hmppcg parallel
// # 25 "<preprocessor>"
// # 49 "fdtd2d.c"
//       for (i = 1 ; i < 2048 ; i++)
//         {
// #pragma hmppcg parallel
// # 30 "<preprocessor>"
// # 52 "fdtd2d.c"
//           for (j = 0 ; j < 2048 ; j++)
//             {
//               ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
//             }
//         }
// 
// 
// #pragma hmppcg parallel
// # 40 "<preprocessor>"
// # 60 "fdtd2d.c"
//       for (i = 0 ; i < 2048 ; i++)
//         {
// #pragma hmppcg parallel
// # 45 "<preprocessor>"
// # 63 "fdtd2d.c"
//           for (j = 1 ; j < 2048 ; j++)
//             {
//               ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
//             }
//         }
// 
// 
// #pragma hmppcg parallel
// # 55 "<preprocessor>"
// # 71 "fdtd2d.c"
//       for (i = 0 ; i < 2048 ; i++)
//         {
// #pragma hmppcg parallel
// # 60 "<preprocessor>"
// # 74 "fdtd2d.c"
//           for (j = 0 ; j < 2048 ; j++)
//             {
//               hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
//             }
//         }
//     }
// }
// 
// 
// /* end of extracted source code for directive set "fdtd" */
// 
// 
//
// ** End of original codelet codelet **



#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <string>
#include <vector>

#include <CL/cl.h>

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

#include <string>
#include <map>
#include <vector>
#include <CL/cl.h>


namespace HMPPCG {

#define NO_KERNELS -2

class DataObject;
enum accessKind
{
  NONE = 0,
  READ_ONLY,
  WRITE_ONLY,
  READ_WRITE
};

void function_marker (){}

// ---------------------------------------------------------------------------
// User data
// ---------------------------------------------------------------------------
class UserData{
public:
  virtual ~UserData(){}
  UserData(){}
};

// ---------------------------------------------------------------------------
// CL context handle
// ---------------------------------------------------------------------------
class CLcontext {
  public :
    CLcontext();
    virtual ~CLcontext();

    DataObject* allocateData (size_t elem_size, size_t nb_elem, accessKind kind);
    void releaseData (DataObject* data);

    DataObject* allocateImage2DData (size_t elem_size, size_t size_1, size_t size2, accessKind kind);
    DataObject* allocateImage3DData (size_t elem_size, size_t size_1, size_t size2, size_t size3, accessKind kind);
    int init          (cl_context context, cl_device_id dev);
    int release       ();
    int wait          ();

    std::string device_name;
    std::vector<cl_kernel>  kernels;
    std::vector<std::string> prog_lines;
    std::vector<DataObject*> data_objects;
    cl_context context;
    cl_device_id device;
    cl_command_queue cmd_queue;
    cl_program program;
    std::string prog_name;
    std::map<std::string,UserData*> map_user_data;
    int readCLFile     (std::vector<std::string> &lines, const char* filename);
    int initContext    ();
    int releaseContext ();

    int          initProgram            ();
    cl_kernel    createKernelWithSource (const char *build_options, const std::string &full_prog_name, const std::string &kernel_name);
#ifndef _WIN32
    cl_kernel    createKernelWithBinary (const std::string &ffull_prog_name, const std::string &kernel_name);
#endif
    int          releaseProgram         ();
    bool         checkAllExts           (std::vector<std::string>);
    bool         checkOneOfThisExt      (std::vector<std::string>);
    bool         checkDoubleAvailable   ();
    bool         checkRestrictedDoubleAvailable();
    std::string  provideExtAvailable    ();

    size_t max_global_memory;
    size_t max_buffer_size;
    size_t max_const_memory;
    size_t max_const_buffer_size;
    std::string extractPath (std::string);

    std::vector <std::string> kernels_names;
};

// ---------------------------------------------------------------------------
// DataObject
// ---------------------------------------------------------------------------


class DataObject
{
  friend DataObject* CLcontext::allocateData(size_t elem_size, size_t nb_elem, accessKind kind);
  friend int CLcontext::releaseContext();
  public :
    enum SectionCopyKind {READ, WRITE };

    cl_mem_flags getCLFlags ();
    void init (size_t elem, size_t size, accessKind kind);
    size_t buf_size;
    size_t elem_size;
    cl_mem mem_obj;
    accessKind a_kind;
    cl_command_queue cmd;

    virtual int readData (void* data, bool async, cl_command_queue cmdq=NULL);
    virtual int readData (void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq=NULL);
    virtual int readRawData (void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq=NULL);

    virtual int writeData (const void* data, bool async, cl_command_queue=NULL);
    virtual int writeData (const void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq=NULL);
    virtual int writeRawData (const void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq=NULL);

    virtual int sectionCopy(void* data, const __hmppcg_DataSection* sections, const size_t *sizes, size_t elsize, int dim, bool async, SectionCopyKind kind, cl_command_queue cmdq=NULL);
    virtual int sectionCopyR(char *data, int offset, int cur, const __hmppcg_DataSection *sections, const size_t *sizes, size_t elsize, int dim, int lastdense, bool async, SectionCopyKind kind, cl_command_queue cmdq=NULL);

    virtual ~DataObject  () {release();}

    virtual int makeQueueWait(cl_command_queue wqueue, cl_command_queue cmdq=NULL);

    virtual int waitOnEvent(cl_event wevent, cl_command_queue cmdq=NULL);

  private :
    virtual int allocate (cl_context& context, cl_device_id& device);
    virtual int release  ();

  protected :
    DataObject   ();
};

class DataObjectDeleter
{
  DataObject * dat;
  CLcontext * context;

public:
  DataObject * getData() { return dat;}
  void setData(DataObject * dataobject) { dat = dataobject;}
  DataObjectDeleter(DataObject * dataobject, CLcontext *c)
    :dat(dataobject), context(c)
  {}

  ~DataObjectDeleter()
  {
    context->releaseData(dat);
  }
};


class Image2DDataObject : public DataObject {
  friend DataObject* CLcontext::allocateImage2DData(size_t elem_size, size_t nb_elem_dim1, size_t nb_elem_dim2, accessKind kind);
  public :
    void init (size_t elem, size_t nb_elem_dim1, size_t nb_elem_dim2, accessKind kind);
    size_t buf_size_dim1;
    size_t buf_size_dim2;
    size_t img_width;
    size_t nb_elem_per_pixel;


    virtual int readData (void* data, bool async, cl_command_queue cmdq=NULL);
    virtual int readData (void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq=NULL);
    virtual int readRawData (void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq=NULL);

    virtual int writeData (const void* data, bool async, cl_command_queue=NULL);
    virtual int writeData (const void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq=NULL);
    virtual int writeRawData (const void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq=NULL);

    virtual int sectionCopy(void* data, const __hmppcg_DataSection* sections, const size_t *sizes, size_t elsize, int dim, bool async, SectionCopyKind kind, cl_command_queue cmdq=NULL);
    virtual int sectionCopyR(char *data, int offset, int cur, const __hmppcg_DataSection *sections, const size_t *sizes, size_t elsize, int dim, int lastdense, bool async, SectionCopyKind kind, cl_command_queue cmdq=NULL);

  private :
    virtual int allocate (cl_context& context, cl_device_id& device);
    Image2DDataObject   ();
    virtual ~Image2DDataObject  () {}
};

class Image3DDataObject : public DataObject {
  friend DataObject* CLcontext::allocateImage3DData(size_t elem_size, size_t nb_elem_dim1, size_t nb_elem_dim2, size_t nb_elem_dim3, accessKind kind);
  public :
    void init (size_t elem, size_t nb_elem_dim1, size_t nb_elem_dim2, size_t nb_elem_dim3, accessKind kind);
    size_t buf_size_dim1;
    size_t buf_size_dim2;
    size_t buf_size_dim3;
    size_t img_width;
    size_t nb_elem_per_pixel;

    virtual int readData (void* data, bool async, cl_command_queue cmdq=NULL);
    virtual int readData (void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq=NULL);
    virtual int readRawData (void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq=NULL);

    virtual int writeData (const void* data, bool async, cl_command_queue=NULL);
    virtual int writeData (const void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq=NULL);
    virtual int writeRawData (const void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq=NULL);

    virtual int sectionCopy(void* data, const __hmppcg_DataSection* sections, const size_t *sizes, size_t elsize, int dim, bool async, SectionCopyKind kind, cl_command_queue cmdq=NULL);
    virtual int sectionCopyR(char *data, int offset, int cur, const __hmppcg_DataSection *sections, const size_t *sizes, size_t elsize, int dim, int lastdense, bool async, SectionCopyKind kind, cl_command_queue cmdq=NULL);

  private :
    virtual int allocate (cl_context& context, cl_device_id& device);
    Image3DDataObject   ();
    virtual ~Image3DDataObject  () {}
};

// ---------------------------------------------------------------------------
// Allocatable Arrays
// ---------------------------------------------------------------------------
#ifndef __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATE
#define __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATE( var, type, nb_dims, ... )             \
        { int hmppcg_alloc_ranges[] = { __VA_ARGS__ };                             \
          int hmppcg_alloc_i;                                                      \
          var.desc_.wholesize_ = 1;                                                \
          for(hmppcg_alloc_i=0; hmppcg_alloc_i<nb_dims; hmppcg_alloc_i++){         \
            int hmppcg_alloc_first = hmppcg_alloc_ranges[2*hmppcg_alloc_i];        \
            int hmppcg_alloc_last  = hmppcg_alloc_ranges[2*hmppcg_alloc_i + 1];    \
            int hmppcg_alloc_size  = hmppcg_alloc_last - hmppcg_alloc_first + 1;   \
            var.desc_.lbounds_[hmppcg_alloc_i] = hmppcg_alloc_first;               \
            var.desc_.sizes_[hmppcg_alloc_i]   = hmppcg_alloc_size;                \
            var.desc_.wholesize_ *= hmppcg_alloc_size;                             \
          }                                                                        \
          var.data_object_ = context.allocateData(sizeof( type ), sizeof( type ) * var.desc_.wholesize_, HMPPCG::READ_WRITE); \
          if(! var.data_object_ ) abort();                                         \
          var.desc_data_object_ = context.allocateData(sizeof( var.desc_ ), sizeof( var.desc_ ), HMPPCG::READ_WRITE); \
          if(! var.desc_data_object_ ) abort();                                    \
          var.ptr_ = ( type * )malloc(var.desc_.wholesize_ * sizeof (type));       \
          if(! var.ptr_ ) abort();                                                 \
          var.desc_data_object_->writeData(& var.desc_, false, context.cmd_queue); \
        }
#endif //__HMPPCG_ALLOCATABLE_ARRAY_ALLOCATE


#ifndef __HMPPCG_ALLOCATABLE_ARRAY_DEALLOCATE
#define __HMPPCG_ALLOCATABLE_ARRAY_DEALLOCATE( var )    \
        {                                               \
          if(var.ptr_)                                  \
          {                                             \
            free(var.ptr_);                             \
            var.ptr_ = NULL;                            \
            context.releaseData(var.data_object_);      \
            var.data_object_ = NULL;                    \
            context.releaseData(var.desc_data_object_); \
            var.desc_data_object_ = NULL;               \
          }                                             \
       }
#endif //__HMPPCG_ALLOCATABLE_ARRAY_DEALLOCATE


#ifndef HMPPCG_ALLOC_ARRAY_DESC
#define HMPPCG_ALLOC_ARRAY_DESC( size )   \
        struct {                          \
          int  lbounds_[ size ];          \
          unsigned int sizes_[ size ];    \
          unsigned int wholesize_;        \
        } desc_;                          \
        HMPPCG::DataObject *data_object_; \
        HMPPCG::DataObject *desc_data_object_;
#endif //HMPPCG_ALLOC_ARRAY_DESC


#ifndef __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATED
#define __HMPPCG_ALLOCATABLE_ARRAY_ALLOCATED( var ) \
        (var.ptr_ != NULL)
#endif //__HMPPCG_ALLOCATABLE_ARRAY_ALLOCATED

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE
#define __HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE( var ) \
        var.desc_.wholesize_
#endif //__HMPPCG_ALLOCATABLE_ARRAY_WHOLESIZE

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_SIZE
#define __HMPPCG_ALLOCATABLE_ARRAY_SIZE( var, d ) \
        var.desc_.sizes_[d]
#endif //__HMPPCG_ALLOCATABLE_ARRAY_SIZE

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_LBOUND
#define __HMPPCG_ALLOCATABLE_ARRAY_LBOUND( var, d ) \
        var.desc_.lbounds_[d]
#endif //__HMPPCG_ALLOCATABLE_ARRAY_LBOUND

#ifndef __HMPPCG_ALLOCATABLE_ARRAY_UBOUND
#define __HMPPCG_ALLOCATABLE_ARRAY_UBOUND( var, d ) \
        (var.desc_.sizes_[d] + var.desc_.lbounds_[d] - 1)
#endif //__HMPPCG_ALLOCATABLE_ARRAY_UBOUND

}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <algorithm>

#include <fstream>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  include <dlfcn.h>
#else
#include <windows.h>
#endif

#include <CL/cl.h>


inline int check_status(const char* file, int line, cl_int status)
{
  if(status != CL_SUCCESS)
  {
    std::string err_msg = "";
    switch (status)
    {
        case CL_DEVICE_NOT_FOUND:
            err_msg = "CL_DEVICE_NOT_FOUND";
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            err_msg = "CL_DEVICE_NOT_AVAILABLE";
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            err_msg = "CL_COMPILER_NOT_AVAILABLE";
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            err_msg = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            break;
        case CL_OUT_OF_RESOURCES:
            err_msg = "CL_OUT_OF_RESOURCES";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            err_msg = "CL_OUT_OF_HOST_MEMORY";
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            err_msg = "CL_PROFILING_INFO_NOT_AVAILABLE";
            break;
        case CL_MEM_COPY_OVERLAP:
            err_msg = "CL_MEM_COPY_OVERLAP";
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            err_msg = "CL_IMAGE_FORMAT_MISMATCH";
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            err_msg = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            err_msg = "CL_BUILD_PROGRAM_FAILURE";
            break;
        case CL_MAP_FAILURE:
            err_msg = "CL_MAP_FAILURE";
            break;
        case CL_INVALID_VALUE:
            err_msg = "CL_INVALID_VALUE";
            break;
        case CL_INVALID_DEVICE_TYPE:
            err_msg = "CL_INVALID_DEVICE_TYPE";
            break;
        case CL_INVALID_PLATFORM:
            err_msg = "CL_INVALID_PLATFORM";
            break;
        case CL_INVALID_DEVICE:
            err_msg = "CL_INVALID_DEVICE";
            break;
        case CL_INVALID_CONTEXT:
            err_msg = "CL_INVALID_CONTEXT";
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            err_msg = "CL_INVALID_QUEUE_PROPERTIES";
            break;
        case CL_INVALID_COMMAND_QUEUE:
            err_msg = "CL_INVALID_COMMAND_QUEUE";
            break;
        case CL_INVALID_HOST_PTR:
            err_msg = "CL_INVALID_HOST_PTR";
            break;
        case CL_INVALID_MEM_OBJECT:
            err_msg = "CL_INVALID_MEM_OBJECT";
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            err_msg = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            break;
        case CL_INVALID_IMAGE_SIZE:
             err_msg = "CL_INVALID_IMAGE_SIZE";
            break;
        case CL_INVALID_SAMPLER:
            err_msg = "CL_INVALID_SAMPLER";
            break;
        case CL_INVALID_BINARY:
            err_msg = "CL_INVALID_BINARY";
            break;
        case CL_INVALID_BUILD_OPTIONS:
            err_msg = "CL_INVALID_BUILD_OPTIONS";
            break;
        case CL_INVALID_PROGRAM:
            err_msg = "CL_INVALID_PROGRAM";
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            err_msg = "CL_INVALID_PROGRAM_EXECUTABLE";
            break;
        case CL_INVALID_KERNEL_NAME:
            err_msg = "CL_INVALID_KERNEL_NAME";
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            err_msg = "CL_INVALID_KERNEL_DEFINITION";
            break;
        case CL_INVALID_KERNEL:
            err_msg = "CL_INVALID_KERNEL";
            break;
        case CL_INVALID_ARG_INDEX:
            err_msg = "CL_INVALID_ARG_INDEX";
            break;
        case CL_INVALID_ARG_VALUE:
            err_msg = "CL_INVALID_ARG_VALUE";
            break;
        case CL_INVALID_ARG_SIZE:
            err_msg = "CL_INVALID_ARG_SIZE";
            break;
        case CL_INVALID_KERNEL_ARGS:
            err_msg = "CL_INVALID_KERNEL_ARGS";
            break;
        case CL_INVALID_WORK_DIMENSION:
            err_msg = "CL_INVALID_WORK_DIMENSION";
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            err_msg = "CL_INVALID_WORK_GROUP_SIZE";
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            err_msg = "CL_INVALID_WORK_ITEM_SIZE";
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            err_msg = "CL_INVALID_GLOBAL_OFFSET";
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            err_msg = "CL_INVALID_EVENT_WAIT_LIST";
            break;
        case CL_INVALID_EVENT:
            err_msg = "CL_INVALID_EVENT";
            break;
        case CL_INVALID_OPERATION:
            err_msg = "CL_INVALID_OPERATION";
            break;
        case CL_INVALID_GL_OBJECT:
            err_msg = "CL_INVALID_GL_OBJECT";
            break;
        case CL_INVALID_BUFFER_SIZE:
            err_msg = "CL_INVALID_BUFFER_SIZE";
            break;
        case CL_INVALID_MIP_LEVEL:
            err_msg = "CL_INVALID_MIP_LEVEL";
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            err_msg = "CL_INVALID_GLOBAL_WORK_SIZE";
            break;
        default :
        err_msg = "Unknown error";
    }
    fprintf(stderr, "%s:%d CL Error %s\n", file, line, err_msg.c_str());
    return -1;
  }
  return 0;
}
// CHECK_STATUS
#define CHECK_STATUS(X) check_status(__FILE__,__LINE__, (X))
namespace HMPPCG {
// ---------------------------------------------------------------------------
// DataObject
// ---------------------------------------------------------------------------


DataObject::DataObject ()
{
  mem_obj = NULL;
  buf_size = 0;
  elem_size = 0;
  a_kind = NONE;
}
void DataObject::init (size_t elem, size_t size, accessKind kind)
{
  buf_size = size;
  elem_size = elem;
  a_kind = kind;
}

cl_mem_flags DataObject::getCLFlags ()
{
  cl_mem_flags flags = 0x0;

  switch (a_kind)
  {
    case READ_ONLY :
      flags = flags | CL_MEM_READ_ONLY;
      break;
    case WRITE_ONLY :
      flags = flags | CL_MEM_WRITE_ONLY;
      break;
    case READ_WRITE :
      flags = flags | CL_MEM_READ_WRITE;
      break;
    default :
      break;
  }
  return flags;
}

int DataObject::allocate(cl_context& context, cl_device_id& device)
{
  cl_int err;
  mem_obj = clCreateBuffer(context, getCLFlags(), buf_size, NULL, &err);
  int my_err = CHECK_STATUS(err);
  if (my_err)
    return my_err;
  cmd = clCreateCommandQueue(context, device, 0, &err);
  return CHECK_STATUS(err);
}

int DataObject::release ()
{
  int my_err = 0;
  if(mem_obj)
  {
    cl_int err = clReleaseMemObject(mem_obj);
    mem_obj = 0;
    my_err = CHECK_STATUS(err);
    if (my_err)
      return my_err;
  }
  if(cmd)
  {
    cl_int err = clReleaseCommandQueue(cmd);
    my_err = CHECK_STATUS(err);
    cmd = 0;
  }
  return my_err;
}

int DataObject::readData(void* data, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool my_async;
  if (async)
    my_async = CL_FALSE;
  else
    my_async = CL_TRUE;
  status = clEnqueueReadBuffer(cmdq, mem_obj, my_async, 0, buf_size, (void*) data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int DataObject::readRawData(void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool my_async;
  if (async)
    my_async = CL_FALSE;
  else
    my_async = CL_TRUE;
  status = clEnqueueReadBuffer(cmdq, mem_obj, my_async, offset, total, (void*) data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int DataObject::readData(void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool my_async;
  if (async)
    my_async = CL_FALSE;
  else
    my_async = CL_TRUE;
  status = clEnqueueReadBuffer(cmdq, mem_obj, my_async, offset * elem_size, nb_elems * elem_size, (void*) data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}


int DataObject::writeData(const void* data, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool my_async;
  if (async)
    my_async = CL_FALSE;
  else
    my_async = CL_TRUE;
  status = clEnqueueWriteBuffer(cmdq, mem_obj, my_async, 0, buf_size, (void*) data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int DataObject::writeData(const void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool my_async;
  if (async)
    my_async = CL_FALSE;
  else
    my_async = CL_TRUE;
  status = clEnqueueWriteBuffer(cmdq, mem_obj, my_async, offset * elem_size, nb_elems * elem_size, (void*) data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int DataObject::writeRawData(const void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool my_async;
  if (async)
    my_async = CL_FALSE;
  else
    my_async = CL_TRUE;
  status = clEnqueueWriteBuffer(cmdq, mem_obj, my_async, offset, total, (void*) data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}


int DataObject::sectionCopy (void* data, const __hmppcg_DataSection* sections, const size_t *sizes, size_t elsize, int dim, bool async, SectionCopyKind kind, cl_command_queue cmdq)
{
  if (! cmdq) cmdq = cmd;
  int i;
  int lastdense = dim;
  for (i = dim - 1 ; i >= 0 ; i --)
  {
    if ((sections[i].from == 0) && (sections[i].to == sizes[i] - 1) && (sections[i].step == 1))
      lastdense = i;
    else
      break;
  }
  return sectionCopyR((char*) data, 0, 0, sections, sizes, elsize, dim, lastdense, async, kind, cmdq);
}

int DataObject::sectionCopyR (char* data, int offset, int cur, const __hmppcg_DataSection *sections, const size_t *sizes, size_t elsize, int dim, int lastdense, bool async, SectionCopyKind kind, cl_command_queue cmdq)
{
  if (! cmdq) cmdq = cmd;
  int return_value = 0;
  int d;
  int size = 1;
  for(d=cur+1;d<dim;d++)
    size *= sizes[d];

  if(cur<(lastdense-1))
  {
    size_t x;
    for(x=sections[cur].from;x<=sections[cur].to;x+=sections[cur].step)
    {
      return_value = sectionCopyR(data,offset+x*size,cur+1,sections,sizes,elsize,dim,lastdense, async, kind, cmdq);
      if (return_value)
        return return_value;
    }
  }
  else
  {
    int step = sections[cur].step;
    if(step == 1)
    {
      int start = (offset + sections[cur].from * size) * elsize;
      int total = (sections[cur].to - sections[cur].from + 1) * size * elsize;
      switch (kind)
      {
        case READ :
          return_value = readRawData ( data+start, start, total, async, cmdq);
//           memcpy(dst+start,src+start,total);
          if (return_value)
            return return_value;
          break;
        case WRITE :
          return_value = writeRawData ( data+start, start, total, async, cmdq);
          if (return_value)
            return return_value;
//             memcpy(dst+start,src+start,total);
          break;
      }
    }
    else
    {
      size_t x;
      for(x=sections[cur].from;x<=sections[cur].to;x+=step)
      {
        int off = (offset + x * size) *elsize;
      switch (kind)
      {
        case READ :
          return_value = readRawData( data+off, off, size*elsize, async, cmdq);
          if (return_value)
            return return_value;
//         memcpy(dst+off,src+off,size * elsize);
          break;
        case WRITE :
          return_value = writeRawData( data+off, off, size*elsize, async, cmdq);
          if (return_value)
            return return_value;
//           memcpy(dst+off,src+off,size * elsize);
          break;
      }
      }
    }
  }
  return return_value;
}

int DataObject::makeQueueWait(cl_command_queue wqueue, cl_command_queue cmdq)
{
  cl_int status;
  cl_event event;
  if (! cmdq) cmdq = cmd;

  status = clEnqueueMarker(cmdq, &event);
  if (status != 0)
    return CHECK_STATUS(status);
  status = clFlush(cmdq);
  if (status != 0)
    return CHECK_STATUS(status);
  status = clEnqueueWaitForEvents(wqueue, 1, &event);
  if (status != 0)
    return CHECK_STATUS(status);

  return CHECK_STATUS(clReleaseEvent(event));
}

int DataObject::waitOnEvent(cl_event wevent, cl_command_queue cmdq) {
  cl_int status;
  if (! cmdq) cmdq = cmd;
  status = clEnqueueWaitForEvents(cmdq, 1, &wevent);
  return CHECK_STATUS(status);
}

// ---------------------------------------------------------------------------
// Image2DDataObject
// ---------------------------------------------------------------------------

Image2DDataObject::Image2DDataObject() : DataObject() {
  buf_size_dim1 = 0;
  buf_size_dim2 = 0;
  img_width = 0;
}

void Image2DDataObject::init(size_t elem, size_t nb_elem_dim1, size_t nb_elem_dim2, accessKind kind)
{
  size_t total = nb_elem_dim1 * nb_elem_dim2;
  DataObject::init (elem, total, kind);
  buf_size_dim1 = nb_elem_dim1;
  buf_size_dim2 = nb_elem_dim2;
  nb_elem_per_pixel = (16 / elem);
  img_width = buf_size_dim1 / nb_elem_per_pixel;
}

int Image2DDataObject::readData(void* data, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool blocking;
  if (async)
    blocking = CL_FALSE;
  else
    blocking = CL_TRUE;
  size_t origin[3] = {0,0,0};
  size_t region[3] = {img_width, buf_size_dim2, 1};
  status = clEnqueueReadImage(cmdq, mem_obj, blocking, origin, region, nb_elem_per_pixel*img_width * elem_size, 0, (void*)data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int Image2DDataObject::readRawData(void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image2DDataObject::readData(void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image2DDataObject::writeData(const void* data, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool blocking;
  if (async)
    blocking = CL_FALSE;
  else
    blocking = CL_TRUE;
  size_t origin[3] = {0,0,0};
  size_t region[3] = {img_width, buf_size_dim2, 1};
  status = clEnqueueWriteImage(cmdq, mem_obj, blocking, origin, region, nb_elem_per_pixel*img_width * elem_size, 0, (void*)data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int Image2DDataObject::writeData(const void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image2DDataObject::writeRawData(const void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image2DDataObject::sectionCopy(void* data, const __hmppcg_DataSection* sections, const size_t *sizes, size_t elsize, int dim, bool async, SectionCopyKind kind, cl_command_queue cmdq) {
  return 1;
}

int Image2DDataObject::sectionCopyR(char *data, int offset, int cur, const __hmppcg_DataSection *sections, const size_t *sizes, size_t elsize, int dim, int lastdense, bool async, SectionCopyKind kind, cl_command_queue cmdq) {
  return 1;
}

int Image2DDataObject::allocate(cl_context& context, cl_device_id& device)
{
  cl_int err;
  cl_image_format img_format;
  img_format.image_channel_order = CL_RGBA;
  img_format.image_channel_data_type = CL_FLOAT;

  size_t image2d_max_width;
  clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &image2d_max_width, NULL);

  if(buf_size_dim1%nb_elem_per_pixel != 0){
    fprintf(stderr, "%s:%d Allocation error : dim[0] must be a multiple of %lu\n", __FILE__, __LINE__, (unsigned long)nb_elem_per_pixel);
    return -1;
  }
  if( img_width > image2d_max_width ) {
    if(buf_size_dim2 == 1) {
      //should try to find the highest divisor
      if((img_width % image2d_max_width) != 0) {
        fprintf(stderr, "%s:%d Allocation error : dim[0] must be a multiple of %lux%lu if bigger.\n", __FILE__, __LINE__, (unsigned long)nb_elem_per_pixel, (unsigned long)image2d_max_width);
        return -1;
      }
      buf_size_dim2 = img_width / image2d_max_width;
      img_width = image2d_max_width;
    } /*else ... no need, handled by cl create img*/
  }

  fprintf(stderr, "DBG /!\\ %s:%d trying allocation of image %lu x %lu\n", __FILE__, __LINE__, (unsigned long)img_width, (unsigned long)buf_size_dim2);
  mem_obj = clCreateImage2D(context, getCLFlags(), &img_format, img_width, buf_size_dim2, 0, NULL, &err);
  int error_code = CHECK_STATUS(err);
  if (error_code)
    return error_code;
  size_t width, height;
  CHECK_STATUS(clGetImageInfo(mem_obj, CL_IMAGE_WIDTH, sizeof(size_t), &width, NULL));
  CHECK_STATUS(clGetImageInfo(mem_obj, CL_IMAGE_HEIGHT, sizeof(size_t), &height, NULL));
  fprintf(stderr, "DBG /!\\ %s:%d Allocated %lu x %lu\n", __FILE__, __LINE__, (unsigned long)width, (unsigned long)height);
  cmd = clCreateCommandQueue(context, device, 0, &err);
  return CHECK_STATUS(err);
}

// ---------------------------------------------------------------------------
// Image3DDataObject
// ---------------------------------------------------------------------------

Image3DDataObject::Image3DDataObject() : DataObject() {
  buf_size_dim1 = 0;
  buf_size_dim2 = 0;
  buf_size_dim3 = 0;
  img_width = 0;
}

void Image3DDataObject::init(size_t elem, size_t nb_elem_dim1, size_t nb_elem_dim2, size_t nb_elem_dim3, accessKind kind)
{
  size_t total = nb_elem_dim1 * nb_elem_dim2;
  DataObject::init (elem, total, kind);
  buf_size_dim1 = nb_elem_dim1;
  buf_size_dim2 = nb_elem_dim2;
  buf_size_dim3 = nb_elem_dim3;
  nb_elem_per_pixel = (16 / elem);
  img_width = buf_size_dim1 / nb_elem_per_pixel;
}

int Image3DDataObject::readData(void* data, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool blocking;
  if (async)
    blocking = CL_FALSE;
  else
    blocking = CL_TRUE;
  size_t origin[3] = {0,0,0};
  size_t region[3] = {img_width, buf_size_dim2, buf_size_dim3};
  status = clEnqueueReadImage(cmdq, mem_obj, blocking, origin, region, nb_elem_per_pixel*img_width * elem_size, 0, (void*)data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int Image3DDataObject::readRawData(void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image3DDataObject::readData(void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image3DDataObject::writeData(const void* data, bool async, cl_command_queue cmdq)
{
  int status;
  if (! cmdq) cmdq = cmd;
  cl_bool blocking;
  if (async)
    blocking = CL_FALSE;
  else
    blocking = CL_TRUE;
  size_t origin[3] = {0,0,0};
  size_t region[3] = {img_width, buf_size_dim2, buf_size_dim3};
  status = clEnqueueWriteImage(cmdq, mem_obj, blocking, origin, region, nb_elem_per_pixel*img_width * elem_size, 0, (void*)data, 0, NULL, NULL);
  if(!status && async)
    status = clFlush(cmdq);
  return CHECK_STATUS(status);
}

int Image3DDataObject::writeData(const void* data, size_t offset, size_t nb_elems, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image3DDataObject::writeRawData(const void* data, size_t offset, size_t total, bool async, cl_command_queue cmdq)
{
  return 1;
}

int Image3DDataObject::sectionCopy(void* data, const __hmppcg_DataSection* sections, const size_t *sizes, size_t elsize, int dim, bool async, SectionCopyKind kind, cl_command_queue cmdq) {
  return 1;
}

int Image3DDataObject::sectionCopyR(char *data, int offset, int cur, const __hmppcg_DataSection *sections, const size_t *sizes, size_t elsize, int dim, int lastdense, bool async, SectionCopyKind kind, cl_command_queue cmdq) {
  return 1;
}

int Image3DDataObject::allocate(cl_context& context, cl_device_id& device)
{
  cl_int err;
  cl_image_format img_format;
  img_format.image_channel_order = CL_RGBA;
  img_format.image_channel_data_type = CL_FLOAT;

  fprintf(stderr, "DBG /!\\ %s:%d trying allocation of image %lu x %lu x %lu\n", __FILE__, __LINE__, (unsigned long)img_width, (unsigned long)buf_size_dim2, (unsigned long)buf_size_dim3);
  mem_obj = clCreateImage3D(context, getCLFlags(), &img_format, img_width, buf_size_dim2, buf_size_dim3, 0, 0, NULL, &err);
  int error_code = CHECK_STATUS(err);
  if (error_code)
    return error_code;
  size_t width, height, depth;
  CHECK_STATUS(clGetImageInfo(mem_obj, CL_IMAGE_WIDTH, sizeof(size_t), &width, NULL));
  CHECK_STATUS(clGetImageInfo(mem_obj, CL_IMAGE_HEIGHT, sizeof(size_t), &height, NULL));
  CHECK_STATUS(clGetImageInfo(mem_obj, CL_IMAGE_DEPTH, sizeof(size_t), &depth, NULL));
  fprintf(stderr, "DBG /!\\ %s:%d Allocated %lu x %lu x %lu\n", __FILE__, __LINE__, (unsigned long)width, (unsigned long)height, (unsigned long)depth);
  cmd = clCreateCommandQueue(context, device, 0, &err);
  return CHECK_STATUS(err);
}

// ---------------------------------------------------------------------------
// CL context handle
// ---------------------------------------------------------------------------

CLcontext::CLcontext ()
{
  program = NULL;
  cmd_queue = NULL;
  device = NULL;
}

CLcontext::~CLcontext ()
{
  for (std::map<std::string,UserData*>::const_iterator it = map_user_data.begin(); it != map_user_data.end(); ++it)
    {
      delete it->second;
    }
}

int CLcontext::readCLFile (std::vector<std::string>& lines, const char* filename)
{
 std::ifstream f;
  f.open(filename);
  if (f.fail())
  {
    fprintf(stderr, "%s:%d File error : unable to open file %s\n", __FILE__, __LINE__, filename);
    return -1;
  }
  std::string line;
  while (std::getline(f, line))
  {
    lines.push_back(line + "\n");
  }
  f.close();
return 0;
}

int CLcontext::init (cl_context ctx, cl_device_id dev)
{
  int err = 0;
  context = ctx;
  device = dev;

  err = initContext();
  return err;

}

DataObject* CLcontext::allocateData (size_t elem_size, size_t nb_elem, accessKind kind)
{
  DataObject* obj = new DataObject();
  obj->init (elem_size, nb_elem, kind);
  int err = obj->allocate(context, device);
  if (err)
  {
    delete(obj);
    return NULL;
  }
  data_objects.push_back(obj);
  return obj;
}

void CLcontext::releaseData (DataObject* data)
{
  delete(data);
  for(std::vector<DataObject *>::iterator iter = data_objects.begin();
      iter != data_objects.end();
      iter++)
  {
    if((*iter) == data)
    {
      data_objects.erase(iter);
      break;
    }
  }
}

DataObject* CLcontext::allocateImage2DData (size_t elem_size, size_t nb_elem_dim1, size_t nb_elem_dim2, accessKind kind)
{
  Image2DDataObject* obj = new Image2DDataObject();
  obj->init (elem_size, nb_elem_dim1, nb_elem_dim2, kind);
  int err = obj->allocate(context, device);
  if (err)
  {
    delete(obj);
    return NULL;
  }
  data_objects.push_back(obj);
  return obj;
}

DataObject* CLcontext::allocateImage3DData (size_t elem_size, size_t nb_elem_dim1, size_t nb_elem_dim2, size_t nb_elem_dim3, accessKind kind)
{
  Image3DDataObject* obj = new Image3DDataObject();
  obj->init (elem_size, nb_elem_dim1, nb_elem_dim2, nb_elem_dim3, kind);
  int err = obj->allocate(context, device);
  if (err)
  {
    delete(obj);
    return NULL;
  }
  data_objects.push_back(obj);
  return obj;
}

int CLcontext::release ()
{
  int err = 0;
  err = releaseProgram ();
  if (err) return err;
  err = releaseContext ();
  return err;
}

int CLcontext::initContext ()
{
  char* buffer;
  size_t size_buf;
  if (CHECK_STATUS( clGetDeviceInfo (device, CL_DEVICE_NAME, 0, NULL, &size_buf) ) ) {
    return -1;
  }
  if (size_buf <= 0)
  {
    fprintf(stderr, "No name found for this device\n");
    device_name = "";
  }
  else
  {
    buffer = (char*) malloc (size_buf);
    if (CHECK_STATUS( clGetDeviceInfo(device, CL_DEVICE_NAME, size_buf, buffer, NULL) ) ) {
      return -1;
    }
    device_name = std::string(buffer);
    free(buffer);
  }

  cl_int err;
  cmd_queue = clCreateCommandQueue(context, device, 0, &err);
  if (CHECK_STATUS(err))
    return -1;
  return 0;
}

int CLcontext::releaseContext ()
{
  int err = 0;
  if (cmd_queue)
    err =  clReleaseCommandQueue(cmd_queue);
  if (err) return err;
  for (unsigned int i = 0; i < data_objects.size(); i++)
  {
    if(data_objects[i])
    {
      err = data_objects[i]->release();
      if (err) return err;
      delete data_objects[i];
    }
  }
  data_objects.clear();
  return err;
}

std::string CLcontext::extractPath (std::string s)
{
  int pos = s.find_last_of("/\\");
  return s.substr(0, pos+1);
}

int CLcontext::initProgram ()
{
  cl_kernel kern;
  std::string full_prog_name;
  std::vector <std::string> global_atomic_exts, local_atomic_exts, khr_byte_addressable_store_exts;
  std::string build_options = "";
  char *env_options = getenv("HMPP_OPENCL_FLAGS");
  if( env_options )
    build_options += env_options;

  global_atomic_exts.push_back("cl_khr_global_int32_base_atomics");
  global_atomic_exts.push_back("cl_khr_global_int32_extended_atomics");
  if( checkAllExts(global_atomic_exts) ) {
    build_options += " -DGLOBAL_ATOMIC_EXTS_SUPPORTED";
  }

  local_atomic_exts.push_back("cl_khr_local_int32_base_atomics");
  local_atomic_exts.push_back("cl_khr_local_int32_extended_atomics");
  if( checkAllExts(local_atomic_exts) ) {
    build_options += " -DLOCAL_ATOMIC_EXTS_SUPPORTED";
  }

  khr_byte_addressable_store_exts.push_back("cl_khr_byte_addressable_store");
  if( checkAllExts(khr_byte_addressable_store_exts) ) {
    build_options += " -DBYTE_ADDRESSABLE_STORE_EXTS_SUPPORTED";
  }

  full_prog_name = prog_name;

#ifndef _WIN32
  Dl_info info;
  if ( dladdr( (void *) &function_marker, &info) ) {
    full_prog_name = std::string(info.dli_fname);
    full_prog_name = extractPath(full_prog_name)+prog_name;
  }
#else
  char szFileName[MAX_PATH];
  std::string module_name = prog_name.substr(0,prog_name.find_first_of("."));
  HINSTANCE hInstance = GetModuleHandle(prog_name.c_str());
  GetModuleFileName(hInstance, szFileName, MAX_PATH);
  full_prog_name = extractPath(std::string(szFileName))+prog_name;
#endif

  for (unsigned int i = 0; i < kernels_names.size();i++)
    {
      kern = createKernelWithSource(build_options.c_str(),full_prog_name,kernels_names[i]);
      if ( kern == NULL )
        return -1;
      else
        kernels.push_back (kern);
    }
  return 0;
}

int CLcontext::releaseProgram()
{
  for (unsigned int i = 0 ; i < kernels.size(); i++)
  {
    clReleaseKernel  (kernels[i]);
  }
  kernels.clear();
  if (program)
    clReleaseProgram (program);
  return 0;
}

int CLcontext::wait ()
{
  clFinish(cmd_queue);
  return 0;
}


bool CLcontext::checkAllExts (std::vector<std::string> ext_list)
{
  std::string exts_str = provideExtAvailable();
  for (unsigned int i = 0; i < ext_list.size(); i++)
  {
    if (exts_str.find(ext_list[i]) == std::string::npos)
      return false;
  }
  return true;
}

bool CLcontext::checkOneOfThisExt (std::vector<std::string> ext_list)
{
  std::string exts_str = provideExtAvailable();
  for (unsigned int i = 0; i < ext_list.size(); i++)
  {
    if (exts_str.find(ext_list[i]) != std::string::npos)
      return true;
  }
  return false;
}

bool CLcontext::checkDoubleAvailable ()
{
  std::vector <std::string> exts;
  exts.push_back ("cl_khr_fp64");
  return checkOneOfThisExt (exts);
}

std::string CLcontext::provideExtAvailable ()
{
  char* buffer;
  size_t size_buf;
  if (CHECK_STATUS( clGetDeviceInfo (device, CL_DEVICE_EXTENSIONS, 0, NULL, &size_buf) ) ) {
    return "";
  }
  if (size_buf <= 0)
  {
    fprintf(stderr, "CL Warning : No extensions was found on this device\n");
    return "";
  }
  buffer = (char*) malloc (size_buf);
  if (CHECK_STATUS( clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size_buf, buffer, NULL) ) ) {
    return "";
  }
  std::string ret_str = std::string(buffer);
  free(buffer);
  return ret_str;
}

cl_kernel CLcontext::createKernelWithSource (const char *build_options, const std::string &full_prog_name, const std::string &kernel_name)
{
  cl_int err;
  char** program_src;
  std::vector<std::string> prog_lines;
  cl_kernel kernel;
  int status;

  readCLFile (prog_lines, full_prog_name.c_str());
  program_src = (char**) malloc (sizeof(char*) * prog_lines.size());
  for (unsigned int i = 0; i < prog_lines.size(); i++)
  {
    program_src[i] = (char*) prog_lines[i].c_str();
  }
  program = clCreateProgramWithSource(context, prog_lines.size(), (const char**) program_src, NULL, &err);
  free(program_src);
  if (CHECK_STATUS(err))
    return NULL;

  status = CHECK_STATUS(clBuildProgram(program, 1, &device, build_options, NULL, NULL));
  if( (status != 0) || (getenv("HMPP_OPENCL_DISPLAY_BUILD_INFO") != NULL) )
  {
    size_t len;
    char buffer[4096];
    clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    fprintf(stderr, "%s\n", buffer);
  }
  if( status != 0 )
    return NULL;

  kernel = clCreateKernel (program,kernel_name.c_str(), &err);
  if(CHECK_STATUS(err))
    return NULL;
  return kernel;
}

#ifndef _WIN32
cl_kernel CLcontext::createKernelWithBinary (const std::string &full_prog_name, const std::string &kernel_name)
{
  cl_int err;
  cl_kernel kernel;
  struct stat s;
  int fd = open(full_prog_name.c_str(), O_RDONLY);
  if(fd == -1) err = CL_INVALID_BINARY;
  if (CHECK_STATUS(err))
    return NULL;

  cl_int status;
  fstat(fd, &s);
  size_t length = s.st_size;
  void* map = mmap(NULL, s.st_size, PROT_EXEC | PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) err = CL_INVALID_BINARY;
  if (CHECK_STATUS(err))
    return NULL;

  program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char**)&map, &status, &err);
  if (CHECK_STATUS(err))
    return NULL;
  if(CHECK_STATUS(clBuildProgram(program, 1, &device, NULL, NULL, NULL)))
  {
    size_t len;
    char buffer[4096];
    clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    fprintf(stderr, "%s\n", buffer);
    return NULL;
  }



  kernel = clCreateKernel (program,kernel_name.c_str(), &err);
  if (CHECK_STATUS(err))
    return NULL;
  return kernel;
}
#endif

}

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

void hmpp_codelet__runFdtd(
  HMPPCG::DataObject* fict,
  HMPPCG::DataObject* ex,
  HMPPCG::DataObject* ey,
  HMPPCG::DataObject* hz,
  int &hmppcg_status, HMPPCG::CLcontext& context)
{
  {
    int32_t __hmppcg_end, t_1;
    for (t_1 = 0, __hmppcg_end = 499; t_1 <= __hmppcg_end; t_1 += 1)
    {
      {
      cl_event kevent;
      size_t global_size[1];
      global_size[0] = ((2047 / (32LL * 8LL)) + 1LL) * (32LL * 8LL);
      size_t local_size[1];
      local_size[0] = 32LL*8LL;
      ey->makeQueueWait(context.cmd_queue);
      fict->makeQueueWait(context.cmd_queue);
      int32_t t_1_1 = t_1;
      clSetKernelArg(context.kernels[0], 0, sizeof(int32_t), (void*) &t_1_1);
      clSetKernelArg(context.kernels[0], 1, sizeof(cl_mem), (void*) &(ey->mem_obj));
      clSetKernelArg(context.kernels[0], 2, sizeof(cl_mem), (void*) &(fict->mem_obj));
      cl_int err = clEnqueueNDRangeKernel(context.cmd_queue, context.kernels[0], 1, NULL, global_size, local_size, 0, NULL, &kevent);
      if(CHECK_STATUS(err)) {hmppcg_status = -1; return;}
      clFlush(context.cmd_queue);
      ey->waitOnEvent(kevent);
      fict->waitOnEvent(kevent);
      clReleaseEvent(kevent);
      }
      ;
      {
      cl_event kevent;
      size_t global_size[2];
      global_size[0] = ((2047 / 32LL) + 1LL) * (32LL);
      global_size[1] = ((2046 / 8LL) + 1LL) * (8LL);
      size_t local_size[2];
      local_size[0] = 32LL;
      local_size[1] = 8LL;
      ey->makeQueueWait(context.cmd_queue);
      hz->makeQueueWait(context.cmd_queue);
      clSetKernelArg(context.kernels[1], 0, sizeof(cl_mem), (void*) &(ey->mem_obj));
      clSetKernelArg(context.kernels[1], 1, sizeof(cl_mem), (void*) &(hz->mem_obj));
      cl_int err = clEnqueueNDRangeKernel(context.cmd_queue, context.kernels[1], 2, NULL, global_size, local_size, 0, NULL, &kevent);
      if(CHECK_STATUS(err)) {hmppcg_status = -1; return;}
      clFlush(context.cmd_queue);
      ey->waitOnEvent(kevent);
      hz->waitOnEvent(kevent);
      clReleaseEvent(kevent);
      }
      ;
      {
      cl_event kevent;
      size_t global_size[2];
      global_size[0] = ((2046 / 32LL) + 1LL) * (32LL);
      global_size[1] = ((2047 / 8LL) + 1LL) * (8LL);
      size_t local_size[2];
      local_size[0] = 32LL;
      local_size[1] = 8LL;
      ex->makeQueueWait(context.cmd_queue);
      hz->makeQueueWait(context.cmd_queue);
      clSetKernelArg(context.kernels[2], 0, sizeof(cl_mem), (void*) &(ex->mem_obj));
      clSetKernelArg(context.kernels[2], 1, sizeof(cl_mem), (void*) &(hz->mem_obj));
      cl_int err = clEnqueueNDRangeKernel(context.cmd_queue, context.kernels[2], 2, NULL, global_size, local_size, 0, NULL, &kevent);
      if(CHECK_STATUS(err)) {hmppcg_status = -1; return;}
      clFlush(context.cmd_queue);
      ex->waitOnEvent(kevent);
      hz->waitOnEvent(kevent);
      clReleaseEvent(kevent);
      }
      ;
      {
      cl_event kevent;
      size_t global_size[2];
      global_size[0] = ((2047 / 32LL) + 1LL) * (32LL);
      global_size[1] = ((2047 / 8LL) + 1LL) * (8LL);
      size_t local_size[2];
      local_size[0] = 32LL;
      local_size[1] = 8LL;
      ex->makeQueueWait(context.cmd_queue);
      ey->makeQueueWait(context.cmd_queue);
      hz->makeQueueWait(context.cmd_queue);
      clSetKernelArg(context.kernels[3], 0, sizeof(cl_mem), (void*) &(ex->mem_obj));
      clSetKernelArg(context.kernels[3], 1, sizeof(cl_mem), (void*) &(ey->mem_obj));
      clSetKernelArg(context.kernels[3], 2, sizeof(cl_mem), (void*) &(hz->mem_obj));
      cl_int err = clEnqueueNDRangeKernel(context.cmd_queue, context.kernels[3], 2, NULL, global_size, local_size, 0, NULL, &kevent);
      if(CHECK_STATUS(err)) {hmppcg_status = -1; return;}
      clFlush(context.cmd_queue);
      ex->waitOnEvent(kevent);
      ey->waitOnEvent(kevent);
      hz->waitOnEvent(kevent);
      clReleaseEvent(kevent);
      }
      ;
    } 
  }
} 
// ---------------------------------------------------------------------------
// HMPP handle
// ---------------------------------------------------------------------------
typedef struct hmpp_handle_struct
{
  HMPPCG::CLcontext context;
  HMPPCG::DataObject* __arg0;
  HMPPCG::DataObject* __arg1;
  HMPPCG::DataObject* __arg2;
  HMPPCG::DataObject* __arg3;
} hmpp_handle_t;


// ---------------------------------------------------------------------------
// hmpp_createInstance()
// ---------------------------------------------------------------------------
HMPP_API hmpp_handle_t * hmpp_createInstance()
{
  hmpp_handle_t * __h = new hmpp_handle_t;
  if(!__h) return 0;
  return __h;
}

// ---------------------------------------------------------------------------
// hmpp_setOpenCLDevice()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_setOpenCLDevice(hmpp_handle_t * __h, cl_context ctx, cl_device_id dev)
{
  __h->context.prog_name = "fdtd_opencl.cc-kernels.cl";
  __h->context.kernels_names.push_back("hmpp_codelet__runFdtd_loop0_");
  __h->context.kernels_names.push_back("hmpp_codelet__runFdtd_loop1_");
  __h->context.kernels_names.push_back("hmpp_codelet__runFdtd_loop2_");
  __h->context.kernels_names.push_back("hmpp_codelet__runFdtd_loop3_");
  if (__h->context.init(ctx, dev) == -1)
  {
    fprintf(stderr, "CL Error during init\n");
    return -1;
  }
  else
  {
    if (!__h->context.checkDoubleAvailable())
    {
      fprintf(stderr,"CL Error: Can't execute codelet on current device: OpenCL extension \"cl_khr_fp64\" for double floating-point precision support is missing.\n" );
      size_t buf_size;
      char *device_vendor = NULL;
      if( CHECK_STATUS( clGetDeviceInfo( dev, CL_DEVICE_VENDOR, 0, NULL, &buf_size) ) == 0 ) {
        device_vendor = (char *)calloc(buf_size, sizeof(char));
        if( CHECK_STATUS( clGetDeviceInfo( dev, CL_DEVICE_VENDOR, buf_size, device_vendor, NULL) ) == 0 )
          if( ! strcmp( device_vendor, "Advanced Micro Devices, Inc." ) )
            fprintf(stderr,"CL Info: You can try to set GPU_DOUBLE_PRECISION=1 before running the application.\n" );
      }
      free( device_vendor );
      return -1;
    }
    if (__h->context.initProgram() == -1)
    {
      fprintf(stderr, "CL Error during init\n");
      return -1;
    }
    return 0;
  }

}
// ---------------------------------------------------------------------------
// hmpp_freeInstance()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_freeInstance(hmpp_handle_t * __h)
{
  __h->context.release();
  delete(__h);
  return 0;
}

// ---------------------------------------------------------------------------
// hmpp_allocateOnHWA()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_allocateOnHWA(hmpp_handle_t * __h, int major, int minor, const size_t * size, size_t elsize, int dim)
{
  size_t total;
  if (!dim) total = 1;
  else total = size[0];
  for(int i = 1; i < dim; i++)
  total *= size[i];
  if (total == 0) total = 1;
  total *= elsize;
  switch(HMPPCG_POP_HASH(major,minor))
  {
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      __h->__arg0 = __h->context.allocateData(elsize, total, HMPPCG::READ_ONLY);
      if  (!__h->__arg0)
       return -1;
       else
       return 0;
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      __h->__arg1 = __h->context.allocateData(elsize, total, HMPPCG::READ_WRITE);
      if  (!__h->__arg1)
       return -1;
       else
       return 0;
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      __h->__arg2 = __h->context.allocateData(elsize, total, HMPPCG::READ_WRITE);
      if  (!__h->__arg2)
       return -1;
       else
       return 0;
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      __h->__arg3 = __h->context.allocateData(elsize, total, HMPPCG::READ_WRITE);
      if  (!__h->__arg3)
       return -1;
       else
       return 0;
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
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      return __h->__arg0->readData(data, async != 0);
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      return __h->__arg1->readData(data, async != 0);
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      return __h->__arg2->readData(data, async != 0);
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      return __h->__arg3->readData(data, async != 0);
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
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      return __h->__arg0->writeData(data, async != 0);
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      return __h->__arg1->writeData(data, async != 0);
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      return __h->__arg2->writeData(data, async != 0);
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      return __h->__arg3->writeData(data, async != 0);
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
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      return __h->__arg0->sectionCopy(data, section, size, elsize, dim, false, HMPPCG::DataObject::READ);
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      return __h->__arg1->sectionCopy(data, section, size, elsize, dim, false, HMPPCG::DataObject::READ);
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      return __h->__arg2->sectionCopy(data, section, size, elsize, dim, false, HMPPCG::DataObject::READ);
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      return __h->__arg3->sectionCopy(data, section, size, elsize, dim, false, HMPPCG::DataObject::READ);
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
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      return __h->__arg0->sectionCopy((void*) data, section, size, elsize, dim , false, HMPPCG::DataObject::WRITE);
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      return __h->__arg1->sectionCopy((void*) data, section, size, elsize, dim , false, HMPPCG::DataObject::WRITE);
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      return __h->__arg2->sectionCopy((void*) data, section, size, elsize, dim , false, HMPPCG::DataObject::WRITE);
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      return __h->__arg3->sectionCopy((void*) data, section, size, elsize, dim , false, HMPPCG::DataObject::WRITE);
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
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg0->cmd);
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg1->cmd);
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg2->cmd);
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg3->cmd);
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
    case HMPPCG_POP_HASH(1,0): // fict@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg0->cmd);
    }
    case HMPPCG_POP_HASH(1,1): // ex@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg1->cmd);
    }
    case HMPPCG_POP_HASH(1,2): // ey@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg2->cmd);
    }
    case HMPPCG_POP_HASH(1,3): // hz@hmpp_codelet__runFdtd
    {
      return clFinish(__h->__arg3->cmd);
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
    case 1: // hmpp_codelet__runFdtd(__arg0,__arg1,__arg2,__arg3)
      hmpp_codelet__runFdtd(  __h->__arg0, __h->__arg1, __h->__arg2, __h->__arg3, status, __h->context);
      return status;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// hmpp_wait()
// ---------------------------------------------------------------------------
HMPP_API int hmpp_wait(hmpp_handle_t * __h,int codelet_id)
{
  return (__h->context.wait() != CL_SUCCESS);
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



