#include "CoMDTypes.h"
#include "gpu_types.h"
#include <memory.h>

#include <cuda_runtime.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64) 
#include <winsock2.h>
#else
#define _XOPEN_SOURCE 500
#include <strings.h>
#include <unistd.h>
#endif

void SetupGpu(int deviceId);
void AllocateGpu(SimFlat *flat, int do_eam, char *method);
void SetBoundaryCells(SimFlat *flat, HaloExchange* hh);		// for communication latency hiding
void CopyDataToGpu(SimFlat *flat, int do_eam);
void GetDataFromGpu(SimFlat *flat);
void DestroyGpu(SimFlat *flat);

