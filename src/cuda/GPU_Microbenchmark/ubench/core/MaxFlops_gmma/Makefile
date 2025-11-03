# Source files split for parallel compilation
# Use wildcard to automatically include all size-specific breakdown files
SRC = MaxFlops_gmma.cu $(wildcard kernels/MaxFlops_gmma_*.cu)

EXE = MaxFlops_gmma

# Add include path for CUTLASS
INCLUDE += -I$(GPUAPPS_ROOT)/src/cuda/cutlass-bench/include -I./

# GMMA is only supported in sm_90a
ARCH?=sm_90a
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
CUDA_CPPFLAGS=
# Generate code for both sm_XXX and compute_XXX (SASS and PTX)
HOPPER_CUDA_CPPFLAGS=$(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch)))

# CUTLASS cute library requires C++17
NVCC_FLAGS := $(HOPPER_CUDA_CPPFLAGS) -std=c++17

include ../../../common/common.mk
