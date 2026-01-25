SRC = mbarrier.cu

EXE = mbarrier

ARCH?=sm_80 sm_90a sm_100a sm_101 sm_120 
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
CUDA_CPPFLAGS=

# Generate code for both sm_XXX and compute_XXX (SASS and PTX)
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

include ../../../common/common.mk
