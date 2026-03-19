SRC = mbarrier.cu

EXE = mbarrier

ARCH_DEFAULT := sm_80 sm_90a sm_100a
ifneq ($(filter 13 14 15 16,$(CUDA_VERSION_MAJOR)),)
ARCH_DEFAULT += sm_110
else
ARCH_DEFAULT += sm_101
endif
ARCH_DEFAULT += sm_120
ARCH ?= $(ARCH_DEFAULT)
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
CUDA_CPPFLAGS=

# Generate code for both sm_XXX and compute_XXX (SASS and PTX)
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

include ../../../common/common.mk
