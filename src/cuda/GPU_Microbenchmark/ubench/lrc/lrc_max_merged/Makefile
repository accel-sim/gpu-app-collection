SRC = lrc_max_merged.cu

EXE = lrc_max_merged

# NVCC_FLGAS = -Xptxas -dlcm=cg -Xptxas -dscm=wt
# LRC is supported on SM_90 and above
ARCH?=sm_90a sm_100a sm_101 sm_120 
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
# but LRC is only supported on SM_90 and above
CUDA_CPPFLAGS=
# Generate code for both sm_XXX and compute_XXX (SASS and PTX)
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17 -O0

include ../../../common/common.mk
