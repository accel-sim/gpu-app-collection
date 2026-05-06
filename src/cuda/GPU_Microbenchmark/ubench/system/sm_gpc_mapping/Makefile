SRC = sm_gpc_mapping.cu

EXE = sm_gpc_mapping

# Thread-block clusters (%cluster_ctarank, cudaLaunchKernelEx with
# cudaLaunchAttributeClusterDimension) require sm_90+.
NVCC_FLAGS += -arch=sm_90

include ../../../common/common.mk
