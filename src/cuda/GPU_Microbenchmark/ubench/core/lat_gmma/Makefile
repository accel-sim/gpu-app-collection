# Source files split for parallel compilation
# Use wildcard to automatically include all size-specific breakdown files
SRC = lat_gmma.cu $(wildcard kernels/lat_gmma_*.cu)

EXE = lat_gmma

# Add include path for CUTLASS
INCLUDE += -I$(GPUAPPS_ROOT)/src/cuda/cutlass-bench/include -I./

# GMMA is only supported in sm_90a
ARCH?=sm_90a
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
CUDA_CPPFLAGS=
# Generate code for both sm_90a and compute_90a (SASS and PTX)
HOPPER_CUDA_CPPFLAGS=$(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch)))

# CUTLASS cute library requires C++17
NVCC_FLAGS := $(HOPPER_CUDA_CPPFLAGS) -std=c++17

include ../../../common/common.mk

# A test executable for checking the library and simulator debugging
TEST_SRC = lat_gmma_test.cu
TEST_EXE = lat_gmma_test
test: $(TEST_SRC)
	$(CC) $(NVCC_FLAGS) $^ -o $(TEST_EXE) $(INCLUDE) -lcudart --cudart shared
	cp $(TEST_EXE) $(BIN_DIR)

# Append the test to the release target
release: .EXTRA_PREREQS = test