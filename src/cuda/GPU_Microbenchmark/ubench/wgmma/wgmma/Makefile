BASE_DIR     := $(shell pwd)
BIN_DIR      := $(BASE_DIR)/../../../bin/
CUTLASS_ROOT ?= $(BASE_DIR)/../../../../cutlass-bench

# WGMMA microbenchmarks.
# wgmma_no_tma / wgmma_tf32_128 use CuTe (CUTLASS) headers; wgmma_verify is standalone.
ALL_BENCHES := wgmma_verify wgmma_no_tma wgmma_tf32_128

# WGMMA requires Hopper (SM90a).
ARCH ?= sm_90a
CC   := nvcc

CUDA_CPPFLAGS=

NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) \
              -std=c++17 \
              -I$(CUTLASS_ROOT)/include \
              -I$(CUTLASS_ROOT)/tools/util/include \
              -DCUTLASS_ARCH_MMA_SM90_SUPPORTED

.DEFAULT_GOAL := all

all: $(ALL_BENCHES)

%.a: %.cu
	$(CC) $(NVCC_FLAGS) --lib $< -o $@

$(ALL_BENCHES): %: %.a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(ALL_BENCHES)

.PHONY: all clean
