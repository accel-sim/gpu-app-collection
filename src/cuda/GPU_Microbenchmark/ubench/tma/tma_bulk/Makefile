# TMA Bulk microbenchmarks
# Build all benchmarks in this folder when invoked from the top-level
# GPU_Microbenchmark make recursion.

BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# All bulk microbenchmarks in this directory.
ALL_BENCHES := tma_bulk ublkcp_g_s ublkcp_s_g ublkpf ublkred_g_s

# TMA is supported on SM_90a and above.
ARCH ?= sm_90a sm_100a sm_101 sm_120
CC   := nvcc

# Keep same codegen style used across GPU_Microbenchmark.
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

.DEFAULT_GOAL := all

all: $(ALL_BENCHES)

# Compile .cu -> .a so PTX is preserved, then link with shared cudart.
%.a: %.cu
	$(CC) $(NVCC_FLAGS) --lib $< -o $@

$(ALL_BENCHES): %: %.a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(ALL_BENCHES)

.PHONY: all clean
