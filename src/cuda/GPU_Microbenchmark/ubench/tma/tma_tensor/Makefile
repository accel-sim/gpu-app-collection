# TMA Tensor microbenchmarks
# Builds the monolithic tma_tensor (manual -o flag) plus per-variant standalone
# binaries (utmapf, utmaldg, utmastg, utmaredg) that the CI runner discovers
# and gates automatically.

BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

ALL_BENCHES := tma_tensor utmapf utmaldg utmastg utmaredg \
               utmapf_3d utmaldg_3d utmastg_3d utmaredg_3d \
               utmaldg_3d_tx

# TMA tensor requires Hopper-class (sm_90a) or newer.
ARCH ?= sm_90a sm_100a sm_101 sm_120
CC   := nvcc

# Generate both SASS and PTX for each target architecture.
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

.DEFAULT_GOAL := all

all: $(ALL_BENCHES)

# Two-stage build: .cu -> .a (preserves PTX in binary) -> executable
%.a: %.cu
	$(CC) $(NVCC_FLAGS) --lib $< -o $@

$(ALL_BENCHES): %: %.a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(ALL_BENCHES)

.PHONY: all clean
