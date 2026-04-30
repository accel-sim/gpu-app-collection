BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

ALL_BENCHES := wgmma_shapes

# WGMMA requires Hopper (SM90a).
ARCH ?= sm_90a
CC   := nvcc

CUDA_CPPFLAGS=

NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) \
              -std=c++17

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
