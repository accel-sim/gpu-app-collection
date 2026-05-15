BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin

CC  := nvcc
EXE := gpgpusim_griddepcontrol_semantics
SRC := griddepcontrol_semantics.cu

# Programmatic dependent launch / grid dependency control is Hopper-era CUDA
# functionality. Keep the default narrow, but allow callers to override ARCH.
ARCH ?= sm_90

CUDA_CPPFLAGS =
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

CUDA_SRC_FILES := $(filter %.cu, $(SRC))
CUDA_LIB_FILES := $(CUDA_SRC_FILES:.cu=.a)

.DEFAULT_GOAL := all

all: $(BIN_DIR)/$(EXE)

$(BIN_DIR):
	mkdir -p $@

%.a: %.cu
	$(CC) $(NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(BIN_DIR)/$(EXE): $(CUDA_LIB_FILES) | $(BIN_DIR)
	$(CC) $(NVCC_FLAGS) $^ -o $@ -lcudart --cudart shared

run: $(BIN_DIR)/$(EXE)
	$(BIN_DIR)/$(EXE)

ptx: $(BIN_DIR)/$(EXE)
	cuobjdump -ptx $(BIN_DIR)/$(EXE) | tee $(EXE).ptx

sass: $(BIN_DIR)/$(EXE)
	cuobjdump -sass $(BIN_DIR)/$(EXE) | tee $(EXE).sass

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(EXE) $(CUDA_LIB_FILES)
	rm -f $(BIN_DIR)/$(EXE)

.PHONY: all clean run ptx sass
