BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin

CC := nvcc

LOWP_ARCH    ?= sm_120a
LOWP_RS_ARCH ?= sm_100a
CUDA_CPPFLAGS =

LOWP_EXE    := lowp_cvt_runtime_hw_sim_diff
LOWP_RS_EXE := lowp_cvt_runtime_hw_sim_diff_rs

LOWP_NVCC_FLAGS := -gencode=arch=compute_$(subst sm_,,$(LOWP_ARCH)),code=$(LOWP_ARCH) \
                   -gencode=arch=compute_$(subst sm_,,$(LOWP_ARCH)),code=compute_$(subst sm_,,$(LOWP_ARCH)) \
                   -std=c++17 -O2
LOWP_RS_NVCC_FLAGS := -gencode=arch=compute_$(subst sm_,,$(LOWP_RS_ARCH)),code=$(LOWP_RS_ARCH) \
                      -gencode=arch=compute_$(subst sm_,,$(LOWP_RS_ARCH)),code=compute_$(subst sm_,,$(LOWP_RS_ARCH)) \
                      -std=c++17 -O2

.DEFAULT_GOAL := all

all: $(BIN_DIR)/$(LOWP_EXE) $(BIN_DIR)/$(LOWP_RS_EXE)

$(BIN_DIR):
	mkdir -p $@

$(LOWP_EXE).a: $(LOWP_EXE).cu
	$(CC) $(LOWP_NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(LOWP_RS_EXE).a: $(LOWP_RS_EXE).cu
	$(CC) $(LOWP_RS_NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(BIN_DIR)/$(LOWP_EXE): $(LOWP_EXE).a | $(BIN_DIR)
	$(CC) $(LOWP_NVCC_FLAGS) $< -o $@ -lcudart --cudart shared

$(BIN_DIR)/$(LOWP_RS_EXE): $(LOWP_RS_EXE).a | $(BIN_DIR)
	$(CC) $(LOWP_RS_NVCC_FLAGS) $< -o $@ -lcudart --cudart shared

ptxas-crosscheck:
	./run_lowp_cvt_ptxas_crosscheck.sh ptx_datatype_cases

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(LOWP_EXE) $(LOWP_RS_EXE)
	rm -f $(BIN_DIR)/$(LOWP_EXE) $(BIN_DIR)/$(LOWP_RS_EXE)

.PHONY: all clean ptxas-crosscheck
