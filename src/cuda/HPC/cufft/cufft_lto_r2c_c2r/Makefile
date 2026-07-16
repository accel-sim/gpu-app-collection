EXECUTABLE := cufft_lto_r2c_c2r_scalable
CUFILES := r2c_c2r_reference.cu
CCFILES := cufft_lto_r2c_c2r_scalable.cpp common.cpp
ADDITIONAL_LIBS := -lcufft
OMIT_CUTIL_LIB := 1
OMIT_SHRUTIL_LIB := 1
SM_VERSIONS := 75 80 90

# LTO fatbin generation variables
CUDA_LTO_ARCH ?= 75
BIN2C ?= $(CUDA_INSTALL_PATH)/bin/bin2c
FATBIN_HEADER := r2c_c2r_lto_callback_device_fatbin.h
NVCC_KEEP_CLEAN := $(FATBIN_HEADER) r2c_c2r_lto_callback_device.fatbin

include ../../../common/common.mk

# Generate fatbin header (custom rule added AFTER common.mk)
$(FATBIN_HEADER): r2c_c2r_lto_callback_device.cu
	@echo "Generating LTO fatbin header..."
	@$(CUDA_INSTALL_PATH)/bin/nvcc --std=c++11 --generate-code arch=compute_$(CUDA_LTO_ARCH),code=lto_$(CUDA_LTO_ARCH) -dc -fatbin $< -o r2c_c2r_lto_callback_device.fatbin
	@$(BIN2C) --name window_callback --type longlong r2c_c2r_lto_callback_device.fatbin > $@

# Make cpp source depend on fatbin header so it gets built first
cufft_lto_r2c_c2r_scalable.cpp: $(FATBIN_HEADER)
