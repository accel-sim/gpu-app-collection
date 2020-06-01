# Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
# George L. Yuan, Jimmy Kwa and the
# University of British Columbia
# Vancouver, BC  V6T 1Z4
# All Rights Reserved.
#
# THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
# TERMS AND CONDITIONS.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
# are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
# (property of NVIDIA).  The files benchmarks/BlackScholes/* and
# benchmarks/template/* are derived from the CUDA SDK available from
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files
# src/gpgpusim_entrypoint.c and src/simplesim-3.0/* are derived from the
# SimpleScalar Toolset available from http://www.simplescalar.com/
# (property of SimpleScalar LLC) and the files src/intersim/* are derived
# from Booksim (Simulator provided with the textbook "Principles and
# Practices of Interconnection Networks" available from
# http://cva.stanford.edu/books/ppin/).  As such, those files are bound by
# the corresponding legal terms and conditions set forth separately (original
# copyright notices are left in files from these sources and where we have
# modified a file our copyright notice appears before the original copyright
# notice).
#
# Using this version of GPGPU-Sim requires a complete installation of CUDA
# version 1.1, which is distributed seperately by NVIDIA under separate terms
# and conditions.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the University of British Columbia nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.
#
# 5. No nonprofit user may place any restrictions on the use of this software,
#  including as modified by the user, by any other authorized user.
#
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
# Ali Bakhoda, George L. Yuan, at the University of British Columbia,
# Vancouver, BC V6T 1Z4

ifeq ($(ARCH),ARM32)
	GNUVERSION = 4.6
	CPP = /usr/bin/arm-linux-gnueabihf-g++-${GNUVERSION}
	CC = /usr/bin/arm-linux-gnueabihf-gcc-${GNUVERSION}
else
	CPP = g++-4.4
	CC = gcc-4.4
endif

NVCC_VERSION:=$(shell nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
INCFLAGEXTRA ?= -I$(GEM5_GPU_BENCHMARKS)/../gem5/util/m5 -I$(GEM5_GPU_BENCHMARKS)/libcuda
ifeq ($(ARCH),ARM32)
	INCFLAGEXTRA += -I/usr/arm-linux-gnueabihf/include -I/usr/lib/gcc/arm-linux-gnueabihf/${GNUVERSION}/include -I/usr/arm-linux-gnueabihf/include/c++/${GNUVERSION}.2/arm-linux-gnueabihf -I/usr/arm-linux-gnueabihf/include/c++/${GNUVERSION}.3/ext -I/usr/include/c++/${GNUVERSION}
endif
CFLAGS     ?=
LDUFLAGS   ?=
EXTRA_OBJS ?=
CCFLAGS    ?=
CUFLAGS    ?=
OPT        ?= -O3
ifeq ($(ARCH),ARM32)
	LIB        ?= -lm5op_arm
else
	LIB        ?= -L/usr/lib64 -lcutil_x86_64 -lm5op_x86
endif
SRCDIR     ?=
ROOTDIR    ?=
ROOTBINDIR ?= bin
BINDIR     ?= $(ROOTBINDIR)
ROOTOBJDIR ?= obj
ifneq ($(NVCC_VERSION),2.3)
LIBDIR     := $(NVIDIA_CUDA_SDK_LOCATION)/lib
COMMONDIR  := $(NVIDIA_CUDA_SDK_LOCATION)/common
SDKINCDIR  := $(NVIDIA_CUDA_SDK_LOCATION)/common/inc/
else
LIBDIR     := $(NVIDIA_CUDA_SDK_LOCATION)/C/lib
COMMONDIR  := $(NVIDIA_CUDA_SDK_LOCATION)/C/common
SDKINCDIR  := $(NVIDIA_CUDA_SDK_LOCATION)/C/common/inc/
endif
GEM5_GPU_BENCHMARKS ?= ../..
INTERMED_FILES := *.cpp*.i *.cpp*.ii *.cu.c *.cudafe*.* *.fatbin.c *.cu.cpp *.linkinfo *.cpp_o core
COMPUTETARGET ?= sm_20

GEM5GPUFLAGS  := -DGEM5_FUSION

MEM_DEBUG ?= 0
ifeq ($(MEM_DEBUG),1)
	GEM5GPUFLAGS :=
endif

SIM_OBJDIR :=
SIM_OBJS +=  $(patsubst %.cpp,$(SIM_OBJDIR)%.cpp_o,$(CCFILES))
SIM_OBJS +=  $(patsubst %.c,$(SIM_OBJDIR)%.c_o,$(CFILES))
SIM_OBJS +=  $(patsubst %.cu,$(SIM_OBJDIR)%.cu_o,$(CUFILES))

.SUFFIXES:

gem5_fusion_$(EXECUTABLE): $(SIM_OBJS)
ifeq ($(ARCH),ARM32)
	$(CPP) $(CFLAGS) $(OPT) $(GEM5GPUFLAGS) $(notdir $(SIM_OBJS)) -L$(GEM5_GPU_BENCHMARKS)/libcuda -lcuda \
		-static -static-libgcc -o gem5_fusion_$(EXECUTABLE) $(LIB) -lm -lc $(EXTRA_OBJS) $(LDUFLAGS)
else
	$(CPP) $(CFLAGS) $(OPT) $(GEM5GPUFLAGS) $(notdir $(SIM_OBJS)) -L$(GEM5_GPU_BENCHMARKS)/libcuda -lcuda \
		-L$(LIBDIR) \
		-lz -static -static-libgcc -o gem5_fusion_$(EXECUTABLE) $(LIB) -lm -lc $(EXTRA_OBJS) $(LDUFLAGS)
endif

%.cpp_o: %.cpp
	$(CPP) $(CCFLAGS) $(OPT) $(INCFLAGEXTRA) -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@)

%.cc_o: %.cc
	$(CPP) $(CCFLAGS) $(OPT) $(INCFLAGEXTRA) -I./ -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@)

%.c_o: %.c
	$(CC) $(CFLAGS) $(OPT) $(INCFLAGEXTRA)  -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@)

%.cu_o: %.cu
ifeq ($(ARCH),ARM32)
	nvcc $(CUFLAGS) $(OPT) -cuda -arch $(COMPUTETARGET) --compiler-options -fno-strict-aliasing \
		$(GEM5GPUFLAGS) -I. -I$(CUDAHOME)/include/ -I$(SDKINCDIR) \
		$(INCFLAGEXTRA) $<
	python ../../common/sizeHack.py -f $<.cpp -t $(COMPUTETARGET)
	python ../../common/alignHack.py -f $<.cpp -t $(COMPUTETARGET)
else
	nvcc $(CUFLAGS) $(OPT) -c -arch $(COMPUTETARGET) --keep --compiler-options -fno-strict-aliasing \
		$(GEM5GPUFLAGS) -I. -I$(CUDAHOME)/include/ -I$(SDKINCDIR) \
		 $(INCFLAGEXTRA) -L$(LIBDIR) -lcutil -DUNIX $< -o $(EXECUTABLE)
	python ../../common/sizeHack.py -f $<.cpp -t $(COMPUTETARGET)
endif
ifneq ($(NVCC_VERSION),1.1)
	$(CPP) $(CFLAGS) $(OPT) -g -c $(notdir $<.cpp) -o $(notdir $@)
else
	$(CC) $(CFLAGS) $(OPT) -g -c $(notdir $<.c) -o $(notdir $@)
endif
#	$(GEM5_GPU_BENCHMARKS)/scripts/gen_ptxinfo

%.cu: %.cu.c

clean:
	rm -f $(INTERMED_FILES) *.cubin *.o *_o *.hash *.ptx *.ptxinfo cubin.bin $(EXECUTABLE) gem5_fusion_$(EXECUTABLE)

