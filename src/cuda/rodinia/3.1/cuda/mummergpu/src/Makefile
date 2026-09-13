include ../../../common/make.config

all: mummergpu
.SUFFIXES : .cu .cu_dbg_o .c_dbg_o .cpp_dbg_o .cu_rel_o .c_rel_o .cpp_rel_o .cubin

include ../experiments/test_rule.mk

CUDA_INSTALL_PATH := $(CUDA_DIR)

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc  $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM90) $(GENCODE_SM100) $(GENCODE_SM103) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75)
CXX        := g++ -fopenmp $(PROFILE)
CC         := gcc 
# -m64 is an x86-only option; aarch64/ppc64le gcc rejects it outright.
M64 := $(if $(filter x86_64,$(shell uname -m)),-m64,)
LINK       := g++ -fopenmp $(M64) -fPIC $(PROFILE)

# Add source files here
STATIC_LIB := libmummergpu.a
# Cuda source files (compiled with cudacc)
CUFILES		:= mummergpu.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	 mummergpu_gold.cpp suffix-tree.cpp PoolMalloc.cpp

################################################################################
# Rules and targets


# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
ROOTDIR    ?= ..
ROOTBINDIR ?= $(ROOTDIR)/bin
BINDIR     ?= $(ROOTBINDIR)
ROOTOBJDIR ?= obj
LIBDIR     := $(ROOTDIR)/lib
# Land in the shared bin dir under the final benchmark name; see the sibling
# rodinia makefiles. Replaces the top-level Makefile's post-build mv.
RODINIA_OUT ?= $(if $(BINSUBDIR),$(BINDIR)/$(BINSUBDIR),$(BINDIR))
BINNAME    ?= mummergpu-rodinia-3.1
# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include 

# Libs
LIB       := -L$(CUDA_LIB_DIR) -L$(LIBDIR) -lcudart -lGL -lGLU ${LIB}

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# Compiler-specific flags
NVCCFLAGS := $(if $(M64),-Xcompiler "$(M64)",)
CXXFLAGS  := $(M64) $(CXXWARN_FLAGS)
CFLAGS    := $(M64) $(CWARN_FLAGS)

# Common flags
COMMONFLAGS ?= 
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O3
	BINSUBDIR   := release
	LIBSUFFIX   :=
endif

# Device emulation configuration
ifeq ($(emu), 1)
	NVCCFLAGS   += -deviceemu
	CUDACCFLAGS += 
	BINSUBDIR   := emu$(BINSUBDIR)
endif

ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

# Add cudacc flags
NVCCFLAGS += -Xcompiler "$(CUDACCFLAGS)"

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)

#.c.c_$(APPEND)_o:
$(OBJDIR)/%.c_o : $(SRCDIR)%.c
	$(CC) $(CFLAGS) -o $@ -c $<

#.cpp.cpp_$(APPEND)_o:
$(OBJDIR)/%.cpp_o : $(SRCDIR)%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

#.cu.cu_$(APPEND)_o:
$(OBJDIR)/%.cu_o : $(SRCDIR)%.cu
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

#.cu.cubin
%.cubin : $(SRCDIR)mummergpu.cu mummergpu_kernel.cu
	$(NVCC) -o $(BINNAME).cubin -cubin $< $(NVCCFLAGS)

#.cu.ptx
%.ptx : $(SRCDIR)%.cu mummergpu_kernel.cu
	$(NVCC) -o $@ -ptx -Xopencc -LIST:source=on $< $(NVCCFLAGS)

makedirectories:
	@mkdir -p $(LIBDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(RODINIA_OUT)

LIBOBJS= \
    $(OBJDIR)/mummergpu_gold.cpp_o \
    $(OBJDIR)/suffix-tree.cpp_o \
    $(OBJDIR)/PoolMalloc.cpp_o \
    $(OBJDIR)/mummergpu.cu_o \

$(LIBDIR)/libmummergpu$(LIBSUFFIX).a: $(LIBOBJS)
	ar qv $@ $(LIBOBJS)

BINOBJS= \
    $(OBJDIR)/mummergpu_main.cpp_o

$(RODINIA_OUT)/$(BINNAME): makedirectories $(LIBDIR)/libmummergpu$(LIBSUFFIX).a $(BINOBJS)
	$(LINK) -o $(RODINIA_OUT)/$(BINNAME)  $(BINOBJS) -lmummergpu$(LIBSUFFIX) $(LIB) 
		

##############################################################################
#                               Auto Generated rules go here                 #
##############################################################################

include ../experiments/rules.mk

##############################################################################
#                              End auto generated rules                      #
##############################################################################


clean: 
	rm -f $(LIBOBJS)
	rm -f $(LIBDIR)/libmummergpu$(LIBSUFFIX).a
	rm -f $(BINOBJS)
	rm -f $(BINDIR)/$(BINNAME)
	rm -f *.linkinfo
	rm -rf $(ROOTBINDIR)
	rm -rf $(ROOTOBJDIR)
	rm -rf $(LIBDIR)

mummergpu: $(RODINIA_OUT)/$(BINNAME)

cubin: mummergpu.cubin

ptx: mummergpu.ptx

