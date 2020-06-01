#===============================================================================
# User Options
#===============================================================================

COMPILER = gnu
OPTIMIZE = yes
DEBUG    = no
PROFILE  = no
PAPI     = no


GENCODE_SM10 := -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 := -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 := -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 := -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 := -gencode=arch=compute_62,code=\"sm_62,compute_62\"

CUOPTS =  $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62)
#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench
#program = XSBench-app-ann-data-sense

OBJS = \
Main.o \
CalculateXS.o \
GridInit.o \
XSutils.o \
Materials.o

HEADERS = \
XSbench_header.h \
cudaHeader.h 
#===============================================================================
# Sets Flags
#===============================================================================

# Regular gcc Compiler
ifeq ($(COMPILER),gnu)
  GCC = nvcc
endif

# BG/Q gcc Cross-Compiler
ifeq ($(MACHINE),bluegene)
  GCC = /bgsys/drivers/toolchain/V1R1M2/gnu-linux/bin/powerpc64-bgq-linux-gcc
endif


# Standard Flags
GCC_S_FLAGS := 
GCC_O_FLAGS =
LDFLAGS = 
NVCC_FLAGS = -I. $(CUOPTS) -g -Xptxas=-v  -Xcompiler "-fopenmp -Wall -std=c99"
#NVCC_FLAGS = -I. -arch=sm_20  -Xptxas=-v  -Xcompiler "-fopenmp -Wall -std=c99"

# Debug Flags
ifeq ($(DEBUG),yes)
  GCC_O_FLAGS += -g
  LDFLAGS  += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  GCC_O_FLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  GCC_O_FLAGS += -O3
endif

# PAPI source 
ifeq ($(PAPI),yes)
  source += papi.c
endif


EXECUTABLE = $(BINDIR)/$(BINSUBDIR)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(OBJS) XSbench_header.h cudaCalcXs.o $(HEADERS)
	nvcc $(NVCC_FLAGS) cudaCalcXs.o $(OBJS) -o $(EXECUTABLE)/$@ $(LDFLAGS) -lcudart

%.o : %.c $(HEADERS)
	$(GCC) $(NVCC_FLAGS) $(GCC_S_FLAGS) $(GCC_O_FLAGS) -o $@ -c $<

cudaCalcXs.o: cudaCalcXs.cu cudaHeader.h XSbench_header.h 
	nvcc  $(NVCC_FLAGS) $(GCC_S_FLAGS) -c cudaCalcXs.cu -o cudaCalcXs.o -I. -lcudart

clean:
	rm -rf XSBench XSBench.dSYM *.o *.os *.ptx *.vir *.ccbak *.s *.so
edit:
	vim -p $(source) do_flops.c papi.c XSbench_header.h
run:
	./XSBench
bgqrun:
	qsub -t 10 -n 1 -O test XSBench
