#===============================================================================
# User Options
#===============================================================================

COMPILER = gnu
OPTIMIZE = yes
DEBUG    = no
PROFILE  = no
PAPI     = no

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
NVCC_FLAGS = -I. -g -Xptxas=-v  -Xcompiler "-fopenmp -Wall -std=c99"
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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(OBJS) XSbench_header.h cudaCalcXs.o $(HEADERS)
	nvcc $(NVCC_FLAGS) cudaCalcXs.o $(OBJS) -o $@ $(LDFLAGS)

%.o : %.c $(HEADERS)
	$(GCC) $(NVCC_FLAGS) $(GCC_S_FLAGS) $(GCC_O_FLAGS) -o $@ -c $<

cudaCalcXs.o: cudaCalcXs.cu cudaHeader.h XSbench_header.h 
	nvcc  $(NVCC_FLAGS) $(GCC_S_FLAGS) -c cudaCalcXs.cu -o cudaCalcXs.o -I.

clean:
	rm -rf XSBench XSBench.dSYM *.o *.os *.ptx *.vir *.ccbak *.s *.so
edit:
	vim -p $(source) do_flops.c papi.c XSbench_header.h
run:
	./XSBench
bgqrun:
	qsub -t 10 -n 1 -O test XSBench
