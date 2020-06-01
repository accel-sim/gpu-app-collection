# Makefile for src-parallel

# specify CUDA main directory
CUDA_HOME ?= $(CUDA_INSTALL_PATH)

SHELL = /bin/bash

### CoMD can be built in either double or single precision and with or
### without MPI.  Select desired precision and MPI here.

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
DO_MPI = OFF

### Set your desired C compiler and any necessary flags.  Note that CoMD
### uses some c99 features.  You can also set flags for optimization and
### specify paths to include files that the compiler can't find on its
### own.  If you need any -L or -l switches to get C standard libraries
### (such as -lm for the math library) put them in C_LIB.
CC = gcc
CFLAGS = -std=c99
OPTFLAGS = -g -O5
INCLUDES = 
C_LIB = -lm


### If you need to specify include paths, library paths, or link flags
### for MPI, put them here.  Put both -L and -l switches into MPI_LIB.
MPI_LIB =
MPI_INCLUDE =

### A place to specify any other include or library switches your
### platform requires.
OTHER_LIB = -L$(CUDA_HOME)/lib64 -lcudart
OTHER_INCLUDE = -I$(CUDA_HOME)/include


### CUDA targets
CUDA_HOME ?= $(CUDA_INSTALL_PATH)

# support sm20 & sm35
#NVCC_FLAGS += -gencode arch=compute_35,code=compute_35 --ptxas-options=-v -lineinfo
NVCC_FLAGS += -gencode arch=compute_20,code=compute_20 --ptxas-options=-v -lineinfo

# example paths set up for cuda, modify as needed
NVCC = $(CUDA_HOME)/bin/nvcc

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
NVCC_FLAGS += -DDOUBLE
else
NVCC_FLAGS += -DSINGLE
endif

NVCC_FLAGS += -O2

#########################################
### Below here, it is pitch black.  
### You are likely to be eaten by a grue.
##########################################

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .c .cu .o

.PHONY: DEFAULT clean distclean depend

BIN_DIR=../bin

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
CFLAGS += -DDOUBLE
else
CFLAGS += -DSINGLE
endif

# Set executable name and add includes & libraries for MPI if needed.
CoMD_VARIANT = CoMD-serial
ifeq ($(DO_MPI), ON)
CoMD_VARIANT = CoMD-mpi
INCLUDES += ${MPI_INCLUDE}
CFLAGS += -DDO_MPI
LDFLAGS += ${MPI_LIB}
endif
CoMD_EXE = ${BIN_DIR}/${CoMD_VARIANT}

LDFLAGS += ${C_LIB} ${OTHER_LIB}
CFLAGS  += ${OPTFLAGS} ${INCLUDES} ${OTHER_INCLUDE}

SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

CU_SOURCES=$(wildcard *.cu)
CU_OBJECTS=$(CU_SOURCES:.cu=.cu.o)

DEFAULT: ${CoMD_EXE}

%.cu.o: %.cu
	${NVCC} -c $< ${NVCC_FLAGS} ${ECX_FLAGS} -o $@

%.o: %.c
	${CC} ${CFLAGS} ${ECX_FLAGS} -c $< -o $@

${CoMD_EXE}: ${BIN_DIR} CoMD_info.h ${OBJECTS} ${CU_OBJECTS}
	${CC} ${CFLAGS} -o ${CoMD_EXE} ${OBJECTS} ${CU_OBJECTS} ${LDFLAGS}

CoMD_info.h: Makefile
	./generate_info_header ${CoMD_VARIANT} "$(CC)" "$(CFLAGS)" "$(LDFLAGS)"

${BIN_DIR}:
	@if [ ! -d ${BIN_DIR} ]; then mkdir -p ${BIN_DIR} ; fi

clean:
	rm -f *.o CoMD_info.h .depend

distclean: clean
	rm -f ${CoMD_EXE} .depend.bak
	rm -rf html latex

depend:
	touch .depend
	makedepend -f .depend -Y. --$(CFLAGS)-- $(SOURCES) 2> /dev/null
	makedepend -a -f .depend -Y. -o.cu.o --$(CFLAGS)-- $(CU_SOURCES) 2> /dev/null

-include .depend
