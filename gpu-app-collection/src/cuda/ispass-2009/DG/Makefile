################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
###############################################################################

# Add source files here
EXECUTABLE	:= DG 
# Cuda source files (compiled with cudacc)
CUFILES		:= src/MaxwellsKernel3d.cu 
# C/C++ source files (compiled with gcc / c++)
CFILES		:= src/Mesh3d.c src/Utils.c src/LoadBalance3d.c src/FacePair3d.c src/ParallelPairs.c src/BuildMaps3d.c src/StartUp3d.c src/MaxwellsRun3d.c \
		src/MaxwellsMPI3d.c src/MaxwellsDriver3d.c
GPGPUSIM_ROOT   := ..
PWD := $(shell pwd)
SRCDIR  = ./src/
HDRDIR  = ./include

MPICMPOPTS := $(shell $(OPENMPI_BINDIR)mpicxx --showme:compile) # extract the required options to compile with mpich 
MPILNKOPTS := $(shell $(OPENMPI_BINDIR)mpicxx --showme:link) # extract the required options to link with mpich 

N ?= 6
HOSTNAME=$(shell hostname | awk '/aamodt/ {print 1;} /arch/ {print 1;}')
ifeq ($(HOSTNAME),1)
INCLUDES = -Dp_N=$(N) -DNDG3d -DCUDA -I/opt/local/include -I/usr/include/malloc -I$(HDRDIR) -I/opt/mpich/include
else
INCLUDES = -Dp_N=$(N) -DNDG3d -DCUDA -I/opt/local/include -I/usr/include/malloc -I$(HDRDIR)  
endif

NEWCC = $(OPENMPI_BINDIR)mpicc
NEWLD = $(OPENMPI_BINDIR)mpicxx
NEWCPP = $(OPENMPI_BINDIR)mpicxx
###############################################################################
# Rules and targets

include ../../common/common.mk

LINKLINE += $(MPILNKOPTS) -L$(PWD)/3rdParty/ParMetis-3.1/ -lparmetis -lmetis -lm 

